import hashlib
import json
import mimetypes
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config import PROJECT_ROOT, settings


CAPTION_CACHE_PATH = PROJECT_ROOT / "data" / "image_caption_cache.json"

CAPTION_PROMPT = """Create a concise, factual caption for this document image.
Focus on visible labels, chart/table meaning, forms, diagrams, and domain-specific facts.
If text is visible in the image, summarize the important text instead of transcribing everything.
Do not invent facts that are not visible. Keep the caption to 1-3 sentences."""


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _safe_project_path(path_value: str) -> Path | None:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        value = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def _context_lines(context: dict | None) -> list[str]:
    context = context or {}
    labels = {
        "section_title": "Section",
        "alt_text": "Alt text",
        "title": "Image title",
        "nearby_text_before": "Nearby text before",
        "nearby_text_after": "Nearby text after",
        "nearby_table_markdown": "Nearby table",
    }
    lines = []
    for key, label in labels.items():
        value = _clean_text(str(context.get(key) or ""))
        if value:
            lines.append(f"{label}: {value[:1200]}")
    return lines


def _build_prompt(context: dict | None) -> str:
    prompt = CAPTION_PROMPT
    lines = _context_lines(context)
    if lines:
        prompt = f"{prompt}\n\nDocument context:\n" + "\n".join(lines)
    return prompt


def _call_gemini(path: Path, prompt: str, mime_type: str, key: str, model: str) -> str:
    """Single cache-free Gemini vision call. Raises on failure (incl. rate limit)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=model,
        contents=[
            prompt,
            types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type),
        ],
    )
    text = _clean_text(getattr(response, "text", None))
    if not text:
        raise RuntimeError("empty caption response")
    return text


def fallback_caption(context: dict | None = None) -> str:
    lines = _context_lines(context)
    if lines:
        return "Image context: " + " ".join(lines)
    return "Image element without an available generated caption."


def caption_image(
    image_path: str,
    *,
    context: dict | None = None,
    cache_path: Path = CAPTION_CACHE_PATH,
    api_key: str | None = None,
    model: str | None = None,
) -> dict:
    """Generate or retrieve a caption for a local project image.

    Returns a dict with text plus status metadata. API failures intentionally
    fall back to contextual text so normalization can continue.
    """
    path = _safe_project_path(image_path)
    if path is None:
        return {
            "text": fallback_caption(context),
            "status": "invalid_path",
            "source": "fallback",
        }
    if not path.exists():
        return {
            "text": fallback_caption(context),
            "status": "missing_image",
            "source": "fallback",
        }

    image_hash = _file_sha256(path)
    caption_model = model or settings.gemini_model
    cache_key = f"{caption_model}:{image_hash}"
    cache = _load_cache(cache_path)
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and _clean_text(cached.get("text")):
        return {
            "text": _clean_text(cached["text"]),
            "status": "cached",
            "source": "gemini",
            "model": cached.get("model") or caption_model,
            "image_hash": image_hash,
        }

    key = api_key if api_key is not None else settings.google_api_key
    if not key:
        return {
            "text": fallback_caption(context),
            "status": "missing_api_key",
            "source": "fallback",
            "image_hash": image_hash,
        }

    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    prompt = _build_prompt(context)

    try:
        text = _call_gemini(path, prompt, mime_type, key, caption_model)
    except Exception as exc:
        return {
            "text": fallback_caption(context),
            "status": "generation_failed",
            "source": "fallback",
            "model": caption_model,
            "image_hash": image_hash,
            "error_type": exc.__class__.__name__,
        }

    cache[cache_key] = {
        "text": text,
        "model": caption_model,
        "image_hash": image_hash,
    }
    _save_cache(cache_path, cache)
    return {
        "text": text,
        "status": "generated",
        "source": "gemini",
        "model": caption_model,
        "image_hash": image_hash,
    }


# --------------------------------------------------------------------------- #
# Batch captioning with multi-key rotation
# --------------------------------------------------------------------------- #

_RATE_LIMIT_MARKERS = ("rate limit", "resource_exhausted", "quota", "429", "too many")


def _is_rate_limit_error(exc: Exception) -> bool:
    text = f"{getattr(exc, 'code', '')} {exc}".lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


class GeminiKeyPool:
    """Round-robin pool that rate-limits each key and cools down on 429s.

    acquire() blocks until some key is allowed to fire, reserves its next slot,
    and returns it. penalize() benches a key after a rate-limit error so the
    other keys keep working.
    """

    def __init__(self, keys: list[str], rpm_per_key: int):
        if not keys:
            raise ValueError("GeminiKeyPool requires at least one API key")
        self._keys = list(keys)
        self._min_interval = 60.0 / max(rpm_per_key, 1)
        self._lock = threading.Lock()
        self._next_free = {k: 0.0 for k in self._keys}      # earliest usable time
        self._cooldown_until = {k: 0.0 for k in self._keys}  # benched until

    def acquire(self) -> str:
        while True:
            with self._lock:
                now = time.monotonic()
                ready = [
                    (self._next_free[k], k)
                    for k in self._keys
                    if self._cooldown_until[k] <= now
                ]
                if ready:
                    next_free, key = min(ready)
                    start = max(now, next_free)
                    self._next_free[key] = start + self._min_interval
                    sleep_for = start - now
                else:
                    sleep_for = min(self._cooldown_until.values()) - now
                    key = None
            time.sleep(max(sleep_for, 0.0))
            if key is not None:
                return key

    def penalize(self, key: str, cooldown: float = 60.0) -> None:
        with self._lock:
            self._cooldown_until[key] = time.monotonic() + cooldown


def caption_images(
    images: list[dict],
    *,
    cache_path: Path = CAPTION_CACHE_PATH,
    api_keys: list[str] | None = None,
    model: str | None = None,
    rpm_per_key: int | None = None,
    max_workers_per_key: int | None = None,
    max_retries: int = 4,
    flush_every: int = 25,
    on_progress=None,
) -> dict[str, dict]:
    """Caption many images concurrently, rotating across all Gemini keys.

    Each item in `images` is a dict with at least ``{"image_path": str}`` and an
    optional ``{"context": dict}``. Returns a mapping of image_path -> result
    dict (same shape as ``caption_image``). The shared cache file is owned here
    and flushed under a lock, so concurrent workers never corrupt it.
    """
    keys = api_keys if api_keys is not None else settings.gemini_api_keys
    if not keys:
        return {
            item["image_path"]: {
                "text": fallback_caption(item.get("context")),
                "status": "missing_api_key",
                "source": "fallback",
            }
            for item in images
        }

    caption_model = model or settings.gemini_model
    rpm = rpm_per_key if rpm_per_key is not None else settings.gemini_rpm_per_key
    per_key = (
        max_workers_per_key
        if max_workers_per_key is not None
        else settings.gemini_max_workers_per_key
    )
    pool = GeminiKeyPool(keys, rpm)
    cache = _load_cache(cache_path)
    cache_lock = threading.Lock()
    results: dict[str, dict] = {}
    completed = 0

    def _flush() -> None:
        with cache_lock:
            _save_cache(cache_path, dict(cache))

    def _worker(item: dict) -> tuple[str, dict]:
        image_path = item["image_path"]
        context = item.get("context")
        path = _safe_project_path(image_path)
        if path is None:
            return image_path, {
                "text": fallback_caption(context),
                "status": "invalid_path",
                "source": "fallback",
            }
        if not path.exists():
            return image_path, {
                "text": fallback_caption(context),
                "status": "missing_image",
                "source": "fallback",
            }

        image_hash = _file_sha256(path)
        cache_key = f"{caption_model}:{image_hash}"
        with cache_lock:
            cached = cache.get(cache_key)
        if isinstance(cached, dict) and _clean_text(cached.get("text")):
            return image_path, {
                "text": _clean_text(cached["text"]),
                "status": "cached",
                "source": "gemini",
                "model": cached.get("model") or caption_model,
                "image_hash": image_hash,
            }

        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        prompt = _build_prompt(context)
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            key = pool.acquire()
            try:
                text = _call_gemini(path, prompt, mime_type, key, caption_model)
            except Exception as exc:  # noqa: BLE001 - classified below
                last_exc = exc
                if _is_rate_limit_error(exc):
                    pool.penalize(key)
                else:
                    time.sleep(min(2 ** attempt, 8))
                continue
            with cache_lock:
                cache[cache_key] = {
                    "text": text,
                    "model": caption_model,
                    "image_hash": image_hash,
                }
            return image_path, {
                "text": text,
                "status": "generated",
                "source": "gemini",
                "model": caption_model,
                "image_hash": image_hash,
            }

        return image_path, {
            "text": fallback_caption(context),
            "status": "generation_failed",
            "source": "fallback",
            "model": caption_model,
            "image_hash": image_hash,
            "error_type": last_exc.__class__.__name__ if last_exc else "Unknown",
        }

    max_workers = max(len(keys) * per_key, 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, item): item for item in images}
        for future in as_completed(futures):
            image_path, result = future.result()
            results[image_path] = result
            completed += 1
            if on_progress is not None:
                on_progress(completed, len(images), image_path, result)
            if completed % flush_every == 0:
                _flush()

    _flush()
    return results
