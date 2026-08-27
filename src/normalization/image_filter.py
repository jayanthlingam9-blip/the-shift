"""Pre-caption image triage.

Drops images that add no retrieval value before they ever reach Gemini:
  * boilerplate  - the same picture repeated across many pages (logos, running
                   headers/footers, watermarks)
  * decorative   - tiny icons or thin banner strips
  * near-dup     - perceptually identical copies (re-saved / layout vs embedded)

Exact duplicates are collapsed by SHA-256; near-duplicates by perceptual hash
(when Pillow + imagehash are installed). Everything dropped is recorded with a
reason so the decision is auditable rather than silent.
"""

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.config import PROJECT_ROOT

# Filenames look like "img_p12_3.jpg" (embedded) or "page_12_image_1_v2.jpg"
# (layout). Both encode the 1-based page number right after img_p / page_.
_PAGE_RE = re.compile(r"(?:img_p|page_)(\d+)", re.IGNORECASE)


@dataclass
class ImageInstance:
    filename: str
    path: Path
    page: int
    bbox: dict  # {"x","y","w","h"} in PDF points (same space as text items)
    category: str  # "embedded" | "layout"
    sha: str
    phash: str | None = None
    group_id: int = -1  # perceptual-dup group; members share one caption


@dataclass
class FilterResult:
    kept: list[ImageInstance] = field(default_factory=list)
    flagged: list[dict] = field(default_factory=list)

    @property
    def groups(self) -> dict[int, list[ImageInstance]]:
        out: dict[int, list[ImageInstance]] = {}
        for inst in self.kept:
            out.setdefault(inst.group_id, []).append(inst)
        return out


def _parse_page(filename: str) -> int:
    match = _PAGE_RE.search(filename)
    return int(match.group(1)) if match else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_photographic(
    path: Path,
    *,
    colorfulness_min: float,
    unique_min: float,
    white_max: float,
) -> bool:
    """Heuristic: True for natural photographs (people, scenery, marketing).

    Photos are color-rich and full-bleed; charts/diagrams/forms use flat fills
    and white space. Separates cleanly on colorfulness + colour diversity +
    background fraction, so real (even colourful) charts are kept.
    """
    try:
        import math

        from PIL import Image

        with Image.open(path) as img:
            small = img.convert("RGB").resize((64, 64))
        px = list(small.getdata())
        n = len(px)
        white = sum(1 for r, g, b in px if r > 235 and g > 235 and b > 235) / n
        unique = len({(r >> 4, g >> 4, b >> 4) for r, g, b in px}) / 4096
        rg = [r - g for r, g, b in px]
        yb = [0.5 * (r + g) - b for r, g, b in px]

        def _mean_std(values: list[float]) -> tuple[float, float]:
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            return mean, std

        mrg, srg = _mean_std(rg)
        myb, syb = _mean_std(yb)
        colorful = math.sqrt(srg**2 + syb**2) + 0.3 * math.sqrt(mrg**2 + myb**2)
        return colorful > colorfulness_min and unique > unique_min and white < white_max
    except Exception:
        return False  # never drop on read error; let captioning decide


def _phash(path: Path) -> str | None:
    try:
        import imagehash
        from PIL import Image

        with Image.open(path) as img:
            return str(imagehash.phash(img))
    except Exception:
        return None


def _hamming(a: str, b: str) -> int:
    # imagehash hex strings -> bit difference
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _iou(a: dict, b: dict) -> float:
    """Intersection-over-union of two bboxes ({x,y,w,h}, same coord space)."""
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    iy = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _area(inst: ImageInstance) -> float:
    return inst.bbox["w"] * inst.bbox["h"]


def _assign_phash_groups(instances: list[ImageInstance], threshold: int) -> None:
    """Single-linkage union of perceptually-similar images (transitive).

    Uses union-find so that A~B and B~C land A, B and C in one group even when
    A and C are just over the threshold — one caption per visual cluster.
    """
    parent = list(range(len(instances)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(len(instances)):
        if instances[i].phash is None:
            continue
        for j in range(i + 1, len(instances)):
            if instances[j].phash is None:
                continue
            if _hamming(instances[i].phash, instances[j].phash) <= threshold:
                union(i, j)

    # Normalise roots to compact, stable group ids.
    remap: dict[int, int] = {}
    for idx, inst in enumerate(instances):
        root = find(idx)
        gid = remap.setdefault(root, len(remap))
        inst.group_id = gid


def filter_images(
    images_meta: list[dict],
    image_dir: Path,
    *,
    boilerplate_min_pages: int = 5,
    min_dimension: float = 28.0,
    max_aspect: float = 7.0,
    phash_threshold: int = 6,
    overlap_iou: float = 0.6,
    prefer_category: str = "embedded",
    detect_photographic: bool = True,
    photo_colorfulness: float = 30.0,
    photo_unique: float = 0.04,
    photo_white_max: float = 0.22,
    collapse_near_dups: bool = True,
) -> FilterResult:
    """Triage parsed image metadata into kept instances and flagged drops.

    `images_meta` is ``images_content_metadata.images`` from the parser's
    ``*.images.json``. `image_dir` is where those files were downloaded.
    """
    result = FilterResult()

    # 1. Build instances for files that actually exist on disk.
    instances: list[ImageInstance] = []
    for meta in images_meta:
        filename = meta.get("filename")
        if not filename:
            continue
        path = image_dir / filename
        if not path.exists():
            result.flagged.append(
                {"filename": filename, "reason": "missing_on_disk", "page": _parse_page(filename)}
            )
            continue
        bbox = meta.get("bbox") or {}
        instances.append(
            ImageInstance(
                filename=filename,
                path=path,
                page=_parse_page(filename),
                bbox={k: float(bbox.get(k, 0) or 0) for k in ("x", "y", "w", "h")},
                category=meta.get("category", "embedded"),
                sha=_sha256(path),
            )
        )

    # 2. Boilerplate: one SHA spread across many distinct pages = logo/header.
    pages_per_sha: dict[str, set[int]] = {}
    for inst in instances:
        pages_per_sha.setdefault(inst.sha, set()).add(inst.page)
    boilerplate_shas = {
        sha for sha, pages in pages_per_sha.items() if len(pages) >= boilerplate_min_pages
    }

    survivors: list[ImageInstance] = []
    for inst in instances:
        w, h = inst.bbox["w"], inst.bbox["h"]
        smaller = min(w, h) if w and h else 0.0
        aspect = (max(w, h) / smaller) if smaller else float("inf")

        if inst.sha in boilerplate_shas:
            result.flagged.append(
                {
                    "filename": inst.filename,
                    "reason": "boilerplate_repeated",
                    "page": inst.page,
                    "sha": inst.sha,
                    "page_count": len(pages_per_sha[inst.sha]),
                }
            )
        elif smaller and (smaller < min_dimension or aspect > max_aspect):
            result.flagged.append(
                {
                    "filename": inst.filename,
                    "reason": "decorative_dimensions",
                    "page": inst.page,
                    "bbox": inst.bbox,
                }
            )
        elif detect_photographic and _is_photographic(
            inst.path,
            colorfulness_min=photo_colorfulness,
            unique_min=photo_unique,
            white_max=photo_white_max,
        ):
            result.flagged.append(
                {
                    "filename": inst.filename,
                    "reason": "photographic_content",
                    "page": inst.page,
                }
            )
        else:
            survivors.append(inst)

    # 3. Exact-dup collapse: keep one instance per SHA (drop the rest as dups).
    #    The SHA-keyed caption cache would dedup the API call anyway, but
    #    dropping here also avoids emitting redundant elements.
    seen_sha: dict[str, ImageInstance] = {}
    deduped: list[ImageInstance] = []
    for inst in survivors:
        if inst.sha in seen_sha:
            result.flagged.append(
                {
                    "filename": inst.filename,
                    "reason": "exact_duplicate",
                    "page": inst.page,
                    "sha": inst.sha,
                    "duplicate_of": seen_sha[inst.sha].filename,
                }
            )
            continue
        seen_sha[inst.sha] = inst
        deduped.append(inst)

    # 4. Same-page overlap dedup: the parser emits the same region twice (an
    #    "embedded" copy and a "layout" copy) with near-identical bboxes. Keep
    #    one per overlapping cluster. This is position-based, so charts that
    #    merely share a template across *different* pages are never merged.
    overlap_dropped: set[str] = set()
    by_page: dict[int, list[ImageInstance]] = defaultdict(list)
    for inst in deduped:
        by_page[inst.page].append(inst)
    for page_insts in by_page.values():
        for i in range(len(page_insts)):
            a = page_insts[i]
            if a.filename in overlap_dropped:
                continue
            for j in range(i + 1, len(page_insts)):
                b = page_insts[j]
                if b.filename in overlap_dropped:
                    continue
                if _iou(a.bbox, b.bbox) < overlap_iou:
                    continue
                # Keep preferred category, else the larger crop; drop the other.
                if a.category == prefer_category and b.category != prefer_category:
                    keep, drop = a, b
                elif b.category == prefer_category and a.category != prefer_category:
                    keep, drop = b, a
                else:
                    keep, drop = (a, b) if _area(a) >= _area(b) else (b, a)
                overlap_dropped.add(drop.filename)
                result.flagged.append(
                    {
                        "filename": drop.filename,
                        "reason": "overlap_duplicate",
                        "page": drop.page,
                        "iou": round(_iou(a.bbox, b.bbox), 2),
                        "duplicate_of": keep.filename,
                    }
                )
    deduped = [inst for inst in deduped if inst.filename not in overlap_dropped]

    # 5. Perceptual near-dup grouping.
    for inst in deduped:
        inst.phash = _phash(inst.path)
    _assign_phash_groups(deduped, phash_threshold)

    # 6. Collapse each perceptual group to a single image. The first member is
    #    kept; the rest are dropped outright (not sent to captioning).
    if collapse_near_dups:
        seen_group: dict[int, ImageInstance] = {}
        collapsed: list[ImageInstance] = []
        for inst in deduped:
            rep = seen_group.get(inst.group_id)
            if rep is None:
                seen_group[inst.group_id] = inst
                collapsed.append(inst)
            else:
                result.flagged.append(
                    {
                        "filename": inst.filename,
                        "reason": "near_duplicate",
                        "page": inst.page,
                        "phash": inst.phash,
                        "duplicate_of": rep.filename,
                    }
                )
        deduped = collapsed

    result.kept = deduped
    return result


def write_flag_manifest(result: FilterResult, manifest_path: Path) -> None:
    import json

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kept_count": len(result.kept),
        "flagged_count": len(result.flagged),
        "caption_group_count": len(result.groups),
        "flagged": result.flagged,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)
