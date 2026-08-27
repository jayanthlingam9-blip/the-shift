"""Normalization: turn parsed sources into ordered NormalizedElements.

Each source type is flattened into a single reading-order stream of
NormalizedElements and written to ``data/normalized/<domain>/<source_type>/``,
mirroring the parsed tree.

  * PDF  - text/heading/table/list items from ``*.structured.json`` (already in
           reading order) with captioned images spliced back in at the exact y
           position they occupied (junk images dropped first by image_filter).
  * HTML - pages -> sections -> elements, ordered by section then element
           ``sequence``; section headings are emitted inline.
  * CSV  - pre-chunked ``*.jsonl`` records, one element per row.
"""

import json
from pathlib import Path

from src.config import PROJECT_ROOT
from src.models import Domain, NormalizedElement
from src.normalization import image_filter
from src.normalization.image_captioning import caption_images, fallback_caption

# Structured-JSON item types that carry retrievable content. Everything else
# (header, footer, link, and the parser's diagnostic types) is dropped.
CONTENT_TYPES = {"heading", "text", "list", "table", "code"}
TEXTLIKE_TYPES = {"text", "list", "code"}

PARSED_ROOT = PROJECT_ROOT / "data" / "parsed"
NORMALIZED_ROOT = PROJECT_ROOT / "data" / "normalized"


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _clean(value) -> str:
    return " ".join(str(value or "").split())


def _output_path(domain: str, source_type: str, document_id: str) -> Path:
    return NORMALIZED_ROOT / domain / source_type / f"{document_id}.json"


def _save(elements: list[NormalizedElement], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([e.model_dump(mode="json") for e in elements], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _item_text(item: dict) -> str:
    return _clean(item.get("md") or item.get("value") or "")


def _item_y(item: dict) -> float:
    bbox = item.get("bbox") or []
    if bbox and isinstance(bbox, list):
        return float(bbox[0].get("y", 0) or 0)
    return 0.0


def _content_items(page: dict) -> list[dict]:
    out = []
    for item in page.get("items", []):
        if item.get("type") not in CONTENT_TYPES:
            continue
        text = _item_text(item)
        if not text:
            continue
        out.append({"type": item["type"], "text": text, "y": _item_y(item)})
    return out


def _image_context(content: list[dict], image_y: float) -> dict:
    above = [c for c in content if c["y"] <= image_y]
    below = [c for c in content if c["y"] > image_y]
    section_title = next((c["text"] for c in reversed(above) if c["type"] == "heading"), "")
    before = next((c["text"] for c in reversed(above) if c["type"] in TEXTLIKE_TYPES), "")
    after = next((c["text"] for c in below if c["type"] in TEXTLIKE_TYPES), "")
    nearest = None
    if above and above[-1]["type"] == "table":
        nearest = above[-1]["text"]
    elif below and below[0]["type"] == "table":
        nearest = below[0]["text"]
    context = {
        "section_title": section_title,
        "nearby_text_before": before,
        "nearby_text_after": after,
    }
    if nearest:
        context["nearby_table_markdown"] = nearest
    return context


def _normalize_pdf(
    domain: str, document_id: str, *, caption: bool, save: bool
) -> list[NormalizedElement]:
    parsed_dir = PARSED_ROOT / domain / "pdf"
    image_dir = PROJECT_ROOT / "data" / "images" / domain / document_id

    structured = json.loads(
        (parsed_dir / f"{document_id}.structured.json").read_text(encoding="utf-8")
    )
    images_doc = json.loads(
        (parsed_dir / f"{document_id}.images.json").read_text(encoding="utf-8")
    )
    images_meta = (images_doc.get("images_content_metadata") or {}).get("images", [])

    triage = image_filter.filter_images(images_meta, image_dir)
    content_by_page = {p["page_number"]: _content_items(p) for p in structured["pages"]}

    # caption one representative per dedup group (kept == groups after collapse)
    groups = triage.groups
    reps = {gid: members[0] for gid, members in groups.items()}
    rep_context = {
        gid: _image_context(content_by_page.get(rep.page, []), rep.bbox["y"])
        for gid, rep in reps.items()
    }
    captions: dict[int, dict] = {}
    if caption and reps:
        batch = [{"image_path": str(rep.path), "context": rep_context[gid]} for gid, rep in reps.items()]
        results = caption_images(batch)
        path_to_gid = {str(rep.path): gid for gid, rep in reps.items()}
        for path, res in results.items():
            captions[path_to_gid[path]] = res
    else:
        for gid, ctx in rep_context.items():
            captions[gid] = {"text": fallback_caption(ctx), "status": "skipped", "source": "fallback"}

    images_by_page: dict[int, list] = {}
    for inst in triage.kept:
        images_by_page.setdefault(inst.page, []).append(inst)

    elements: list[NormalizedElement] = []
    order = 0
    domain_enum = Domain(domain)
    for page in structured["pages"]:
        page_no = page["page_number"]
        content = content_by_page.get(page_no, [])
        page_images = sorted(images_by_page.get(page_no, []), key=lambda i: (i.bbox["y"], i.bbox["x"]))
        combined: list[tuple[float, str, object]] = [
            (float(idx), "content", item) for idx, item in enumerate(content)
        ]
        for j, inst in enumerate(page_images):
            below_count = sum(1 for c in content if c["y"] < inst.bbox["y"])
            combined.append((below_count - 0.5 + j * 1e-6, "image", inst))
        combined.sort(key=lambda t: t[0])

        for _, kind, payload in combined:
            if kind == "content":
                item = payload  # type: ignore[assignment]
                elements.append(
                    NormalizedElement(
                        element_id=f"{document_id}:p{page_no}:{order:05d}",
                        document_id=document_id,
                        domain=domain_enum,
                        element_type=item["type"],
                        text=item["text"],
                        metadata={"source_type": "pdf", "page_number": page_no, "order": order},
                    )
                )
            else:
                inst = payload  # type: ignore[assignment]
                cap = captions.get(inst.group_id, {})
                elements.append(
                    NormalizedElement(
                        element_id=f"{document_id}:p{page_no}:{Path(inst.filename).stem}",
                        document_id=document_id,
                        domain=domain_enum,
                        element_type="image",
                        text=cap.get("text") or fallback_caption(None),
                        metadata={"source_type": "pdf", "page_number": page_no, "order": order},
                    )
                )
            order += 1

    if save:
        _save(elements, _output_path(domain, "pdf", document_id))
    return elements


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def _normalize_html(
    domain: str, document_id: str, *, save: bool, **_
) -> list[NormalizedElement]:
    parsed_dir = PARSED_ROOT / domain / "html"
    structured = json.loads(
        (parsed_dir / f"{document_id}.structured.json").read_text(encoding="utf-8")
    )

    elements: list[NormalizedElement] = []
    order = 0
    domain_enum = Domain(domain)
    for page in structured.get("pages", []):
        page_no = page.get("page_number", 1)
        for section in page.get("sections", []):
            heading = _clean(section.get("heading"))
            level = section.get("heading_level") or 0
            if heading and level >= 1:
                elements.append(
                    NormalizedElement(
                        element_id=f"{document_id}:p{page_no}:{order:05d}",
                        document_id=document_id,
                        domain=domain_enum,
                        element_type="heading",
                        text=heading,
                        metadata={"source_type": "html", "page_number": page_no, "order": order},
                    )
                )
                order += 1
            for el in sorted(section.get("elements", []), key=lambda e: e.get("sequence", 0)):
                text = _clean(el.get("text"))
                if not text:  # empty image placeholders etc.
                    continue
                elements.append(
                    NormalizedElement(
                        element_id=f"{document_id}:p{page_no}:{order:05d}",
                        document_id=document_id,
                        domain=domain_enum,
                        element_type=el.get("type", "text"),
                        text=text,
                        metadata={"source_type": "html", "page_number": page_no, "order": order},
                    )
                )
                order += 1

    if save:
        _save(elements, _output_path(domain, "html", document_id))
    return elements


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def _normalize_csv(
    domain: str, document_id: str, *, save: bool, **_
) -> list[NormalizedElement]:
    parsed_dir = PARSED_ROOT / domain / "csv"
    elements: list[NormalizedElement] = []
    order = 0
    domain_enum = Domain(domain)

    for jsonl in sorted(parsed_dir.glob(f"{document_id}.*.jsonl")):
        with jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = _clean(rec.get("retrieval_text"))
                if not text:
                    continue
                elements.append(
                    NormalizedElement(
                        element_id=f"{document_id}:{order:05d}",
                        document_id=document_id,
                        domain=domain_enum,
                        element_type="record",
                        text=text,
                        metadata={"source_type": "csv", "page_number": None, "order": order},
                    )
                )
                order += 1

    if save:
        _save(elements, _output_path(domain, "csv", document_id))
    return elements


# --------------------------------------------------------------------------- #
# dispatch + discovery
# --------------------------------------------------------------------------- #
_NORMALIZERS = {
    "pdf": _normalize_pdf,
    "html": _normalize_html,
    "csv": _normalize_csv,
}


def normalize_document(
    domain: str,
    document_id: str,
    source_type: str,
    *,
    caption: bool = True,
    save: bool = True,
) -> list[NormalizedElement]:
    """Normalize one document of any supported source type."""
    normalizer = _NORMALIZERS.get(source_type)
    if normalizer is None:
        raise ValueError(f"Unsupported source_type: {source_type}")
    return normalizer(domain, document_id, caption=caption, save=save)


def _discover_documents() -> list[tuple[str, str, str]]:
    """(domain, document_id, source_type) for every parsed document."""
    out: list[tuple[str, str, str]] = []
    for domain in (d.value for d in Domain):
        # pdf / html share the structured.json convention
        for source_type in ("pdf", "html"):
            src_dir = PARSED_ROOT / domain / source_type
            if not src_dir.exists():
                continue
            for path in sorted(src_dir.glob("*.structured.json")):
                out.append((domain, path.name[: -len(".structured.json")], source_type))
        # csv datasets are keyed by their metadata.json
        csv_dir = PARSED_ROOT / domain / "csv"
        if csv_dir.exists():
            for path in sorted(csv_dir.glob("*.metadata.json")):
                out.append((domain, path.name[: -len(".metadata.json")], "csv"))
    return out


def normalize_all(*, caption: bool = True, save: bool = True) -> dict[str, int]:
    """Normalize every discovered document; returns element counts per doc."""
    counts: dict[str, int] = {}
    for domain, document_id, source_type in _discover_documents():
        elements = normalize_document(domain, document_id, source_type, caption=caption, save=save)
        counts[f"{domain}/{source_type}/{document_id}"] = len(elements)
    return counts
