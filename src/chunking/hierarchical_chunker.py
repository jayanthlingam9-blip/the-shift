"""Hierarchical parent/child chunking for PDF/HTML normalized elements.

Sections are delimited by ``heading`` elements; each heading opens a parent
section that accumulates the following content. Children are token-bounded
windows within a section (small-to-big retrieval): precise children improve
context precision, while coherent parents -- retrieved by expanding
``parent_id`` -- improve context recall and faithfulness.

Tables are kept whole (never split mid-row). Image captions flow inline with
surrounding narrative so we avoid isolated low-content chunks that dilute
retrieval relevance. PDF headings carry markdown ``#`` levels, so
``section_path`` is a true nested breadcrumb; HTML headings (no ``#``) flatten
to sequential top-level sections.
"""

from __future__ import annotations

import re
import uuid

import tiktoken

from src.models import Chunk, NormalizedElement, ParentSection

_ENCODER = tiktoken.get_encoding("cl100k_base")

_HEADING = "heading"
_TABLE = "table"
_IMAGE = "image"


def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _heading_level(text: str) -> int:
    match = re.match(r"^(#{1,6})\s", text)
    return len(match.group(1)) if match else 1


def _clean_title(text: str) -> str:
    return text.lstrip("#").strip()


def _det_uuid(*parts: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, ":".join(parts)))


def _split_oversized(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split a single element whose text exceeds ``max_tokens`` into windows."""
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    windows: list[str] = []
    step = max(max_tokens - overlap_tokens, 1)
    start = 0
    while start < len(tokens):
        windows.append(_ENCODER.decode(tokens[start : start + max_tokens]))
        start += step
    return windows


_TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def _split_table(markdown: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split an oversized markdown table on row boundaries, repeating the header.

    Each window stays self-describing (header + a slice of rows), so it embeds
    and retrieves cleanly. Falls back to token windows if it isn't a parseable
    markdown table.
    """
    if _count_tokens(markdown) <= max_tokens:
        return [markdown]

    lines = markdown.splitlines()
    if len(lines) >= 2 and _TABLE_SEPARATOR.match(lines[1]):
        header, body = lines[:2], lines[2:]
    elif lines and lines[0].lstrip().startswith("|"):
        header, body = lines[:1], lines[1:]
    else:
        return _split_oversized(markdown, max_tokens, overlap_tokens)

    if not body:
        return _split_oversized(markdown, max_tokens, overlap_tokens)

    header_text = "\n".join(header)
    header_tokens = _count_tokens(header_text)
    windows: list[str] = []
    current: list[str] = []
    current_tokens = header_tokens
    for row in body:
        row_tokens = _count_tokens(row)
        if current and current_tokens + row_tokens > max_tokens:
            windows.append("\n".join(header + current))
            current = []
            current_tokens = header_tokens
        current.append(row)
        current_tokens += row_tokens
    if current:
        windows.append("\n".join(header + current))
    return windows


def _expand_oversized_elements(
    elements: list[NormalizedElement],
    budget: int,
    overlap_tokens: int,
) -> list[NormalizedElement]:
    """Pre-split any single element larger than the parent budget.

    Keeps parents bounded even when the parser emits one enormous table (or
    text block) as a single element. Headings/captions are short and untouched,
    so image element ids -- used to resolve image_paths -- stay intact.
    """
    expanded: list[NormalizedElement] = []
    for element in elements:
        text = element.text or ""
        if _count_tokens(text) <= budget:
            expanded.append(element)
            continue
        pieces = (
            _split_table(text, budget, overlap_tokens)
            if element.element_type == _TABLE
            else _split_oversized(text, budget, overlap_tokens)
        )
        for piece_index, piece in enumerate(pieces):
            expanded.append(
                element.model_copy(
                    update={
                        "text": piece,
                        "element_id": f"{element.element_id}#p{piece_index}",
                    }
                )
            )
    return expanded


def _segment_sections(elements: list[NormalizedElement]) -> list[dict]:
    """Group ordered elements into heading-delimited sections.

    Maintains a heading stack so each section records its nested breadcrumb
    (``path``). Heading-only sections are dropped, but their titles remain on
    the stack so deeper sections keep the full breadcrumb.
    """
    sections: list[dict] = []
    stack: list[tuple[int, str]] = []
    current: dict | None = None

    for element in elements:
        if element.element_type == _HEADING:
            level = _heading_level(element.text)
            title = _clean_title(element.text)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            if current is not None:
                sections.append(current)
            current = {"title": title, "path": [t for _, t in stack], "content": []}
        else:
            if current is None:
                current = {"title": None, "path": [], "content": []}
            current["content"].append(element)

    if current is not None:
        sections.append(current)

    return [section for section in sections if section["content"]]


def _split_section(section: dict, budget: int) -> list[dict]:
    """Split an oversized section into bounded parts at element boundaries.

    Keeps the same title/breadcrumb so retrieval context is preserved, but
    prevents heading-less or very long sections from becoming one huge parent.
    """
    parts: list[list[NormalizedElement]] = []
    current: list[NormalizedElement] = []
    current_tokens = 0
    for element in section["content"]:
        tokens = _count_tokens(element.text or "")
        if current and current_tokens + tokens > budget:
            parts.append(current)
            current = []
            current_tokens = 0
        current.append(element)
        current_tokens += tokens
    if current:
        parts.append(current)

    return [
        {"title": section["title"], "path": section["path"], "content": part}
        for part in parts
    ]


def create_chunks(
    elements: list[NormalizedElement],
    *,
    document_id,
    source_type: str,
    file_name: str,
    domain=None,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
    max_tokens: int = 800,
    parent_max_tokens: int = 2000,
) -> tuple[list[ParentSection], list[Chunk]]:
    """Turn one document's normalized elements into parents + child chunks."""
    if domain is None:
        if not elements:
            return [], []
        domain = elements[0].domain

    document_id = str(document_id)
    elements = _expand_oversized_elements(elements, parent_max_tokens, overlap_tokens)
    parents: list[ParentSection] = []
    children: list[Chunk] = []
    parent_index = 0

    for section in _segment_sections(elements):
        for part in _split_section(section, parent_max_tokens):
            parent_id = _det_uuid(document_id, "parent", str(parent_index))
            content = part["content"]
            parents.append(
                ParentSection(
                    parent_id=parent_id,
                    document_id=document_id,
                    domain=domain,
                    source_type=source_type,
                    section_title=part["title"],
                    section_path=part["path"],
                    page_numbers=_pages(content),
                    parent_text="\n\n".join(e.text for e in content if e.text),
                    metadata={"file_name": file_name, "section_index": parent_index},
                )
            )
            children.extend(
                _build_children(
                    part,
                    parent_id=parent_id,
                    document_id=document_id,
                    source_type=source_type,
                    file_name=file_name,
                    domain=domain,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    max_tokens=max_tokens,
                )
            )
            parent_index += 1

    return parents, children


def _pages(elements: list[NormalizedElement]) -> list[int]:
    pages = {
        e.metadata.get("page_number")
        for e in elements
        if e.metadata.get("page_number") is not None
    }
    return sorted(pages)


def _build_children(
    section: dict,
    *,
    parent_id: str,
    document_id: str,
    source_type: str,
    file_name: str,
    domain,
    target_tokens: int,
    overlap_tokens: int,
    max_tokens: int,
) -> list[Chunk]:
    children: list[Chunk] = []
    # Shared counter so table and text chunks get unique, ordered ids.
    index = 0

    def make_chunk(*, text, pages, modalities, image_refs, table_markdown=None):
        nonlocal index
        meta = {"file_name": file_name, "chunk_index": index}
        if image_refs:
            meta["image_refs"] = list(image_refs)
        chunk = Chunk(
            chunk_id=_det_uuid(parent_id, "child", str(index)),
            parent_id=parent_id,
            document_id=document_id,
            domain=domain,
            source_type=source_type,
            file_name=file_name,
            section_title=section["title"],
            page_numbers=sorted(p for p in pages if p is not None),
            modalities=sorted(modalities) or ["text"],
            text_content=text,
            table_markdown=table_markdown,
            metadata=meta,
        )
        index += 1
        return chunk

    # Split content into ordered segments: text runs and standalone tables.
    segments: list[tuple[str, object]] = []
    run: list[NormalizedElement] = []
    for element in section["content"]:
        if element.element_type == _TABLE:
            if run:
                segments.append(("text", run))
                run = []
            if (element.text or "").strip():
                segments.append(("table", element))
        elif (element.text or "").strip():
            run.append(element)
    if run:
        segments.append(("text", run))

    for kind, payload in segments:
        if kind == "table":
            element = payload  # type: ignore[assignment]
            page = element.metadata.get("page_number")
            for window in _split_table(element.text, max_tokens, overlap_tokens):
                children.append(
                    make_chunk(
                        text=window,
                        pages={page},
                        modalities={"table"},
                        image_refs=[],
                        table_markdown=window,
                    )
                )
            continue

        # Expand each element into token-bounded units, preserving page/modality.
        units: list[tuple[str, object, str, str | None, int]] = []
        for element in payload:  # type: ignore[assignment]
            page = element.metadata.get("page_number")
            is_image = element.element_type == _IMAGE
            modality = "image" if is_image else "text"
            ref = element.element_id if is_image else None
            for window in _split_oversized(element.text, max_tokens, overlap_tokens):
                units.append((window, page, modality, ref, _count_tokens(window)))

        buffer: list[tuple] = []
        buffer_tokens = 0
        for unit in units:
            if buffer and buffer_tokens + unit[4] > target_tokens:
                children.append(_emit(make_chunk, buffer))
                buffer, buffer_tokens = _overlap_tail(buffer, overlap_tokens)
            buffer.append(unit)
            buffer_tokens += unit[4]
        if buffer:
            children.append(_emit(make_chunk, buffer))

    return children


def _emit(make_chunk, units: list[tuple]) -> Chunk:
    return make_chunk(
        text="\n\n".join(u[0] for u in units).strip(),
        pages={u[1] for u in units},
        modalities={u[2] for u in units},
        image_refs=[u[3] for u in units if u[3]],
    )


def _overlap_tail(units: list[tuple], budget: int) -> tuple[list[tuple], int]:
    """Trailing units (re-included in the next chunk) summing up to ``budget``."""
    if budget <= 0:
        return [], 0
    tail: list[tuple] = []
    total = 0
    for unit in reversed(units):
        if total >= budget and tail:
            break
        tail.insert(0, unit)
        total += unit[4]
    return tail, total
