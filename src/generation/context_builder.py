"""Build the grounded-context block handed to the generator.

Each hit is emitted as a numbered source ([1], [2], ...) so the numbering lines
up with the labels from generation.citations.build_citations -- the model is
told to cite those same markers, which keeps answer citations traceable back to
a specific retrieval unit.
"""


def _source_label(hit: dict) -> str:
    """Short provenance line for a hit, used as a context header."""
    parts = [hit.get("section_title") or hit.get("metadata", {}).get("file_name")]
    domain = hit.get("domain")
    if domain:
        parts.append(f"domain={domain}")
    return " | ".join(part for part in parts if part)


def build_context(hits: list[dict]) -> str:
    """Render hits as numbered source blocks for the prompt."""
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        header = _source_label(hit)
        prefix = f"[{index}]" + (f" {header}" if header else "")
        blocks.append(f"{prefix}\n{hit.get('text', '') or ''}")
    return "\n\n".join(blocks)
