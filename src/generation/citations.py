from src.models import Citation


def build_citations(hits: list[dict]) -> list[Citation]:
    citations: list[Citation] = []
    for index, hit in enumerate(hits, start=1):
        citations.append(
            Citation(
                document_id=hit["document_id"],
                chunk_id=hit["chunk_id"],
                label=f"[{index}]",
                metadata=hit.get("metadata", {}),
            )
        )
    return citations
