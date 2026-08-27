"""Cross-encoder reranking via Voyage (rerank-2.5).

Takes the RRF-fused candidate list and re-scores each candidate against the
raw query with a cross-encoder, which is far more precise than the first-stage
dense/sparse similarity. Returns hits sorted by relevance with a `rerank_score`
attached, optionally truncated to the strongest `top_k`.
"""

from src.config import settings

_client = None


def _get_client():
    global _client
    if _client is None:
        import voyageai

        _client = voyageai.Client(api_key=settings.voyage_api_key)
    return _client


def rerank(query: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
    """Re-rank `hits` (each must carry `text`) against `query`, best-first."""
    if not settings.voyage_api_key:
        raise RuntimeError("VOYAGE_API_KEY is required.")
    if not hits:
        return []

    documents = [hit["text"] or "" for hit in hits]
    result = _get_client().rerank(
        query,
        documents,
        model=settings.voyage_rerank_model,
        top_k=top_k,
    )

    reranked: list[dict] = []
    for item in result.results:
        hit = dict(hits[item.index])
        hit["rerank_score"] = float(item.relevance_score)
        reranked.append(hit)
    return reranked
