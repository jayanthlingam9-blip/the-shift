"""End-to-end RAG answer: hybrid retrieval -> grounded generation + citations.

Ties the retrieval pipeline (route -> sparse+semantic -> RRF -> parent
expansion -> rerank) to the generation layer, returning the answer alongside
the numbered citations and the raw contexts (handy for RAGAS, which scores the
answer against exactly these retrieved contexts).
"""

import psycopg

from src.generation.citations import build_citations
from src.generation.context_builder import build_context
from src.generation.gemini_generator import generate_answer
from src.retrieval.search import hybrid_search


def _filter_by_score(hits: list[dict], min_score: float) -> list[dict]:
    """Keep hits at/above the rerank-score floor, but never drop the top hit."""
    if not hits or min_score <= 0:
        return hits
    kept = [hit for hit in hits if hit.get("rerank_score", 1.0) >= min_score]
    return kept or hits[:1]


def answer_query(
    connection: psycopg.Connection,
    query: str,
    candidate_limit: int = 20,
    rerank_top_k: int = 8,
    route_top_k: int = 2,
    expand_parents: bool = True,
    context_min_score: float = 0.1,
) -> dict:
    """Answer `query` over the corpus. Returns answer, citations, contexts, hits.

    `context_min_score` drops reranked hits whose Voyage relevance score falls
    below the floor before they reach the prompt -- this removes weak distractor
    contexts (raising context precision) while keeping the strongest hit even if
    everything scored low.
    """
    hits = hybrid_search(
        connection,
        query,
        candidate_limit=candidate_limit,
        rerank_top_k=rerank_top_k,
        route_top_k=route_top_k,
        expand_parents=expand_parents,
    )
    hits = _filter_by_score(hits, context_min_score)

    context = build_context(hits)
    answer = generate_answer(query, context)
    citations = build_citations(hits)

    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "contexts": [hit.get("text", "") or "" for hit in hits],
        "hits": hits,
    }
