"""Hybrid retrieval orchestrator: route -> (sparse + semantic) -> RRF -> rerank.

The full first-stage + second-stage pipeline:

  1. centroid-route the query to its domain(s)            (router.route)
  2. embed the query once dense and once sparse
  3. run dense (semantic) and sparse (SPLADE) search,
     each scoped to the routed domain(s)                  (dense/sparse_search)
  4. fuse the two ranked lists with reciprocal rank fusion (rrf)
  5. expand child chunks to their parent sections          (expand_to_parents)
  6. cross-encoder rerank the expanded candidates          (reranker.rerank)

Returns the top reranked hits, best-first.
"""

import psycopg

from src.embeddings.splade_embeddings import embed_sparse
from src.embeddings.voyage_embeddings import embed_texts
from src.retrieval.dense import dense_search
from src.retrieval.parent_expansion import expand_to_parents
from src.retrieval.reranker import rerank
from src.retrieval.router import route
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.sparse import sparse_search


def hybrid_search(
    connection: psycopg.Connection,
    query: str,
    candidate_limit: int = 20,
    rerank_top_k: int = 8,
    route_top_k: int = 2,
    expand_parents: bool = True,
) -> list[dict]:
    """Run the hybrid pipeline for `query` and return the top reranked hits.

    `candidate_limit` is the depth fetched from each first-stage retriever,
    `rerank_top_k` the number of hits kept after reranking, and `route_top_k`
    the number of domains the centroid router is allowed to fan out to. Default
    2 (soft routing): finance/legal centroids overlap, so hard routing (1) can
    discard the correct domain on boundary queries -- top-2 keeps a safety net
    and the reranker sorts out the cross-domain candidates. `route_top_k <= 0`
    disables routing and searches the whole corpus. `expand_parents` swaps child
    chunks for their parent sections before reranking (small-to-big).
    """
    domains: list[str] | None = None
    if route_top_k > 0:
        domains = [domain for domain, _distance in route(connection, query, route_top_k)]

    dense_vector = embed_texts([query], input_type="query")[0]
    sparse_vector = embed_sparse([query])[0]

    dense_hits = dense_search(connection, dense_vector, domains, candidate_limit)
    sparse_hits = sparse_search(connection, sparse_vector, domains, candidate_limit)

    fused = reciprocal_rank_fusion([dense_hits, sparse_hits])
    if not fused:
        return []

    if expand_parents:
        fused = expand_to_parents(connection, fused)

    return rerank(query, fused, top_k=rerank_top_k)
