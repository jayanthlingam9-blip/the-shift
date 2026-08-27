"""Sparse (lexical) retrieval over the sparsevec(30522) SPLADE columns.

Queries each embedded retrieval-unit table by SPLADE inner product
(sparsevec_ip HNSW) and merges the per-table results. pgvector's `<#>` returns
the *negative* inner product, so we order by it ascending (most similar first)
and flip the sign into `sparse_score` (larger = better) for readability.
Optionally scoped to the router-chosen domain(s).
"""

import psycopg
from pgvector import SparseVector
from pgvector.psycopg import register_vector

from src.config import settings
from src.retrieval._sources import SOURCES


def sparse_search(
    connection: psycopg.Connection,
    query_sparse_embedding: dict[int, float],
    domains: list[str] | None = None,
    limit: int = 20,
) -> list[dict]:
    """SPLADE inner-product nearest retrieval units to the query's sparse vector.

    Returns up to `limit` hits best-first, each a dict with a unified
    `chunk_id`, `source_table`, `domain`, `text`, `metadata`, and
    `sparse_score` (larger = stronger lexical match).
    """
    register_vector(connection)
    query_vector = SparseVector(query_sparse_embedding, settings.splade_vocab_size)

    hits: list[dict] = []
    with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        for table, id_col, text_col, domain_expr in SOURCES:
            sql = f"""
                SELECT
                    {id_col}::text      AS chunk_id,
                    document_id::text   AS document_id,
                    {text_col}          AS text,
                    {domain_expr}       AS domain,
                    metadata            AS metadata,
                    (sparse_embedding <#> %(qvec)s) AS neg_ip
                FROM {table}
                WHERE sparse_embedding IS NOT NULL
            """
            params: dict = {"qvec": query_vector, "limit": limit}
            if domains:
                sql += f" AND {domain_expr} = ANY(%(domains)s)"
                params["domains"] = domains
            sql += " ORDER BY neg_ip LIMIT %(limit)s;"

            cursor.execute(sql, params)
            for row in cursor.fetchall():
                row["source_table"] = table
                row["sparse_score"] = -float(row.pop("neg_ip"))
                row["metadata"] = row["metadata"] or {}
                hits.append(row)

    hits.sort(key=lambda h: h["sparse_score"], reverse=True)
    return hits[:limit]
