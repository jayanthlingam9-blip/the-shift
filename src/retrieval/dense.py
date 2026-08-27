"""Dense (semantic) retrieval over the halfvec(1024) columns.

Queries each embedded retrieval-unit table by cosine distance (halfvec_cosine
HNSW) and merges the per-table results into one ranked list. Optionally scoped
to the domain(s) chosen by the centroid router so the search only touches the
relevant corpus.
"""

import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector

from src.retrieval._sources import SOURCES


def dense_search(
    connection: psycopg.Connection,
    query_embedding: list[float],
    domains: list[str] | None = None,
    limit: int = 20,
) -> list[dict]:
    """Cosine-nearest retrieval units to the query's dense vector.

    Returns up to `limit` hits best-first, each a dict with a unified
    `chunk_id` (the row's primary id), `source_table`, `domain`, `text`,
    `metadata`, and `dense_distance` (smaller = closer).
    """
    register_vector(connection)
    query_vector = HalfVector(query_embedding)

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
                    dense_embedding <=> %(qvec)s AS dense_distance
                FROM {table}
                WHERE dense_embedding IS NOT NULL
            """
            params: dict = {"qvec": query_vector, "limit": limit}
            if domains:
                sql += f" AND {domain_expr} = ANY(%(domains)s)"
                params["domains"] = domains
            sql += " ORDER BY dense_distance LIMIT %(limit)s;"

            cursor.execute(sql, params)
            for row in cursor.fetchall():
                row["source_table"] = table
                row["dense_distance"] = float(row["dense_distance"])
                row["metadata"] = row["metadata"] or {}
                hits.append(row)

    hits.sort(key=lambda h: h["dense_distance"])
    return hits[:limit]
