"""Centroid-based domain routing.

Each domain gets one centroid = the mean dense vector of every embedded
retrieval unit in that domain (child_chunks + the two hospital_* rollups).
At query time the query vector is compared to each centroid by cosine
distance and search is routed to the nearest domain.
"""

import psycopg
from pgvector import HalfVector
from pgvector.psycopg import register_vector

from src.embeddings.voyage_embeddings import embed_texts

# Every table holding embedded retrieval units, with how its domain is read.
# child_chunks has a domain column; the hospital_* rollups (csv) carry it in
# metadata instead.
_EMBEDDED_SOURCES = (
    ("child_chunks", "domain"),
    ("hospital_profiles", "metadata->>'domain'"),
    ("hospital_category_docs", "metadata->>'domain'"),
)


def refresh_centroids(connection: psycopg.Connection) -> list[tuple[str, int]]:
    """(Re)compute and store one centroid per domain. Returns (domain, count)."""
    union = "\n            UNION ALL\n".join(
        f"SELECT {domain_expr} AS domain, dense_embedding AS v FROM {table} "
        f"WHERE dense_embedding IS NOT NULL"
        for table, domain_expr in _EMBEDDED_SOURCES
    )
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO domain_centroids (domain, centroid, source_count)
            SELECT domain, avg(v::vector)::halfvec(1024), count(*)
            FROM (
                {union}
            ) s
            GROUP BY domain
            ON CONFLICT (domain) DO UPDATE SET
                centroid = EXCLUDED.centroid,
                source_count = EXCLUDED.source_count,
                updated_at = NOW();
            """
        )
        cursor.execute(
            "SELECT domain, source_count FROM domain_centroids ORDER BY domain;"
        )
        rows = cursor.fetchall()
    connection.commit()
    return rows


def route(
    connection: psycopg.Connection,
    query_text: str,
    top_k: int = 1,
) -> list[tuple[str, float]]:
    """Rank domains for a query by cosine distance to their centroid.

    Returns [(domain, cosine_distance), ...] best-first. The caller searches
    the top-1 domain (or top_k for soft routing).
    """
    register_vector(connection)
    query_vector = HalfVector(embed_texts([query_text], input_type="query")[0])
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT domain, (centroid <=> %s) AS distance
            FROM domain_centroids
            ORDER BY distance
            LIMIT %s;
            """,
            (query_vector, top_k),
        )
        return [(domain, float(distance)) for domain, distance in cursor.fetchall()]
