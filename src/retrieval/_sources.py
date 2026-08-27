"""Shared spec of embedded retrieval-unit tables.

Mirrors router._EMBEDDED_SOURCES and scripts/index_chunks.TARGETS: the three
tables that carry dense_embedding/sparse_embedding columns. child_chunks has a
domain column; the hospital_* rollups (csv) carry domain in metadata instead.
Each entry is (table, id_col, text_col, domain_expr).
"""

# (table, id column, embedded text column, SQL expr that yields the domain)
SOURCES: tuple[tuple[str, str, str, str], ...] = (
    ("child_chunks", "chunk_id", "text_content", "domain"),
    ("hospital_profiles", "profile_id", "retrieval_text", "metadata->>'domain'"),
    ("hospital_category_docs", "category_doc_id", "retrieval_text", "metadata->>'domain'"),
)
