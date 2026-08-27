import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector import HalfVector, SparseVector
from pgvector.psycopg import register_vector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.embeddings.splade_embeddings import embed_sparse
from src.embeddings.voyage_embeddings import embed_texts

# (table, id column, text column to embed) for each retrieval-unit type.
TARGETS = [
    ("child_chunks", "chunk_id", "text_content"),
    ("hospital_profiles", "profile_id", "retrieval_text"),
    ("hospital_category_docs", "category_doc_id", "retrieval_text"),
]


def _fetch_unembedded(
    connection: psycopg.Connection,
    table: str,
    id_col: str,
    text_col: str,
    limit: int,
) -> list[tuple]:
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {id_col}, {text_col}
            FROM {table}
            WHERE dense_embedding IS NULL
              AND {text_col} IS NOT NULL
              AND length(trim({text_col})) > 0
            ORDER BY {id_col}
            LIMIT %s;
            """,
            (limit,),
        )
        return cursor.fetchall()


def _embed_table(connection: psycopg.Connection, table: str, id_col: str, text_col: str) -> int:
    done = 0
    while True:
        rows = _fetch_unembedded(
            connection, table, id_col, text_col, settings.embed_batch_size
        )
        if not rows:
            break

        ids = [row[0] for row in rows]
        texts = [row[1] for row in rows]

        dense = embed_texts(texts, input_type="document")
        sparse = embed_sparse(texts)

        updates = [
            (
                HalfVector(dense[i]),
                SparseVector(sparse[i], settings.splade_vocab_size),
                ids[i],
            )
            for i in range(len(ids))
        ]
        with connection.cursor() as cursor:
            cursor.executemany(
                f"""
                UPDATE {table}
                SET dense_embedding = %s,
                    sparse_embedding = %s
                WHERE {id_col} = %s;
                """,
                updates,
            )
        connection.commit()

        done += len(rows)
        print(f"  {table}: {done:,} embedded")
    return done


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from .env")

    for table, id_col, text_col in TARGETS:
        print(f"\nEmbedding {table} (text={text_col}) ...")
        with psycopg.connect(database_url) as connection:
            register_vector(connection)
            total = _embed_table(connection, table, id_col, text_col)
        print(f"  {table}: done ({total:,} rows)")

    print("\nIndexing complete.")


if __name__ == "__main__":
    main()
