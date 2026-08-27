import argparse
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.answer import answer_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the hybrid RAG pipeline a question.")
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument("--rerank-top-k", type=int, default=8)
    parser.add_argument("--route-top-k", type=int, default=2)
    parser.add_argument("--no-parents", action="store_true", help="Skip parent expansion.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from .env")

    with psycopg.connect(database_url) as connection:
        register_vector(connection)
        result = answer_query(
            connection,
            args.question,
            rerank_top_k=args.rerank_top_k,
            route_top_k=args.route_top_k,
            expand_parents=not args.no_parents,
        )

    print("\n" + result["answer"] + "\n")
    print("Sources:")
    for citation, hit in zip(result["citations"], result["hits"]):
        label = (
            hit.get("section_title")
            or hit.get("metadata", {}).get("file_name")
            or hit["chunk_id"]
        )
        print(f"  {citation.label} {label} (domain={hit.get('domain')})")


if __name__ == "__main__":
    main()
