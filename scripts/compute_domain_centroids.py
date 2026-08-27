import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.router import refresh_centroids


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from .env")

    with psycopg.connect(database_url) as connection:
        rows = refresh_centroids(connection)

    print("Domain centroids computed:")
    for domain, count in rows:
        print(f"  {domain:10} <- {count:,} vectors")


if __name__ == "__main__":
    main()
