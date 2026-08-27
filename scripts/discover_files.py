import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "input"

sys.path.insert(0, str(PROJECT_ROOT))

from src.database.repository import upsert_document
from src.discovery.file_discovery import discover_documents


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    database_url = os.environ["DATABASE_URL"]

    documents = discover_documents(
        input_dir=INPUT_DIR,
        project_root=PROJECT_ROOT,
    )

    with psycopg.connect(database_url) as connection:
        for document in documents:
            upsert_document(connection, document)

            print(
                f"Discovered: {document.domain:<8} "
                f"{document.source_type:<5} "
                f"{document.file_name}"
            )

        connection.commit()

    print(f"\nTotal discovered sources: {len(documents)}")


if __name__ == "__main__":
    main()