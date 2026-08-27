import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.repository import (
    fetch_documents_for_parsing,
    mark_document_failed,
    mark_document_parsed,
    update_document_status,
)
from src.parsing.csv_parser import parse_hcahps_csv


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from .env")

    with psycopg.connect(database_url) as connection:
        documents = fetch_documents_for_parsing(connection, source_type="csv")

    if not documents:
        print("No discovered, parsing, or failed CSV documents need parsing.")
        return

    for document in documents:
        print(f"\nParsing CSV: {document['file_name']}")

        try:
            with psycopg.connect(database_url) as connection:
                update_document_status(
                    connection,
                    document["document_id"],
                    "parsing",
                )
                connection.commit()

                outputs = parse_hcahps_csv(
                    connection=connection,
                    document_id=document["document_id"],
                    file_path=document["file_path"],
                    domain=document["domain"],
                )

                mark_document_parsed(
                    connection,
                    document["document_id"],
                    outputs,
                )
                connection.commit()

            print(
                f"Parsed: {document['file_name']} | "
                f"rows={outputs['row_count']:,} | "
                f"hospitals={outputs['hospital_profile_count']:,} | "
                f"category_docs={outputs['hospital_category_doc_count']:,}"
            )
        except Exception as exc:
            with psycopg.connect(database_url) as connection:
                mark_document_failed(
                    connection,
                    document["document_id"],
                    exc,
                )
                connection.commit()
            print(f"Failed: {document['file_name']} | {exc}")
            raise


if __name__ == "__main__":
    main()