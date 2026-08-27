import argparse
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
from src.parsing.html_parser import (
    find_existing_html_outputs,
    parse_html_crawl,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl and deterministically structure discovered HTML sources."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pending HTML sources without crawling or updating Neon.",
    )
    return parser.parse_args()


def save_status(database_url: str, document_id, status: str) -> None:
    with psycopg.connect(database_url) as connection:
        update_document_status(connection, document_id, status)


def save_parsed(database_url: str, document_id, outputs: dict) -> None:
    with psycopg.connect(database_url) as connection:
        mark_document_parsed(connection, document_id, outputs)


def save_failed(database_url: str, document_id, error: Exception) -> None:
    with psycopg.connect(database_url) as connection:
        mark_document_failed(connection, document_id, error)


def main() -> None:
    args = parse_arguments()
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.environ.get("DATABASE_URL")
    firecrawl_api_key = os.environ.get("FIRECRAWL_API_KEY")

    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from .env")
    if not firecrawl_api_key and not args.dry_run:
        raise RuntimeError("FIRECRAWL_API_KEY is missing from .env")

    with psycopg.connect(database_url) as connection:
        documents = fetch_documents_for_parsing(connection, source_type="html")

    if not documents:
        print("No discovered, parsing, or failed HTML sources need parsing.")
        return

    print(f"HTML sources to parse: {len(documents)}")
    for document in documents:
        existing = find_existing_html_outputs(
            document["file_name"],
            document["domain"],
        )
        action = "reuse existing local outputs" if existing else "start Firecrawl crawl"
        print(f"  {document['file_name']} -> {action}")

    if args.dry_run:
        print("\nDry run complete. No crawls were started and Neon was not updated.")
        return

    failures = []
    for document in documents:
        name = document["file_name"]
        domain = document["domain"]
        config = (document.get("metadata") or {}).get("firecrawl_config") or {}
        print(f"\nHTML source: {name}")

        try:
            existing = find_existing_html_outputs(name, domain)
            if existing:
                outputs = existing
                print(f"[{name}] Reusing existing local outputs")
            else:
                save_status(database_url, document["document_id"], "parsing")
                outputs = parse_html_crawl(
                    name=name,
                    domain=domain,
                    config=config,
                    api_key=firecrawl_api_key,
                )

            save_parsed(database_url, document["document_id"], outputs)
            print(
                f"Parsed: {name} | pages={outputs['page_count']} | "
                f"sections={outputs['section_count']}"
            )
        except Exception as exc:
            failures.append((name, str(exc)))
            try:
                save_failed(database_url, document["document_id"], exc)
            except Exception as database_exc:
                print(f"Could not save failure status: {database_exc}")
            print(f"Failed: {name} | {exc}")

    print(f"\nCompleted: {len(documents) - len(failures)}")
    print(f"Failed:    {len(failures)}")
    if failures:
        raise RuntimeError(
            "Some HTML sources failed: "
            + ", ".join(name for name, _ in failures)
        )


if __name__ == "__main__":
    main()