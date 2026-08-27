import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.repository import (
    fetch_documents_for_parsing,
    mark_document_failed,
    mark_document_parsed,
    save_llamaparse_job,
    update_document_status,
)
from src.parsing.pdf_parser import find_existing_pdf_outputs, parse_pdf


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse discovered PDFs using the configured LlamaCloud keys."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show PDF/key assignments without creating jobs or updating Neon.",
    )
    return parser.parse_args()


def load_api_keys() -> dict[str, str]:
    keys = {}

    for number in range(1, 4):
        alias = f"key_{number}"
        value = os.environ.get(f"LLAMA_CLOUD_API_KEY_{number}")
        if value:
            keys[alias] = value

    if not keys:
        raise RuntimeError(
            "Add LLAMA_CLOUD_API_KEY_1, LLAMA_CLOUD_API_KEY_2, and/or "
            "LLAMA_CLOUD_API_KEY_3 to .env"
        )

    return keys


def existing_job_details(document: dict) -> tuple[str | None, str | None]:
    llamaparse = (document.get("metadata") or {}).get("llamaparse") or {}
    return llamaparse.get("job_id"), llamaparse.get("api_key_alias")


def update_status(
    database_url: str,
    document_id,
    status: str,
) -> None:
    with psycopg.connect(database_url) as connection:
        update_document_status(connection, document_id, status)


def save_job(
    database_url: str,
    document_id,
    job_id: str,
    api_key_alias: str,
) -> None:
    with psycopg.connect(database_url) as connection:
        save_llamaparse_job(connection, document_id, job_id, api_key_alias)


def save_parsed_outputs(
    database_url: str,
    document_id,
    outputs: dict,
) -> None:
    with psycopg.connect(database_url) as connection:
        mark_document_parsed(connection, document_id, outputs)


def save_failure(
    database_url: str,
    document_id,
    error: Exception | str,
) -> None:
    with psycopg.connect(database_url) as connection:
        mark_document_failed(connection, document_id, error)


def parse_one_document(
    document: dict,
    api_key_alias: str,
    api_key: str,
    database_url: str,
) -> dict:
    document_id = document["document_id"]
    existing_job_id, _ = existing_job_details(document)

    existing_outputs = find_existing_pdf_outputs(
        document["file_path"],
        document["domain"],
    )
    if existing_outputs:
        if existing_job_id:
            existing_outputs["job_id"] = existing_job_id

        save_parsed_outputs(database_url, document_id, existing_outputs)
        return {
            "file_name": document["file_name"],
            "status": "reused",
            "job_id": existing_job_id or "existing-output",
            "image_count": existing_outputs["downloaded_image_count"],
        }

    update_status(database_url, document_id, "parsing")

    def save_new_job(job_id: str) -> None:
        save_job(database_url, document_id, job_id, api_key_alias)

    try:
        outputs = parse_pdf(
            file_path=document["file_path"],
            domain=document["domain"],
            api_key=api_key,
            existing_job_id=existing_job_id,
            on_job_created=save_new_job,
        )
        save_parsed_outputs(database_url, document_id, outputs)

        return {
            "file_name": document["file_name"],
            "status": "parsed",
            "job_id": outputs["job_id"],
            "image_count": outputs["downloaded_image_count"],
        }
    except Exception as exc:
        try:
            save_failure(database_url, document_id, exc)
        except Exception as database_exc:
            print(
                f"Could not save failure status for {document['file_name']}: "
                f"{database_exc}"
            )
        raise


def assign_documents(
    documents: list[dict],
    api_keys: dict[str, str],
) -> list[tuple[dict, str, str]]:
    aliases = list(api_keys)
    assignments = []

    for index, document in enumerate(documents):
        _, existing_alias = existing_job_details(document)
        alias = existing_alias or aliases[index % len(aliases)]

        if alias not in api_keys:
            raise RuntimeError(
                f"{document['file_name']} uses stored API key alias "
                f"{alias!r}, but that key is not available in .env"
            )

        assignments.append((document, alias, api_keys[alias]))

    return assignments


def parse_key_queue(
    assignments: list[tuple[dict, str, str]],
    database_url: str,
) -> list[dict]:
    """Parse one key's assigned PDFs sequentially."""
    results = []

    for document, alias, api_key in assignments:
        try:
            result = parse_one_document(
                document,
                alias,
                api_key,
                database_url,
            )
            results.append(result)
        except Exception as exc:
            results.append(
                {
                    "file_name": document["file_name"],
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return results


def main() -> None:
    args = parse_arguments()
    load_dotenv(PROJECT_ROOT / ".env")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from .env")

    api_keys = load_api_keys()

    with psycopg.connect(database_url) as connection:
        documents = fetch_documents_for_parsing(connection, source_type="pdf")

    if not documents:
        print("No discovered or failed PDFs need parsing.")
        return

    assignments = assign_documents(documents, api_keys)

    print(f"PDFs to parse: {len(assignments)}")
    for document, alias, _ in assignments:
        job_id, _ = existing_job_details(document)
        existing_outputs = find_existing_pdf_outputs(
            document["file_path"],
            document["domain"],
        )
        if existing_outputs:
            action = "reuse existing local outputs"
        elif job_id:
            action = f"resume {job_id}"
        else:
            action = "create new job"
        print(f"  {document['file_name']} -> {alias} ({action})")

    if args.dry_run:
        print("\nDry run complete. No jobs were created and Neon was not updated.")
        return

    queues = {
        alias: [assignment for assignment in assignments if assignment[1] == alias]
        for alias in api_keys
    }
    queues = {alias: queue for alias, queue in queues.items() if queue}

    results = []
    with ThreadPoolExecutor(max_workers=len(api_keys)) as executor:
        futures = {
            executor.submit(parse_key_queue, queue, database_url): alias
            for alias, queue in queues.items()
        }

        for future in as_completed(futures):
            alias = futures[future]
            key_results = future.result()
            results.extend(key_results)

            for result in key_results:
                file_name = result["file_name"]
                if result["status"] != "failed":
                    print(
                        f"Parsed: {file_name} | key={alias} | "
                        f"job={result['job_id']} | "
                        f"images={result['image_count']}"
                    )
                else:
                    print(f"Failed: {file_name} | key={alias} | {result['error']}")

    failures = [result for result in results if result["status"] == "failed"]

    print(f"\nCompleted: {len(assignments) - len(failures)}")
    print(f"Failed:    {len(failures)}")

    if failures:
        failed_names = ", ".join(result["file_name"] for result in failures)
        raise RuntimeError(
            f"Some PDFs failed: {failed_names}. Re-run this script to resume "
            "them from their saved LlamaParse job IDs."
        )


if __name__ == "__main__":
    main()