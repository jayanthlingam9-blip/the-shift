#!/usr/bin/env python3
from pathlib import Path
import os, json, sys
from dotenv import load_dotenv
import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("DATABASE_URL missing")
    sys.exit(1)

# import project helpers
from src.parsing.pdf_parser import parse_pdf
from src.database.repository import (
    mark_document_parsed,
    save_llamaparse_job,
    mark_document_failed,
)


def load_api_keys() -> dict:
    keys = {}
    for number in range(1, 4):
        alias = f"key_{number}"
        value = os.environ.get(f"LLAMA_CLOUD_API_KEY_{number}")
        if value:
            keys[alias] = value
    return keys

api_keys = load_api_keys()
if not api_keys:
    print("Warning: no LLAMA_CLOUD_API_KEY_* found in .env; recovery may fail if job requires the original key.")

query = """
SELECT document_id, file_name, domain, file_path, metadata
FROM documents
WHERE status = 'parsed' AND metadata->'parse_outputs' IS NOT NULL
ORDER BY updated_at DESC
"""

with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(query)
        rows = cur.fetchall()
        if not rows:
            print("No parsed documents with parse_outputs found.")
            sys.exit(0)
        for row in rows:
            document_id = row["document_id"]
            file_name = row["file_name"]
            domain = row["domain"]
            file_path = row["file_path"]
            metadata = row.get("metadata") or {}
            parse_outputs = metadata.get("parse_outputs") or {}
            llamaparse = metadata.get("llamaparse") or {}

            expected = []
            for key in (
                "markdown_path",
                "structured_json_path",
                "metadata_json_path",
                "images_json_path",
                "images_directory",
            ):
                val = parse_outputs.get(key)
                expected.append((key, val))

            missing = [k for k, p in expected if not p or not (PROJECT_ROOT / p).exists()]
            if not missing:
                print(f"{file_name}: all outputs present")
                continue

            print(f"{file_name}: missing outputs {missing}")

            # Decide job id and API key
            job_id = llamaparse.get("job_id") or parse_outputs.get("job_id")
            api_key_alias = llamaparse.get("api_key_alias")
            api_key = api_keys.get(api_key_alias) if api_key_alias else None
            if not api_key:
                # fallback to any available key
                if api_keys:
                    api_key_alias, api_key = next(iter(api_keys.items()))
                    print(f"Using fallback API key alias {api_key_alias}")
                else:
                    print("No API keys available in .env; cannot recover this document.")
                    continue

            def on_job_created(new_job_id: str) -> None:
                with psycopg.connect(DATABASE_URL) as conn2:
                    save_llamaparse_job(conn2, document_id, new_job_id, api_key_alias or "key_1")

            try:
                outputs = parse_pdf(
                    file_path=file_path,
                    domain=domain,
                    api_key=api_key,
                    existing_job_id=job_id,
                    on_job_created=on_job_created,
                )
                with psycopg.connect(DATABASE_URL) as conn2:
                    mark_document_parsed(conn2, document_id, outputs)
                print(f"{file_name}: recovered outputs saved")
            except Exception as exc:
                print(f"{file_name}: failed to recover outputs: {exc}")
                try:
                    with psycopg.connect(DATABASE_URL) as conn2:
                        mark_document_failed(conn2, document_id, str(exc)[:2000])
                except Exception as db_exc:
                    print(f"Could not mark document failed in DB: {db_exc}")
