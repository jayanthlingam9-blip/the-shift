import json
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.hierarchical_chunker import create_chunks
from src.config import settings
from src.database.repository import (
    clear_document_chunks,
    insert_child_chunks,
    insert_parent_sections,
    update_document_status,
)
from src.models import NormalizedElement

NORMALIZED_ROOT = PROJECT_ROOT / "data" / "normalized"
IMAGES_ROOT = PROJECT_ROOT / "data" / "images"
CHUNKS_ROOT = PROJECT_ROOT / "data" / "chunks"


def _fetch_chunkable_documents(connection: psycopg.Connection) -> list[dict]:
    with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(
            """
            SELECT document_id, domain, file_name, source_type
            FROM documents
            WHERE source_type IN ('pdf', 'html')
            ORDER BY domain, file_name;
            """
        )
        return cursor.fetchall()


def _load_elements(domain: str, source_type: str, slug: str) -> list[NormalizedElement]:
    path = NORMALIZED_ROOT / domain / source_type / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"Normalized file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [NormalizedElement(**item) for item in raw]


def _resolve_image_paths(chunk: dict, domain: str, slug: str) -> None:
    """Fill image_paths from the chunker's image_refs (element ids)."""
    refs = chunk["metadata"].get("image_refs")
    if not refs:
        return
    image_dir = IMAGES_ROOT / domain / slug
    paths: list[str] = []
    for ref in refs:
        stem = ref.split(":")[-1]
        for match in sorted(image_dir.glob(f"{stem}.*")):
            paths.append(match.relative_to(PROJECT_ROOT).as_posix())
            break
    chunk["image_paths"] = paths


def _write_chunks_json(
    document: dict,
    slug: str,
    parent_rows: list[dict],
    child_rows: list[dict],
) -> Path:
    """Persist a document's parents + children as JSON under data/chunks/."""
    out_dir = CHUNKS_ROOT / document["domain"] / document["source_type"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.json"
    payload = {
        "document_id": str(document["document_id"]),
        "file_name": document["file_name"],
        "domain": document["domain"],
        "source_type": document["source_type"],
        "parent_section_count": len(parent_rows),
        "child_chunk_count": len(child_rows),
        "parent_sections": parent_rows,
        "child_chunks": child_rows,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from .env")

    with psycopg.connect(database_url) as connection:
        documents = _fetch_chunkable_documents(connection)

    if not documents:
        print("No PDF/HTML documents to chunk.")
        return

    for document in documents:
        slug = Path(document["file_name"]).stem
        domain = document["domain"]
        source_type = document["source_type"]
        print(f"\nChunking {source_type}: {document['file_name']} (slug={slug})")

        elements = _load_elements(domain, source_type, slug)
        parents, children = create_chunks(
            elements,
            document_id=document["document_id"],
            source_type=source_type,
            file_name=document["file_name"],
            domain=domain,
            target_tokens=settings.chunk_target_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            max_tokens=settings.chunk_max_tokens,
            parent_max_tokens=settings.chunk_parent_max_tokens,
        )

        parent_rows = [p.model_dump(mode="json") for p in parents]
        child_rows = [c.model_dump(mode="json") for c in children]
        for row in child_rows:
            _resolve_image_paths(row, domain, slug)

        with psycopg.connect(database_url) as connection:
            clear_document_chunks(connection, document["document_id"])
            insert_parent_sections(connection, parent_rows)
            insert_child_chunks(connection, child_rows)
            update_document_status(connection, document["document_id"], "chunked")
            connection.commit()

        out_path = _write_chunks_json(document, slug, parent_rows, child_rows)
        print(
            f"  parents={len(parent_rows):,} | children={len(child_rows):,} "
            f"| json={out_path.relative_to(PROJECT_ROOT).as_posix()}"
        )

    print("\nChunking complete.")


if __name__ == "__main__":
    main()
