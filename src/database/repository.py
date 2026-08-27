from importlib.metadata import PackageNotFoundError, version

import psycopg
from psycopg.types.json import Jsonb

from src.discovery.file_discovery import DiscoveredDocument


PARSER_PACKAGES = {
    "llamaparse": "llama-cloud",
    "firecrawl": "firecrawl-py",
    "pandas": "pandas",
}


def get_parser_version(parser_used: str) -> str | None:
    package = PARSER_PACKAGES.get(parser_used)

    if not package:
        return None

    try:
        return version(package)
    except PackageNotFoundError:
        return None


def upsert_document(
    connection: psycopg.Connection,
    document: DiscoveredDocument,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (
                document_id,
                domain,
                file_name,
                source_type,
                file_path,
                parser_used,
                parser_version,
                status,
                metadata
            )
            VALUES (
                %(document_id)s,
                %(domain)s,
                %(file_name)s,
                %(source_type)s,
                %(file_path)s,
                %(parser_used)s,
                %(parser_version)s,
                'discovered',
                %(metadata)s
            )
            ON CONFLICT (document_id) DO UPDATE SET
                domain = EXCLUDED.domain,
                file_name = EXCLUDED.file_name,
                source_type = EXCLUDED.source_type,
                file_path = EXCLUDED.file_path,
                parser_used = EXCLUDED.parser_used,
                parser_version = EXCLUDED.parser_version,
                metadata = EXCLUDED.metadata,
                updated_at = NOW();
            """,
            {
                "document_id": document.document_id,
                "domain": document.domain,
                "file_name": document.file_name,
                "source_type": document.source_type,
                "file_path": document.file_path,
                "parser_used": document.parser_used,
                "parser_version": get_parser_version(document.parser_used),
                "metadata": Jsonb(document.metadata),
            },
        )

def fetch_documents_for_parsing(
    connection: psycopg.Connection,
    source_type: str | None = None,
) -> list[dict]:
    query = """
        SELECT
            document_id,
            domain,
            file_name,
            source_type,
            file_path,
            parser_used,
            parser_version,
            status,
            metadata
        FROM documents
        WHERE status IN ('discovered', 'parsing', 'failed')
    """

    params = []

    if source_type:
        query += " AND source_type = %s"
        params.append(source_type)

    query += " ORDER BY domain, file_name;"

    with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        cursor.execute(query, params)
        return cursor.fetchall()


def update_document_status(
    connection: psycopg.Connection,
    document_id,
    status: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
            SET status = %s,
                updated_at = NOW()
            WHERE document_id = %s;
            """,
            (status, document_id),
        )


def save_llamaparse_job(
    connection: psycopg.Connection,
    document_id,
    job_id: str,
    api_key_alias: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
            SET metadata = jsonb_set(
                    metadata,
                    '{llamaparse}',
                    %s,
                    true
                ),
                updated_at = NOW()
            WHERE document_id = %s;
            """,
            (
                Jsonb(
                    {
                        "job_id": job_id,
                        "api_key_alias": api_key_alias,
                        "tier": "agentic",
                        "version": "latest",
                    }
                ),
                document_id,
            ),
        )


def mark_document_parsed(
    connection: psycopg.Connection,
    document_id,
    outputs: dict,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
            SET status = 'parsed',
                metadata = jsonb_set(
                    metadata,
                    '{parse_outputs}',
                    %s,
                    true
                ),
                updated_at = NOW()
            WHERE document_id = %s;
            """,
            (Jsonb(outputs), document_id),
        )


def mark_document_failed(
    connection: psycopg.Connection,
    document_id,
    error: Exception | str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
            SET status = 'failed',
                metadata = jsonb_set(
                    metadata,
                    '{parse_error}',
                    %s,
                    true
                ),
                updated_at = NOW()
            WHERE document_id = %s;
            """,
            (Jsonb(str(error)[:2000]), document_id),
        )


def clear_csv_data(
    connection: psycopg.Connection,
    document_id,
) -> None:
    """Remove derived CSV rows before a clean, repeatable re-import."""
    with connection.cursor() as cursor:
        cursor.execute(
            "DELETE FROM hcahps_records WHERE document_id = %s;",
            (document_id,),
        )
        cursor.execute(
            "DELETE FROM hospital_profiles WHERE document_id = %s;",
            (document_id,),
        )


def insert_hcahps_records(
    connection: psycopg.Connection,
    records: list[dict],
) -> None:
    if not records:
        return

    columns = [
        "document_id",
        "source_row_number",
        "facility_id",
        "facility_name",
        "address",
        "city",
        "state",
        "zip_code",
        "county",
        "telephone",
        "measure_id",
        "question",
        "answer_description",
        "star_rating",
        "star_rating_footnote",
        "answer_percent",
        "answer_percent_footnote",
        "linear_mean_value",
        "completed_surveys",
        "completed_surveys_footnote",
        "response_rate_percent",
        "response_rate_footnote",
        "survey_start_date",
        "survey_end_date",
    ]

    with connection.cursor() as cursor:
        with cursor.copy(
            """
            COPY hcahps_records (
                document_id,
                source_row_number,
                facility_id,
                facility_name,
                address,
                city,
                state,
                zip_code,
                county,
                telephone,
                measure_id,
                question,
                answer_description,
                star_rating,
                star_rating_footnote,
                answer_percent,
                answer_percent_footnote,
                linear_mean_value,
                completed_surveys,
                completed_surveys_footnote,
                response_rate_percent,
                response_rate_footnote,
                survey_start_date,
                survey_end_date
            )
            FROM STDIN
            """,
        ) as copy:
            for record in records:
                copy.write_row(tuple(record[column] for column in columns))


def upsert_hospital_profile(
    connection: psycopg.Connection,
    profile: dict,
) -> None:
    values = dict(profile)
    values["category_summaries"] = Jsonb(values["category_summaries"])
    values["metadata"] = Jsonb(values["metadata"])

    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO hospital_profiles (
                profile_id,
                document_id,
                facility_id,
                hospital_name,
                address,
                city,
                state,
                zip_code,
                county,
                telephone,
                survey_start_date,
                survey_end_date,
                completed_surveys,
                response_rate_percent,
                category_summaries,
                retrieval_text,
                metadata
            )
            VALUES (
                %(profile_id)s,
                %(document_id)s,
                %(facility_id)s,
                %(hospital_name)s,
                %(address)s,
                %(city)s,
                %(state)s,
                %(zip_code)s,
                %(county)s,
                %(telephone)s,
                %(survey_start_date)s,
                %(survey_end_date)s,
                %(completed_surveys)s,
                %(response_rate_percent)s,
                %(category_summaries)s,
                %(retrieval_text)s,
                %(metadata)s
            )
            ON CONFLICT (document_id, facility_id) DO UPDATE SET
                hospital_name = EXCLUDED.hospital_name,
                address = EXCLUDED.address,
                city = EXCLUDED.city,
                state = EXCLUDED.state,
                zip_code = EXCLUDED.zip_code,
                county = EXCLUDED.county,
                telephone = EXCLUDED.telephone,
                survey_start_date = EXCLUDED.survey_start_date,
                survey_end_date = EXCLUDED.survey_end_date,
                completed_surveys = EXCLUDED.completed_surveys,
                response_rate_percent = EXCLUDED.response_rate_percent,
                category_summaries = EXCLUDED.category_summaries,
                retrieval_text = EXCLUDED.retrieval_text,
                metadata = EXCLUDED.metadata,
                updated_at = NOW();
            """,
            values,
        )


def clear_document_chunks(
    connection: psycopg.Connection,
    document_id,
) -> None:
    """Remove derived parent/child rows before a clean, repeatable re-chunk."""
    with connection.cursor() as cursor:
        cursor.execute(
            "DELETE FROM child_chunks WHERE document_id = %s;",
            (document_id,),
        )
        cursor.execute(
            "DELETE FROM parent_sections WHERE document_id = %s;",
            (document_id,),
        )


def insert_parent_sections(
    connection: psycopg.Connection,
    sections: list[dict],
) -> None:
    if not sections:
        return

    values = []
    for section in sections:
        item = dict(section)
        item["metadata"] = Jsonb(item["metadata"])
        values.append(item)

    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO parent_sections (
                parent_id,
                document_id,
                domain,
                source_type,
                section_title,
                section_path,
                page_numbers,
                parent_text,
                metadata
            )
            VALUES (
                %(parent_id)s,
                %(document_id)s,
                %(domain)s,
                %(source_type)s,
                %(section_title)s,
                %(section_path)s,
                %(page_numbers)s,
                %(parent_text)s,
                %(metadata)s
            );
            """,
            values,
        )


def insert_child_chunks(
    connection: psycopg.Connection,
    chunks: list[dict],
) -> None:
    if not chunks:
        return

    values = []
    for chunk in chunks:
        item = dict(chunk)
        item["metadata"] = Jsonb(item["metadata"])
        values.append(item)

    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO child_chunks (
                chunk_id,
                parent_id,
                document_id,
                domain,
                source_type,
                file_name,
                page_numbers,
                section_title,
                modalities,
                text_content,
                table_markdown,
                table_html,
                image_paths,
                metadata
            )
            VALUES (
                %(chunk_id)s,
                %(parent_id)s,
                %(document_id)s,
                %(domain)s,
                %(source_type)s,
                %(file_name)s,
                %(page_numbers)s,
                %(section_title)s,
                %(modalities)s,
                %(text_content)s,
                %(table_markdown)s,
                %(table_html)s,
                %(image_paths)s,
                %(metadata)s
            );
            """,
            values,
        )


def upsert_hospital_category_docs(
    connection: psycopg.Connection,
    category_docs: list[dict],
) -> None:
    if not category_docs:
        return

    values = []
    for category_doc in category_docs:
        item = dict(category_doc)
        item["metadata"] = Jsonb(item["metadata"])
        values.append(item)

    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO hospital_category_docs (
                category_doc_id,
                profile_id,
                document_id,
                facility_id,
                category,
                retrieval_text,
                table_markdown,
                measure_ids,
                source_row_numbers,
                metadata
            )
            VALUES (
                %(category_doc_id)s,
                %(profile_id)s,
                %(document_id)s,
                %(facility_id)s,
                %(category)s,
                %(retrieval_text)s,
                %(table_markdown)s,
                %(measure_ids)s,
                %(source_row_numbers)s,
                %(metadata)s
            )
            ON CONFLICT (document_id, facility_id, category) DO UPDATE SET
                retrieval_text = EXCLUDED.retrieval_text,
                table_markdown = EXCLUDED.table_markdown,
                measure_ids = EXCLUDED.measure_ids,
                source_row_numbers = EXCLUDED.source_row_numbers,
                metadata = EXCLUDED.metadata,
                updated_at = NOW();
            """,
            values,
        )