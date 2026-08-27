"""Child-to-parent expansion for context recall.

First-stage retrieval scores small `child_chunks` (token-bounded windows) for
precision. Before reranking we swap each child for its `parent_sections` text so
the reranker and the generator see the full heading-delimited section -- this
is the "small-to-big" pattern. Multiple children of the same parent collapse
into one parent (the best-ranked occurrence wins), which also shrinks the
candidate set. Non-child units (the hospital_* rollups) and any child without a
parent pass through unchanged.
"""

import psycopg


def expand_to_parents(
    connection: psycopg.Connection,
    hits: list[dict],
) -> list[dict]:
    """Replace child-chunk hits with their parent sections, deduped, in order.

    Each surviving hit keeps its first-stage score fields (e.g. `rrf_score`);
    expanded hits gain `parent_id`, get their `text` swapped to the parent's
    `parent_text`, and carry the parent `section_title`/`section_path`.
    """
    child_ids = [
        hit["chunk_id"] for hit in hits if hit.get("source_table") == "child_chunks"
    ]

    parents: dict[str, dict] = {}
    if child_ids:
        with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
            cursor.execute(
                """
                SELECT
                    c.chunk_id::text   AS chunk_id,
                    p.parent_id::text  AS parent_id,
                    p.parent_text      AS parent_text,
                    p.section_title    AS section_title,
                    p.section_path     AS section_path
                FROM child_chunks c
                JOIN parent_sections p ON c.parent_id = p.parent_id
                WHERE c.chunk_id = ANY(%s);
                """,
                (child_ids,),
            )
            parents = {row["chunk_id"]: row for row in cursor.fetchall()}

    expanded: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        parent = parents.get(hit["chunk_id"]) if hit.get("source_table") == "child_chunks" else None

        if parent:
            key = ("parent", parent["parent_id"])
            if key in seen:
                continue
            seen.add(key)
            new_hit = dict(hit)
            new_hit["parent_id"] = parent["parent_id"]
            new_hit["text"] = parent["parent_text"]
            new_hit["section_title"] = parent["section_title"]
            new_hit["section_path"] = parent["section_path"]
            expanded.append(new_hit)
        else:
            key = (hit.get("source_table", ""), hit["chunk_id"])
            if key in seen:
                continue
            seen.add(key)
            expanded.append(hit)

    return expanded
