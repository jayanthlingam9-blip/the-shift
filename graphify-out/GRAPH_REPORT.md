# Graph Report - The Shift  (2026-08-27)

## Corpus Check
- 73 files · ~522,986 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 453 nodes · 769 edges · 44 communities (34 shown, 10 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 21 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `16cca5c5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 159|Community 159]]
- [[_COMMUNITY_Community 165|Community 165]]
- [[_COMMUNITY_Community 168|Community 168]]
- [[_COMMUNITY_Community 169|Community 169]]

## God Nodes (most connected - your core abstractions)
1. `NormalizedElement` - 18 edges
2. `create_chunks()` - 14 edges
3. `Connection` - 14 edges
4. `run_ragas_evaluation()` - 13 edges
5. `parse_hcahps_csv()` - 13 edges
6. `parse_pdf()` - 13 edges
7. `hybrid_search()` - 13 edges
8. `embed_texts()` - 12 edges
9. `answer_query()` - 12 edges
10. `filter_images()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Connection` --uses--> `NormalizedElement`  [INFERRED]
  scripts/create_chunks.py → src/models.py
- `NormalizedElement` --uses--> `NormalizedElement`  [INFERRED]
  scripts/create_chunks.py → src/models.py
- `Path` --uses--> `NormalizedElement`  [INFERRED]
  scripts/create_chunks.py → src/models.py
- `_caption_with_progress()` --calls--> `caption_images()`  [EXTRACTED]
  run_captioning.py → src/normalization/image_captioning.py
- `main()` --calls--> `refresh_centroids()`  [EXTRACTED]
  scripts/compute_domain_centroids.py → src/retrieval/router.py

## Import Cycles
- None detected.

## Communities (44 total, 10 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.12
Nodes (29): save_llamaparse_job(), build_parse_options(), _download_images(), _download_structured_json(), find_existing_pdf_outputs(), parse_pdf(), Return a manifest when all required raw outputs already exist., Return the shared LlamaParse v2 configuration for project PDFs. (+21 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (24): _build_prompt(), _call_gemini(), caption_image(), caption_images(), _clean_text(), _context_lines(), fallback_caption(), _file_sha256() (+16 more)

### Community 2 - "Community 2"
Cohesion: 0.19
Nodes (19): _area(), _assign_phash_groups(), filter_images(), FilterResult, _hamming(), ImageInstance, _iou(), _is_photographic() (+11 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (37): by_domain, finance, healthcare, legal, out_of_scope, answer_relevancy, context_precision, context_recall (+29 more)

### Community 4 - "Community 4"
Cohesion: 0.21
Nodes (14): clear_document_chunks(), insert_child_chunks(), insert_parent_sections(), Remove derived parent/child rows before a clean, repeatable re-chunk., _fetch_chunkable_documents(), _load_elements(), main(), Connection (+6 more)

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (13): scored, total, scored, total, scored, total, coverage, answer_relevancy (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.12
Nodes (15): 1. Environment, 2. Database, 3. Secrets, Conventions, Data model, Knowledge graph, Notebooks, Repository layout (+7 more)

### Community 7 - "Community 7"
Cohesion: 0.09
Nodes (46): clear_csv_data(), fetch_documents_for_parsing(), get_parser_version(), insert_hcahps_records(), mark_document_failed(), mark_document_parsed(), Remove derived CSV rows before a clean, repeatable re-import., update_document_status() (+38 more)

### Community 8 - "Community 8"
Cohesion: 0.36
Nodes (11): clean_text(), element_from_tag(), extract_structured_page(), parse_html_crawl(), save_combined_markdown(), save_json(), save_jsonl(), stable_id() (+3 more)

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (5): Core Pipeline, graphify, Project Context, Project Goal, Security Rules

### Community 10 - "Community 10"
Cohesion: 0.17
Nodes (26): BaseModel, _clean(), _content_items(), _discover_documents(), _image_context(), _item_text(), _item_y(), normalize_all() (+18 more)

### Community 19 - "Community 19"
Cohesion: 0.06
Nodes (32): BaseSettings, embed_sparse(), _load_model(), Sparse embeddings via SPLADE (naver/splade-cocondenser-ensembledistil).  Each te, dense_search(), Dense (semantic) retrieval over the halfvec(1024) columns.  Queries each embedde, Cosine-nearest retrieval units to the query's dense vector.      Returns up to `, expand_to_parents() (+24 more)

### Community 165 - "Community 165"
Cohesion: 0.16
Nodes (27): Chunk, _build_children(), _clean_title(), _count_tokens(), create_chunks(), _det_uuid(), _emit(), _expand_oversized_elements() (+19 more)

### Community 168 - "Community 168"
Cohesion: 0.11
Nodes (20): Citation, answer_query(), _filter_by_score(), End-to-end RAG answer: hybrid retrieval -> grounded generation + citations.  Tie, Keep hits at/above the rerank-score floor, but never drop the top hit., Answer `query` over the corpus. Returns answer, citations, contexts, hits., build_citations(), build_context() (+12 more)

### Community 169 - "Community 169"
Cohesion: 0.07
Nodes (38): embed_texts(), _get_client(), Dense embeddings via Voyage (text-only: voyage-3-large, 1024 dims).  Images were, Embed a batch of texts. Caller is responsible for sane batch sizes.      Retries, _build_judge(), _ensure_ragas_importable(), _judge_keys(), _load_dataset() (+30 more)

## Knowledge Gaps
- **71 isolated node(s):** `PreToolUse`, `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall` (+66 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `update_document_status()` connect `Community 7` to `Community 0`, `Community 4`?**
  _High betweenness centrality (0.161) - this node is a cross-community bridge._
- **Why does `embed_texts()` connect `Community 169` to `Community 19`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `answer_query()` connect `Community 168` to `Community 169`, `Community 19`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `NormalizedElement` (e.g. with `Chunk` and `ParentSection`) actually correct?**
  _`NormalizedElement` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `PreToolUse`, `faithfulness`, `answer_relevancy` to the rest of the system?**
  _144 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.12310606060606061 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.11954022988505747 - nodes in this community are weakly interconnected._