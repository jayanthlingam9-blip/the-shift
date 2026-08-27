# The Shift

A local, multimodal, hybrid **RAG** pipeline over domain-specific legal, finance, and medical
documents — built end to end from raw PDF/HTML/CSV to a grounded, cited answer, and scored with
RAGAS.

The system parses 12 real-world documents with layout-aware parsers, captions the figures that
matter, chunks hierarchically (small children for precision, big parents for recall), indexes both
dense and sparse vectors in Postgres/pgvector, routes each query to its domain by centroid, fuses
two retrievers with RRF, reranks with a cross-encoder, and generates an answer that either cites
its sources or refuses.

```
data/input/ ──► discovery ──► parser router ──► normalize (+ image filter & captioning)
                                                        │
                                          hierarchical parent/child chunking
                                                        │
                             Voyage dense  ┐            ▼
                             SPLADE sparse ┴──► Postgres 17 + pgvector (halfvec / sparsevec)
                                                        │
   query ──► centroid domain routing ──► dense ∥ sparse search ──► RRF fusion
                                                        │
                            child→parent expansion ──► Voyage rerank ──► score floor
                                                        │
                                       Gemini generation ──► answer + numbered citations
                                                        │
                                                RAGAS evaluation
```

---

## What makes it more than a demo

| Concern | How it is handled |
| --- | --- |
| **Layout-heavy PDFs** | LlamaParse *agentic* tier with aggressive table extraction, specialized chart parsing, OCR, and printed-page-number extraction. Embedded **and** layout images are saved. |
| **Image noise** | [`image_filter.py`](src/normalization/image_filter.py) triages images *before* they cost an API call — SHA-256 exact dupes, perceptual-hash near-dupes, repeated logos/headers/watermarks, and tiny decorative strips are dropped with an auditable reason. |
| **Multimodal content** | Surviving figures are captioned by Gemini into text ([`image_captioning.py`](src/normalization/image_captioning.py)), cached by content hash, and indexed alongside the prose — so charts become retrievable. |
| **Precision vs. recall** | Small token-bounded child chunks are what gets embedded and searched; a hit is swapped for its heading-delimited **parent section** before reranking (small-to-big). Parents are capped at 2000 tokens so a heading-less section cannot become one giant blob. |
| **Lexical vs. semantic** | Voyage dense embeddings *and* SPLADE learned-sparse vectors, fused by reciprocal rank fusion — exact identifiers (measure IDs, article numbers) survive that dense-only retrieval loses. |
| **Cross-domain bleed** | One mean dense **centroid per domain**; the query is routed to its nearest domains. Soft routing (top-2) by default, because finance and legal centroids overlap and hard routing can discard the correct domain on boundary queries. |
| **Storage** | `halfvec` (fp16) dense columns halve vector + HNSW index size at negligible recall cost. The 100 MB HCAHPS fact table is deliberately left **unembedded** — it exists for structured filtering, not vector search. |
| **Hallucination** | A rerank-score floor drops weak distractor contexts before they reach the prompt, and the generator is instructed to refuse when the context does not support an answer. Out-of-scope questions in the eval set verify the refusal path. |
| **Cost & rate limits** | Multiple Gemini keys rotate with a per-key RPM budget and bounded worker pool; indexing is resumable (only `NULL`-embedding rows are re-embedded); captions are cached on disk. |

---

## Results

Latest RAGAS run — 28 questions (24 in-scope + 4 out-of-scope), judged by Gemini with Voyage
embeddings, i.e. the same providers the pipeline itself uses.

| Slice | Faithfulness | Answer relevancy | Context precision | Context recall | n |
| --- | --- | --- | --- | --- | --- |
| **Overall** | **0.984** | 0.469 | 0.688 | **0.905** | 28 |
| In-scope only | 0.981 | 0.548 | 0.688 | 0.889 | 24 |
| Finance | 0.978 | 0.627 | 0.667 | 0.907 | 9 |
| Legal | **1.000** | 0.564 | **0.870** | **0.963** | 9 |
| Healthcare | 0.958 | 0.403 | 0.400 | 0.750 | 6 |
| Out-of-scope | 1.000 | 0.000 | — | 1.000 | 4 |

Reading the numbers: **faithfulness ~0.98 and context recall ~0.90** say the pipeline retrieves the
right evidence and stays grounded in it. The low **overall answer relevancy is expected and partly
correct** — the 4 out-of-scope questions score 0.0 *because the system correctly refuses to answer
them*, which drags the mean down; that is why the in-scope slice is reported separately. Healthcare
is the weakest domain: the HCAHPS CSV rollups compete with each other in retrieval, so context
precision suffers.

Raw output: `data/eval/ragas_results.json`, with per-question scores in `data/eval/ragas_per_row.csv`.

---

## The corpus

Twelve documents across three domains. They are **not committed** — they are third-party
publications, and the HCAHPS CSV alone is 100 MB. Drop your own copies into the matching folder:

| Domain | File | Type | Source |
| --- | --- | --- | --- |
| legal | `gdpr.pdf` | PDF | EU General Data Protection Regulation (Reg. 2016/679) |
| legal | `RBI.pdf` | PDF | Reserve Bank of India master direction |
| legal | `sebi_circular_2026.pdf` | PDF | SEBI circular |
| finance | `jpm_proxy_2026.pdf` | PDF | JPMorgan Chase proxy statement |
| finance | `ril_annual_report_2025.pdf` | PDF | Reliance Industries annual report FY 2024-25 |
| finance | `msft_2025_10k.json` | HTML manifest | Microsoft FY2025 10-K, crawled from SEC EDGAR |
| medical | `united_healthcare_policy.pdf` | PDF | UnitedHealthcare coverage policy |
| medical | `dailymed_ozempic_prescribing_label.json` | HTML manifest | DailyMed Ozempic prescribing label |
| medical | `hcahps_hospital.csv` | CSV | CMS HCAHPS hospital survey (~100 MB) |

HTML sources are **not** stored as `.html`. They are declared as small JSON *manifests* naming a URL
plus a Firecrawl crawl config, and fetched at parse time:

```json
[{ "name": "msft_2025_10k", "domain": "finance", "source_type": "html",
   "parser_used": "firecrawl", "firecrawl_mode": "crawl",
   "firecrawl_config": { "url": "https://...", "limit": 10,
                         "scrapeOptions": { "formats": ["markdown", "html"] } } }]
```

Document IDs are deterministic `uuid5(domain:source_type:path)` values, so re-running discovery over
the same file is idempotent.

---

## Setup

### 1. Environment

```powershell
python -m venv rag_env
.\rag_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **LlamaCloud SDK conflict.** [`pdf_parser.py`](src/parsing/pdf_parser.py) targets the
> Stainless-generated **`llama-cloud==2.8.0`** SDK, which `requirements-core.txt` pins exactly for
> this reason. Do **not** add `llama-parse` or `llama-cloud-services` — they hard-pin
> `llama-cloud==0.1.46` (the older Fern SDK, incompatible API) under the same `llama_cloud` import
> name, so installing either overwrites the pin and breaks parsing with
> `ImportError: cannot import name 'FilesResource'`. Recovery: uninstall all three, delete the
> leftover `site-packages/llama_cloud/` folder and stray `*.dist-info`, then
> `pip install --no-cache-dir "llama-cloud==2.8.0"`.

Dependency groups: `requirements-core.txt` (parsing, DB, image triage, API clients) ·
`requirements-ml.txt` (torch/transformers for SPLADE) · `requirements-eval.txt` (RAGAS, pytest) ·
`requirements-notebook.txt` (JupyterLab). `requirements.txt` pulls in all four.

### 2. Database

Local Postgres 17 with pgvector, via Docker:

```powershell
docker compose up -d                   # data persists in the pgdata volume
python .\scripts\create_database.py    # applies src/database/schema.sql
```

The host port is **15432**, not 5432 — Windows reserves 5413-5812 via Hyper-V/winnat. Change
`POSTGRES_PASSWORD` in `docker-compose.yml` and keep it in sync with `DATABASE_URL`.

`docker compose down` stops and keeps the data; `docker compose down -v` deletes the volume.

### 3. Secrets

Copy `.env.example` to `.env` and fill in:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Postgres connection string (aliased to `NEON_DATABASE_URL` in config) |
| `LLAMA_CLOUD_API_KEY_1` | LlamaParse PDF parsing. **Numbered, not bare** — `parse_documents.py` reads `_1`/`_2`/`_3` and rotates across whichever are set; `_1` is required |
| `FIRECRAWL_API_KEY` | HTML crawling |
| `VOYAGE_API_KEY` | Dense embeddings and reranking |
| `GEMINI_API_KEY` | Generation, image captioning, RAGAS judge. `GEMINI_API_KEY_1` is equivalent; `_2`/`_3` are optional and rotate to raise the per-minute quota |
| `VOYAGE_EMBED_MODEL` | default `voyage-3-large` (1024-dim) |
| `VOYAGE_RERANK_MODEL` | default `rerank-2.5` |
| `GEMINI_MODEL` | default `gemini-2.5-flash` |
| `SPLADE_MODEL` | default `naver/splade-cocondenser-ensembledistil` |

The RAGAS judge is not separately configurable — it reuses `GEMINI_MODEL`.

`.env` is gitignored. Never commit it.

> If you change `VOYAGE_EMBED_MODEL`, keep `voyage_embed_dim` in [`src/config.py`](src/config.py)
> and the `halfvec(1024)` columns in [`schema.sql`](src/database/schema.sql) in sync.

---

## Running the pipeline

From the project root, in order:

```powershell
python .\scripts\create_database.py           # extension + tables + HNSW indexes
python .\scripts\discover_files.py            # walk data/input/, register documents
python .\scripts\parse_documents.py           # PDFs via LlamaParse, HTML via Firecrawl
python .\scripts\parse_csv.py                 # HCAHPS -> facts + profile/category rollups
python .\run_captioning.py                    # image filter -> Gemini captions -> normalize
python .\scripts\create_chunks.py             # hierarchical parent/child chunks
python .\scripts\index_chunks.py              # Voyage dense + SPLADE sparse (resumable)
python .\scripts\compute_domain_centroids.py  # one centroid per domain, for routing
```

Then query:

```powershell
python .\scripts\query.py "What is JPMorgan's ROTCE?"
python .\scripts\query.py "..." --rerank-top-k 12 --route-top-k 0   # 0 = search whole corpus
python .\scripts\query.py "..." --no-parents                        # skip parent expansion
```

And evaluate:

```powershell
python .\scripts\evaluate.py                                # full 28-question suite
python .\scripts\evaluate.py --limit 3 --raise-exceptions   # surface a judge error
python .\scripts\evaluate.py --context-min-score 0.2        # tighten the score floor
```

Helper scripts: `parse_html.py` (HTML only), `recover_parsed_outputs.py` (rebuild `data/parsed/`
from the database), `backup_db.ps1` / `restore_db.ps1` (pg_dump round-trip).

---

## Repository layout

```
src/
├── config.py                         Pydantic settings; every tunable lives here
├── models.py                         SourceDocument, NormalizedElement, ParentSection, Chunk, Citation
├── discovery/file_discovery.py       Walks data/input/, uuid5 document IDs, Firecrawl manifests
├── parsing/
│   ├── pdf_parser.py                 LlamaParse agentic tier + image extraction
│   ├── html_parser.py                Firecrawl crawl -> markdown/html
│   └── csv_parser.py                 HCAHPS -> fact rows + hospital/category rollup documents
├── normalization/
│   ├── image_filter.py               Dedupe / boilerplate / decorative triage before captioning
│   ├── image_captioning.py           Gemini captions, content-hash cached, key rotation
│   └── normalize.py                  Parser outputs -> NormalizedElement stream
├── chunking/hierarchical_chunker.py  Token-bounded children under heading-delimited parents
├── embeddings/
│   ├── voyage_embeddings.py          Dense, with input_type=query|document
│   └── splade_embeddings.py          Learned-sparse, capped at 512 terms for HNSW
├── retrieval/
│   ├── router.py                     Centroid computation + query-time domain routing
│   ├── dense.py / sparse.py          Per-modality first-stage search, domain-scoped
│   ├── rrf.py                        Reciprocal rank fusion (k = 60)
│   ├── parent_expansion.py           Child hit -> parent section (small-to-big)
│   ├── reranker.py                   Voyage cross-encoder rerank
│   └── search.py                     Orchestrates the whole retrieval chain
├── generation/
│   ├── context_builder.py            Assembles reranked hits into the prompt
│   ├── gemini_generator.py           Grounded generation; refuses when unsupported
│   ├── citations.py                  Numbered citations bound to chunk IDs
│   └── answer.py                     answer_query(): retrieval + generation + score floor
├── evaluation/
│   ├── eval_dataset.py               28 hand-authored question / ground_truth / domain items
│   └── ragas_evaluator.py            RAGAS with Gemini judge + Voyage embeddings, sliced by domain
└── database/
    ├── schema.sql                    Tables, halfvec/sparsevec columns, HNSW indexes, centroids
    └── repository.py                 All SQL: upserts, batch inserts, embedding writes
```

### Data model

| Table | Role |
| --- | --- |
| `documents` | One row per source file or URL; parser used, status, metadata |
| `parent_sections` | Heading-delimited sections; the *expansion* target |
| `child_chunks` | Token-bounded retrieval units — **embedded** (dense + sparse) |
| `hcahps_records` | Raw CSV facts; **not embedded**, exists for structured filtering |
| `hospital_profiles` | Per-facility rollup — **embedded** |
| `hospital_category_docs` | Per-facility-per-category rollup — **embedded** |
| `domain_centroids` | One mean dense vector per domain, for query routing |

---

## Testing

```powershell
pytest
```

Covers chunking, citations, database, discovery, embeddings, normalization, parsing, retrieval, and
RRF. Some tests are placeholders — a green run does not by itself prove the full pipeline is live.

---

## Notebooks

`notebooks/01_pipeline_exploration.ipynb` · `02_retrieval_experiments.ipynb` ·
`03_ragas_evaluation.ipynb` — exploratory companions to the scripts, not the source of truth.

---

## Knowledge graph

`graphify-out/` holds an AST-derived knowledge graph of this repo (141 nodes, 173 edges, 34
communities). Query it instead of grepping:

```
graphify query "how does parent expansion work"
graphify path "hybrid_search" "generate_answer"
graphify explain "domain routing"
graphify update .            # refresh after code changes (AST-only, no API cost)
```

`graphify-out/wiki/index.md` is the browsable entry point; `GRAPH_REPORT.md` is the full
architecture dump.

---

## Conventions

- Source documents live **only** under `data/input/`; everything else in `data/` is generated.
- Reusable code lives in `src/` and is imported as `src.*`; `scripts/` holds runnable entry points
  and inserts the project root onto `sys.path`.
- If your editor flags unresolved imports, point it at the project venv (or set
  `$env:PYTHONPATH = ".\src"` for a quick PowerShell session).
- Never commit `.env`, API keys, credentials, database dumps, or parsed private documents.
