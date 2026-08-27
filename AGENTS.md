## Project Context

This is a local multimodal hybrid RAG project over domain-specific files.

Before making architectural or pipeline changes, read `README.md` for the full architecture and data flow.

If `README.md` and this file conflict, follow `AGENTS.md`.

## Security Rules

- Only read, write, search, and process files inside this project folder.
- Do not inspect the user's home directory, Desktop, Downloads, Documents, or any folder outside this repository.
- Do not recursively scan parent directories.
- Process source documents only from `data/input/`.
- Do not read or print `.env` values.
- Never expose API keys, tokens, credentials, or private document contents.
- Use `.env.example` only to understand required environment variable names.
- Never commit secrets, raw API keys, credentials, tokens, parsed private files, or local database outputs.

## Project Goal

Build a local notebook/script-based multimodal RAG system for 12 files across domains:

- legal
- finance
- medical

Supported source types:

- PDF
- HTML
- CSV

The system should parse, chunk, embed, retrieve, rerank, generate grounded answers, cite sources, and evaluate with RAGAS.

## Core Pipeline

Local domain files
  -> file discovery
  -> parser router
  -> raw parsed outputs
  -> normalized elements
  -> hierarchical parent-child chunking
  -> dense + sparse indexing
  -> Neon Postgres + pgvector
  -> hybrid retrieval
  -> RRF fusion
  -> child-to-parent expansion
  -> Voyage reranker
  -> Gemini generation
  -> grounded answer + citations

## graphify

This project has a fresh AST/code knowledge graph at graphify-out/ with 141 nodes, 173 edges, 34 communities, god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost). If the graph was deleted or intentionally reset, run `graphify update . --force` from the repository root to recreate graphify-out/.
