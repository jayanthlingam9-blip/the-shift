from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    neon_database_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("neon_database_url", "database_url"),
    )
    voyage_api_key: str | None = None
    google_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "google_api_key", "gemini_api_key", "google_api_key_1", "gemini_api_key_1"
        ),
    )
    google_api_key_2: str | None = Field(
        default=None,
        validation_alias=AliasChoices("google_api_key_2", "gemini_api_key_2"),
    )
    google_api_key_3: str | None = Field(
        default=None,
        validation_alias=AliasChoices("google_api_key_3", "gemini_api_key_3"),
    )
    llama_cloud_api_key_1: str | None = None
    llama_cloud_api_key_2: str | None = None
    firecrawl_api_key: str | None = None

    # Per-key Gemini rate limit and batch concurrency (tune to your tier).
    gemini_rpm_per_key: int = 10
    gemini_max_workers_per_key: int = 2

    gemini_model: str = "gemini-2.5-flash"
    # Judge model for RAGAS metric scoring. Unset means "reuse
    # gemini_model"; set RAGAS_JUDGE_MODEL to score with a different (e.g.
    # stronger or cheaper) model than the one generating the answers.
    ragas_judge_model: str | None = None
    voyage_embed_model: str = "voyage-3-large"
    voyage_rerank_model: str = "rerank-2.5"
    splade_model: str = "naver/splade-cocondenser-ensembledistil"

    # Dense embedding dimension. MUST match the halfvec(dim) columns in
    # schema.sql. Confirm against the configured voyage_embed_model; drop to
    # 512 (if the model supports output_dimension) to halve vector storage.
    voyage_embed_dim: int = 1024
    splade_vocab_size: int = 30522  # matches sparsevec(...) columns

    # Hierarchical chunking (PDF/HTML). Children are token-bounded windows for
    # context precision; parents are heading-delimited sections expanded via
    # parent_id for context recall. Sized in cl100k tokens.
    chunk_target_tokens: int = 512
    chunk_overlap_tokens: int = 64
    chunk_max_tokens: int = 800
    # Cap parent-section size so heading-less or very long sections don't
    # become a single giant parent (which dilutes context precision on
    # parent expansion). Oversized sections are split at element boundaries.
    chunk_parent_max_tokens: int = 2000

    # Embedding / indexing. Voyage dense + SPLADE sparse, written to the
    # halfvec/sparsevec columns. splade_max_terms keeps each sparse vector
    # under pgvector's 1000-nonzero HNSW limit (SPLADE typically emits far
    # fewer). Indexing is batched and resumable (only NULL rows are embedded).
    embed_batch_size: int = 64
    splade_batch_size: int = 16
    splade_max_terms: int = 512

    data_input_dir: Path = PROJECT_ROOT / "data" / "input"
    data_parsed_dir: Path = PROJECT_ROOT / "data" / "parsed"
    data_normalized_dir: Path = PROJECT_ROOT / "data" / "normalized"
    data_chunks_dir: Path = PROJECT_ROOT / "data" / "chunks"
    data_images_dir: Path = PROJECT_ROOT / "data" / "images"

    @property
    def gemini_api_keys(self) -> list[str]:
        """All configured Gemini keys, in rotation order, de-duplicated."""
        seen: dict[str, None] = {}
        for key in (self.google_api_key, self.google_api_key_2, self.google_api_key_3):
            if key and key not in seen:
                seen[key] = None
        return list(seen)


settings = Settings()
