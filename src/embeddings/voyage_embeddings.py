"""Dense embeddings via Voyage (text-only: voyage-3-large, 1024 dims).

Images were captioned into text during normalization, so a text model covers
every retrieval unit. output_dimension is pinned to settings.voyage_embed_dim
to match the halfvec(1024) columns.
"""

import time

from src.config import settings

_client = None
_MAX_RETRIES = 10


def _get_client():
    global _client
    if _client is None:
        import voyageai

        _client = voyageai.Client(api_key=settings.voyage_api_key)
    return _client


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed a batch of texts. Caller is responsible for sane batch sizes.

    Retries transient network/rate-limit errors with exponential backoff so a
    single dropped connection doesn't kill a long indexing run.
    """
    if not settings.voyage_api_key:
        raise RuntimeError("VOYAGE_API_KEY is required.")
    if not texts:
        return []

    import voyageai.error as voyage_error

    transient = (
        voyage_error.APIConnectionError,
        voyage_error.RateLimitError,
        voyage_error.ServerError,
        voyage_error.Timeout,
    )

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            result = _get_client().embed(
                texts,
                model=settings.voyage_embed_model,
                input_type=input_type,
                output_dimension=settings.voyage_embed_dim,
                truncation=True,
            )
            return result.embeddings
        except transient as exc:
            last_error = exc
            wait = min(2**attempt, 60)
            print(f"  [voyage] transient error ({type(exc).__name__}); retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"Voyage embed failed after {_MAX_RETRIES} retries") from last_error
