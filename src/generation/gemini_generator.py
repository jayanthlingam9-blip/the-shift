"""Grounded answer generation via Gemini (gemini-2.5-flash).

Answers strictly from the numbered context produced by
generation.context_builder.build_context, citing the same [n] markers so the
answer stays traceable. Uses the google.genai client exactly as
normalization.image_captioning does, with simple retry across the configured
Gemini keys for transient/rate-limit errors.
"""

import time

from src.config import settings

_SYSTEM_INSTRUCTION = """You are a precise question-answering assistant over a \
finance, healthcare, and legal knowledge base. Follow these rules:
- Answer ONLY from the numbered sources in the context; never use outside \
knowledge or invent facts.
- Lead with a direct, complete answer to the question. Restate the key entities \
from the question so the answer stands on its own, then give the specific \
figures/facts that answer it.
- Be concise and on-topic: include only what the question asks for, with no \
preamble like "Based on the context" and no unrelated detail from the sources.
- Cite each fact with the matching marker, e.g. [1] or [2], using the exact \
numbers shown.
- ONLY if none of the sources contain the answer, reply exactly: "I don't have \
enough information in the retrieved context to answer that." Do not refuse when \
the answer is present."""

_MAX_RETRIES = 6


def _build_prompt(query: str, context: str) -> str:
    return (
        f"Context (numbered sources):\n{context}\n\n"
        f"Question: {query}\n\n"
        "Give a direct, self-contained answer using only the sources above, "
        "citing markers like [1]. Do not add information the question did not ask for."
    )


def generate_answer(query: str, context: str) -> str:
    """Generate a grounded, cited answer for `query` over `context`."""
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is required.")
    if not context.strip():
        return "I don't have enough information in the retrieved context to answer that."

    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types

    prompt = _build_prompt(query, context)
    keys = settings.gemini_api_keys or [settings.google_api_key]

    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        key = keys[attempt % len(keys)]
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    temperature=0.0,
                ),
            )
            text = (getattr(response, "text", None) or "").strip()
            if text:
                return text
            raise RuntimeError("empty generation response")
        except genai_errors.APIError as exc:
            last_error = exc
            wait = min(2**attempt, 30)
            print(f"  [gemini] API error ({type(exc).__name__}); retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError(
        f"Gemini generation failed after {_MAX_RETRIES} retries"
    ) from last_error
