"""Sparse embeddings via SPLADE (naver/splade-cocondenser-ensembledistil).

Each text maps to a {token_id: weight} dict over the 30,522-term BERT vocab,
which lands directly in the sparsevec(30522) columns (0-based indices). Weights
are SPLADE's max-pooled log(1 + relu(MLM logits)) over the (masked) sequence.

Kept to the top splade_max_terms weights per text so no vector exceeds
pgvector's 1000-nonzero HNSW limit (SPLADE usually emits far fewer anyway).
"""

from functools import lru_cache

import torch

from src.config import settings


@lru_cache(maxsize=1)
def _load_model():
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.splade_model)
    model = AutoModelForMaskedLM.from_pretrained(settings.splade_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return tokenizer, model, device


@torch.no_grad()
def embed_sparse(
    texts: list[str],
    batch_size: int | None = None,
    max_length: int = 512,
) -> list[dict[int, float]]:
    if not texts:
        return []

    batch_size = batch_size or settings.splade_batch_size
    max_terms = settings.splade_max_terms
    tokenizer, model, device = _load_model()

    out: list[dict[int, float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        logits = model(**encoded).logits  # [B, T, V]
        weighted = torch.log1p(torch.relu(logits))
        masked = weighted * encoded["attention_mask"].unsqueeze(-1)
        vectors, _ = masked.max(dim=1)  # [B, V]

        for vector in vectors:
            nonzero = torch.nonzero(vector, as_tuple=False).squeeze(-1)
            if nonzero.numel() > max_terms:
                values = vector[nonzero]
                top = torch.topk(values, max_terms).indices
                nonzero = nonzero[top]
            out.append(
                {int(idx): float(vector[idx]) for idx in nonzero.tolist()}
            )
    return out
