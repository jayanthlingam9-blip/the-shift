def reciprocal_rank_fusion(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            item_id = item["chunk_id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            items[item_id] = item
    return sorted(({**items[item_id], "rrf_score": score} for item_id, score in scores.items()), key=lambda x: x["rrf_score"], reverse=True)
