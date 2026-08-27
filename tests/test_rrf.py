from src.retrieval.rrf import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_orders_common_hits_first() -> None:
    rankings = [
        [{"chunk_id": "a"}, {"chunk_id": "b"}],
        [{"chunk_id": "b"}, {"chunk_id": "a"}],
    ]
    fused = reciprocal_rank_fusion(rankings)
    assert {item["chunk_id"] for item in fused} == {"a", "b"}
    assert all("rrf_score" in item for item in fused)
