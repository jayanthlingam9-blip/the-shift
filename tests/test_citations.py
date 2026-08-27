from src.generation.citations import build_citations


def test_build_citations() -> None:
    citations = build_citations([{"document_id": "doc-1", "chunk_id": "chunk-1", "metadata": {}}])
    assert citations[0].label == "[1]"
    assert citations[0].document_id == "doc-1"
