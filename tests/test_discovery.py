from src.discovery.file_discovery import SUPPORTED_SUFFIXES


def test_supported_suffixes() -> None:
    assert ".pdf" in SUPPORTED_SUFFIXES
    assert ".html" in SUPPORTED_SUFFIXES
    assert ".csv" in SUPPORTED_SUFFIXES
