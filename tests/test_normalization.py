from pathlib import Path

from PIL import Image

from src.normalization import image_captioning


def test_caption_image_falls_back_without_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(image_captioning, "PROJECT_ROOT", tmp_path)
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (120, 120), color="white").save(image_path)

    result = image_captioning.caption_image(
        str(image_path),
        context={"alt_text": "Revenue chart by quarter"},
        cache_path=tmp_path / "cache.json",
        api_key="",
    )

    assert result["status"] == "missing_api_key"
    assert result["source"] == "fallback"
    assert "Revenue chart by quarter" in result["text"]


def test_caption_image_rejects_paths_outside_project(tmp_path: Path) -> None:
    image_path = tmp_path / "outside.png"
    Image.new("RGB", (120, 120), color="white").save(image_path)

    result = image_captioning.caption_image(
        str(image_path),
        context={"section_title": "Risk factors"},
        cache_path=tmp_path / "cache.json",
        api_key="",
    )

    assert result["status"] == "invalid_path"
    assert "Risk factors" in result["text"]
