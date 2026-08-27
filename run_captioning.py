"""Run filter -> caption -> normalize across all parsed docs, with progress."""

import src.normalization.normalize as N
from src.normalization.image_captioning import caption_images as _caption_images


def _caption_with_progress(batch, **kwargs):
    def _progress(done, total, path, result):
        name = path.replace("\\", "/").split("/")[-1]
        print(f"[{done}/{total}] {result['status']:16} {name}", flush=True)

    kwargs["on_progress"] = _progress
    return _caption_images(batch, **kwargs)


# Route normalize's captioning through the progress-printing wrapper.
N.caption_images = _caption_with_progress


if __name__ == "__main__":
    counts = N.normalize_all()
    print("\nDone. Elements per document:")
    for doc, n in counts.items():
        print(f"  {doc}: {n}")
