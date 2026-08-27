import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalization.normalize import normalize_all


def main() -> None:
    results = normalize_all()
    print(f"Normalized {len(results)} document outputs.")


if __name__ == "__main__":
    main()
