import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ragas_evaluator import run_ragas_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation of the RAG pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions.")
    parser.add_argument(
        "--raise-exceptions",
        action="store_true",
        help="Surface the underlying judge error instead of scoring NaN (use with a small --limit).",
    )
    parser.add_argument("--rerank-top-k", type=int, default=8)
    parser.add_argument("--route-top-k", type=int, default=2)
    parser.add_argument("--context-min-score", type=float, default=0.1)
    args = parser.parse_args()

    run_ragas_evaluation(
        rerank_top_k=args.rerank_top_k,
        route_top_k=args.route_top_k,
        context_min_score=args.context_min_score,
        limit=args.limit,
        raise_exceptions=args.raise_exceptions,
    )


if __name__ == "__main__":
    main()
