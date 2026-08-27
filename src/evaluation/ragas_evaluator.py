"""RAGAS evaluation of the hybrid RAG pipeline.

Loads an eval dataset of {question, ground_truth} items, runs the full
answer_query pipeline for each question to collect the generated answer and the
retrieved contexts, then scores them with RAGAS (faithfulness, answer
relevancy, context precision, context recall). The judge LLM is Gemini and the
judge embeddings are Voyage -- the same providers the pipeline itself uses --
so no extra provider is introduced. Scores are reported overall and sliced by
the per-item `domain` tag (finance / healthcare / legal / out_of_scope).

The eval dataset lives in src/evaluation/eval_dataset.py (EVAL_DATASET, a list
of {question, ground_truth, domain} items). An empty list returns early without
calling any model.
"""

import json
import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from src.config import PROJECT_ROOT, settings
from src.embeddings.voyage_embeddings import embed_texts
from src.generation.answer import answer_query

DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "eval" / "ragas_results.json"


class _VoyageEmbeddings:
    """Minimal langchain-style embeddings adapter over Voyage for RAGAS."""

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text], input_type="query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(list(texts), input_type="document")


def _ensure_ragas_importable() -> None:
    """Make `import ragas` work on a modern langchain stack.

    ragas 0.4.3's ragas/llms/base.py hard-imports
    `langchain_community.chat_models.vertexai.ChatVertexAI`, a module that
    current langchain-community no longer ships. We drive ragas with the Gemini
    judge and never touch VertexAI, so we register a tiny stand-in module that
    only needs to expose the `ChatVertexAI` symbol for the import to succeed.
    Contained here (no site-packages patching) and a no-op once the real module
    exists again.
    """
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    try:
        __import__(name)
        return  # real module present -- nothing to shim
    except Exception:
        pass

    try:
        from langchain_google_vertexai import ChatVertexAI  # type: ignore
    except Exception:
        ChatVertexAI = type("ChatVertexAI", (), {})  # never instantiated by us

    shim = types.ModuleType(name)
    shim.ChatVertexAI = ChatVertexAI
    sys.modules[name] = shim


def _judge_keys() -> list[str]:
    """Gemini keys for the RAGAS judge.

    Generation already uses the first key, so the judge gets the remaining
    (otherwise-unused) keys to spread its much heavier load across separate
    per-key rate limits. Falls back to the single key if only one is configured.
    """
    keys = settings.gemini_api_keys
    if not keys:
        raise RuntimeError("No Gemini API keys configured (set GOOGLE_API_KEY*).")
    return keys[1:] if len(keys) > 1 else keys


def _round_robin_llm(wrappers: list):
    """Wrap several single-key judge LLMs, rotating one Gemini key per call."""
    import itertools
    import threading

    from ragas.llms.base import BaseRagasLLM

    class _RoundRobinLLM(BaseRagasLLM):
        def __init__(self, subs: list):
            self._subs = subs
            self._cycle = itertools.cycle(subs)
            self._lock = threading.Lock()
            self.run_config = subs[0].run_config
            self.multiple_completion_supported = False
            self.cache = None

        def _next(self):
            with self._lock:
                return next(self._cycle)

        def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            return self._next().generate_text(prompt, n, temperature, stop, callbacks)

        async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None):
            return await self._next().agenerate_text(prompt, n, temperature, stop, callbacks)

        def is_finished(self, response) -> bool:
            return self._subs[0].is_finished(response)

        def set_run_config(self, run_config) -> None:
            self.run_config = run_config
            for sub in self._subs:
                sub.set_run_config(run_config)

    return _RoundRobinLLM(wrappers)


def _build_judge():
    """Construct the (llm, embeddings) RAGAS judges. Imported lazily.

    The judge LLM round-robins across the Gemini keys not used by generation,
    so metric scoring is spread over multiple per-key rate limits.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    # bypass_n=True: Gemini rejects n>1 ("Multiple candidates is not enabled"),
    # which answer_relevancy/context_precision trigger. With bypass_n, ragas
    # issues n separate single-candidate calls instead of one n>1 call.
    wrappers = [
        LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=settings.gemini_model, google_api_key=key, temperature=0.0
            ),
            bypass_n=True,
        )
        for key in _judge_keys()
    ]
    llm = wrappers[0] if len(wrappers) == 1 else _round_robin_llm(wrappers)
    print(f"  [ragas] judge LLM rotating over {len(wrappers)} Gemini key(s)")
    return llm, LangchainEmbeddingsWrapper(_VoyageEmbeddings())


def _load_dataset() -> list[dict]:
    from src.evaluation.eval_dataset import EVAL_DATASET

    return list(EVAL_DATASET)


_CITATION_RE = re.compile(r"\s*\[\d+\]")


def _strip_citations(answer: str) -> str:
    """Drop inline [n] citation markers before scoring.

    The markers are provenance, not content; leaving them in adds embedding
    noise that depresses answer_relevancy (and is irrelevant to faithfulness).
    """
    return _CITATION_RE.sub("", answer).strip()


def run_ragas_evaluation(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    rerank_top_k: int = 8,
    route_top_k: int = 2,
    context_min_score: float = 0.1,
    limit: int | None = None,
    raise_exceptions: bool = False,
) -> dict:
    """Run the pipeline over the eval dataset and score it with RAGAS.

    `limit` evaluates only the first N items (quick diagnostic runs).
    `raise_exceptions=True` surfaces the underlying judge error instead of
    silently scoring NaN -- use it on a small `limit` to see why a metric
    (e.g. context_precision) fails to compute.
    """
    items = _load_dataset()
    if limit:
        items = items[:limit]
    if not items:
        print("Eval dataset is empty -- nothing to evaluate.")
        return {}

    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from .env")

    # Import RAGAS + build the judges up front so a missing/broken eval
    # dependency fails fast, before spending real API calls on the pipeline loop.
    _ensure_ragas_importable()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    llm, embeddings = _build_judge()

    questions, answers, contexts, ground_truths, domains = [], [], [], [], []
    with psycopg.connect(database_url) as connection:
        register_vector(connection)
        for item in items:
            result = answer_query(
                connection,
                item["question"],
                rerank_top_k=rerank_top_k,
                route_top_k=route_top_k,
                context_min_score=context_min_score,
            )
            questions.append(item["question"])
            answers.append(_strip_citations(result["answer"]))
            contexts.append(result["contexts"])
            ground_truths.append(item.get("ground_truth", ""))
            domains.append(item.get("domain", "unknown"))

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # Spread judge calls across the rotated keys; surface errors when asked.
    from ragas.run_config import RunConfig

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=len(_judge_keys()) * 4),
        raise_exceptions=raise_exceptions,
    )

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    frame = scores.to_pandas()
    frame["domain"] = domains

    # Persist per-row scores so NaN/low rows can be inspected per question.
    # ragas 0.4.3's to_pandas() uses the new schema names (user_input/...), so
    # pick whichever question column exists rather than hardcoding one.
    per_row_path = output_path.with_name("ragas_per_row.csv")
    question_col = next((c for c in ("question", "user_input") if c in frame.columns), None)
    diag_cols = [
        "domain",
        *([question_col] if question_col else []),
        *[m for m in metric_names if m in frame.columns],
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame[diag_cols].to_csv(per_row_path, index=False, encoding="utf-8")

    summary = _summarize(frame, metric_names)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_summary(summary, metric_names)
    print(f"Wrote {output_path}")
    print(f"Wrote {per_row_path} (per-question scores for diagnosis)")
    return summary


def _mean(series) -> float | None:
    """Mean of a metric column, ignoring NaN; None if nothing scored."""
    clean = series.dropna()
    return round(float(clean.mean()), 4) if len(clean) else None


def _slice_scores(group, present: list[str]) -> dict:
    scores = {name: _mean(group[name]) for name in present}
    scores["count"] = int(len(group))
    return scores


def _summarize(frame, metric_names: list[str]) -> dict:
    """Aggregate RAGAS scores overall, in-scope only, and sliced by domain.

    `in_scope` excludes the out_of_scope refusal questions, whose
    answer_relevancy (≈0 by design for a correct refusal) and context_precision
    are artifacts that distort the headline numbers.
    """
    present = [name for name in metric_names if name in frame.columns]

    overall = _slice_scores(frame, present)
    in_scope_frame = frame[frame["domain"] != "out_of_scope"]
    in_scope = _slice_scores(in_scope_frame, present) if len(in_scope_frame) else {}

    by_domain: dict[str, dict] = {}
    for domain, group in frame.groupby("domain"):
        by_domain[str(domain)] = _slice_scores(group, present)

    # How many rows each metric actually scored (rest are NaN = judge errors).
    coverage = {
        name: {"scored": int(frame[name].notna().sum()), "total": int(len(frame))}
        for name in present
    }

    return {
        "overall": overall,
        "in_scope": in_scope,
        "by_domain": by_domain,
        "coverage": coverage,
    }


def _print_summary(summary: dict, metric_names: list[str]) -> None:
    """Print an overall + per-domain table of metric scores."""
    present = [name for name in metric_names if name in summary["overall"]]
    header = f"{'slice':<14}{'n':>4}  " + "  ".join(f"{name[:16]:>16}" for name in present)
    print("\n" + header)
    print("-" * len(header))

    def row(label: str, scores: dict) -> str:
        cells = []
        for name in present:
            value = scores.get(name)
            cells.append(f"{value:>16.4f}" if value is not None else f"{'-':>16}")
        return f"{label:<14}{scores.get('count', 0):>4}  " + "  ".join(cells)

    print(row("OVERALL", summary["overall"]))
    if summary.get("in_scope"):
        print(row("IN_SCOPE", summary["in_scope"]))
    print("-" * len(header))
    for domain in sorted(summary["by_domain"]):
        print(row(domain, summary["by_domain"][domain]))

    coverage = summary.get("coverage", {})
    if coverage:
        print("\nscored (non-NaN) rows per metric -- low coverage = judge errors:")
        for name in present:
            cov = coverage.get(name, {})
            print(f"  {name:<20} {cov.get('scored', 0)}/{cov.get('total', 0)}")
