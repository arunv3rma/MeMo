"""
BrowseComp+ Evaluation Script — No Context
===========================================
Variant of main_for_bcp.py that sends ONLY the raw question to the model,
with no evidence documents. Useful as a closed-book / parametric-knowledge
baseline to compare against the evidence-docs version.

Answers are batch-judged using run_evaluation from
evaluation_pipeline/deepeval_utils.py.

Usage (OpenRouter):
    python main_for_bcp_no_context.py \
        --model "anthropic/claude-3.5-sonnet" \
        --questions_file /path/to/browsecomp_plus_questions.jsonl \
        --max_concurrent 30

Usage (vLLM):
    python main_for_bcp_no_context.py \
        --base_url http://localhost:4325/v1 \
        --model auto \
        --questions_file /path/to/browsecomp_plus_questions.jsonl \
        --max_concurrent 32

Requirements:
    pip install openai tqdm python-dotenv deepeval

Set OPENROUTER_API_KEY in your environment or a .env file (required for judging).
"""

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../evaluation_pipeline"))
from deepeval_utils import run_evaluation as deepeval_run_evaluation

load_dotenv()

JUDGE_MODEL = "google/gemini-2.5-flash-lite"

_THIS_DIR = os.path.dirname(__file__)
DEFAULT_QUERY_IDS_FILE = os.path.join(
    _THIS_DIR, "../../data_synthesis_pipeline/data_subsets/bcp_300_queries_id.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_model(model: str, base_url: str) -> str:
    """If model is 'auto', query the /models endpoint and return the first model ID."""
    if model != "auto":
        return model
    import urllib.request
    url = base_url.rstrip("/").removesuffix("/v1") + "/v1/models"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
    model_id = data["data"][0]["id"]
    print(f"  Auto-detected model: {model_id}")
    return model_id


def load_query_ids(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))


def load_questions(path: str, allowed_ids: set[str], max_questions: int | None = None) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("query_id") not in allowed_ids:
                continue
            questions.append({
                "question_no": data.get("query_id"),
                "question": data.get("query"),
                "groundtruth": data.get("answer"),
            })
            if max_questions is not None and len(questions) >= max_questions:
                break
    return questions


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """\
{question}

Instructions:
- Answer the question using your knowledge.
- Think step by step before answering.
- Be concise and direct in your final answer.
"""


def build_answer_prompt(question: str) -> str:
    return ANSWER_PROMPT.format(question=question)


# ---------------------------------------------------------------------------
# Single-example async processing
# ---------------------------------------------------------------------------


async def process_entry(
    answer_client: AsyncOpenAI,
    model_id: str,
    entry: dict,
    idx: int,
    max_tokens: int,
    answer_semaphore: asyncio.Semaphore,
) -> dict:
    result = {
        "id": entry["question_no"],
        "idx": idx,
        "question": entry["question"],
        "groundtruth": entry["groundtruth"],
        "model_response_raw": None,
        "score": None,
        "judge_response_raw": None,
        "error": None,
    }

    prompt = build_answer_prompt(entry["question"])

    try:
        async with answer_semaphore:
            response = await answer_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
        result["model_response_raw"] = response.choices[0].message.content if response.choices else ""
        print(f"  [{idx}] answer collected")

    except Exception as e:
        result["error"] = str(e)
        result["score"] = 0.0
        print(f"  [{idx}] ERROR: {e}", file=sys.stderr)

    return result


# ---------------------------------------------------------------------------
# Async answer collection loop
# ---------------------------------------------------------------------------


async def collect_answers(args, questions: list[dict]) -> list[dict]:
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or "no-key"
    answer_client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=openrouter_key if args.base_url == "https://openrouter.ai/api/v1" else "EMPTY",
    )
    answer_semaphore = asyncio.Semaphore(args.max_concurrent)

    tasks = [
        process_entry(
            answer_client, args.model, entry, idx,
            args.max_tokens, answer_semaphore,
        )
        for idx, entry in enumerate(questions)
    ]

    results = []
    for coro in async_tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Collecting answers from {args.model} on BrowseComp+ (no context)",
    ):
        results.append(await coro)

    results.sort(key=lambda r: r["idx"])
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(results: list[dict]) -> dict:
    errored = [r for r in results if r.get("error") is not None]
    scored = [r for r in results if r.get("score") is not None and r.get("error") is None]
    avg_score = sum(r["score"] for r in scored) / len(scored) if scored else None

    return {
        "benchmark": "browsecomp_plus",
        "context": "none (closed-book)",
        "total_attempted": len(results),
        "total_scored": len(scored),
        "total_errored": len(errored),
        "avg_score_pct": round(avg_score * 100, 2) if avg_score is not None else None,
        "judge_model": JUDGE_MODEL,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on BrowseComp+ (no context, closed-book, LLM-judged)"
    )
    parser.add_argument(
        "--model", type=str, default="openai/gpt-4o-mini",
        help="Model ID, or 'auto' to detect from the base_url /v1/models endpoint.",
    )
    parser.add_argument(
        "--base_url", type=str, default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible API base URL (e.g. http://localhost:4325/v1 for vLLM).",
    )
    parser.add_argument(
        "--questions_file", type=str, required=True,
        help="Path to the BrowseComp+ JSONL questions file.",
    )
    parser.add_argument(
        "--query_ids_file", type=str, default=DEFAULT_QUERY_IDS_FILE,
        help="JSON file containing the list of query IDs to evaluate (default: bcp_300_queries_id.json).",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--metrics_output", type=str, default=None)
    parser.add_argument(
        "--max_questions", type=int, default=None,
        help="Cap number of questions (useful for testing).",
    )
    parser.add_argument("--max_concurrent", type=int, default=30)
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()
    args.model = resolve_model(args.model, args.base_url)

    print(f"\n{'=' * 60}")
    print("BrowseComp+ Evaluation  (no context / closed-book)")
    print(f"{'=' * 60}")

    model_dir = args.model.replace("/", "_")
    if args.base_url != "https://openrouter.ai/api/v1":
        model_dir += "_vllm"
    os.makedirs(model_dir, exist_ok=True)
    output_path = args.output or os.path.join(model_dir, "bcp_results_no_context.json")
    metrics_path = args.metrics_output or os.path.join(model_dir, "bcp_metrics_no_context.json")

    allowed_ids = load_query_ids(args.query_ids_file)
    if args.max_questions is not None and args.max_questions > len(allowed_ids):
        print(f"  WARNING: --max_questions {args.max_questions} exceeds query ID list size ({len(allowed_ids)}); clamping to {len(allowed_ids)}", file=sys.stderr)
    print(f"Loading questions from {args.questions_file} ...")
    print(f"  Filtering to {len(allowed_ids)} query IDs from {args.query_ids_file}")
    questions = load_questions(args.questions_file, allowed_ids, max_questions=args.max_questions)
    print(f"  Loaded {len(questions)} questions")

    print(f"\nModel           : {args.model}")
    print(f"Judge model     : {JUDGE_MODEL}")
    print(f"Max concurrent  : {args.max_concurrent}")
    print(f"Max resp tokens : {args.max_tokens}")
    print(f"Context         : none (closed-book)\n")

    # Step 1: collect model answers
    results = asyncio.run(collect_answers(args, questions))

    # Step 2: batch judge with deepeval
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or "no-key"
    judge_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

    scoreable = [(i, r) for i, r in enumerate(results) if r.get("model_response_raw") and not r.get("error")]
    data_pairs = [
        {"question": r["question"], "ans1": r["model_response_raw"], "ans2": r["groundtruth"]}
        for _, r in scoreable
    ]

    print(f"\nJudging {len(data_pairs)} answers with {JUDGE_MODEL} ...")
    eval_results = deepeval_run_evaluation(judge_client, JUDGE_MODEL, data_pairs)

    scores = []
    for res in eval_results:
        _, result_list = res
        for test_result in result_list:
            metric = test_result.metrics_data[0]
            scores.append({"score": float(metric.score), "reason": metric.reason or ""})
        break

    for (i, _), ev in zip(scoreable, scores):
        results[i]["score"] = ev["score"]
        results[i]["judge_response_raw"] = ev["reason"]

    metrics = compute_metrics(results)
    metrics["model"] = args.model

    print(f"\n{'=' * 60}")
    print("RESULTS — BrowseComp+  (no context / closed-book)")
    print(f"{'=' * 60}")
    print(f"Model           : {args.model}")
    print(f"Total attempted : {metrics['total_attempted']}")
    print(f"Total scored    : {metrics['total_scored']}")
    print(f"Errors          : {metrics['total_errored']}")
    if metrics["avg_score_pct"] is not None:
        print(f"Avg score       : {metrics['avg_score_pct']}%")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nPer-query results saved to : {output_path}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Aggregated metrics saved to : {metrics_path}")


if __name__ == "__main__":
    main()
