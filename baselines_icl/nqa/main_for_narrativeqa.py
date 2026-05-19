"""
NarrativeQA Evaluation Script — With Full Story Context
========================================================
Evaluates any OpenAI-compatible model against the NarrativeQA benchmark.

For each question the model is given the full story text as context and asked
to answer in free text. Answers are then batch-judged using
run_evaluation_multi_answer from evaluation_pipeline/deepeval_utils.py.

Usage (OpenRouter):
    python main_for_narrativeqa.py \\
        --model "anthropic/claude-3.5-sonnet" \\
        --split valid \\
        --max_questions 500 \\
        --max_docs 10 \\
        --max_concurrent 8

Usage (vLLM):
    python main_for_narrativeqa.py \\
        --base_url http://localhost:4325/v1 \\
        --model auto \\
        --split valid \\
        --max_concurrent 10

Requirements:
    pip install openai tqdm python-dotenv deepeval

Set OPENROUTER_API_KEY in your environment or a .env file (required for judging).
"""

import argparse
import asyncio
import csv
import json
import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../evaluation_pipeline"))
from deepeval_utils import run_evaluation_multi_answer

load_dotenv()

NARRATIVEQA_DIR = os.path.expanduser("~/narrativeqa-master")
JUDGE_MODEL = "google/gemini-2.5-flash-lite"

DOC_ID_SUBSET = [
    "4b30ab1c49b62dc59b9773954958d9ac6807a865",
    "4f485054f9d450534fddba184f0996e32575d1be",
    "5339e9db4aca0b74b644e736274989344864f0ba",
    "86fc18a4b6b2079beea7ee8c92ec1850d24a04e2",
    "8b801dd3b69733e83206589fe5646f63838e5e1b",
    "94dc6d01df88f97843813efb11e5d7e48562c869",
    "c7c075c49018828bf6027da5c5534834779d1adf",
    "ea371ba9cc41bf32aa8b37549662d2d3a38f084d",
    "ebbac675f4d13527d9d6af2aeddff8f92b7e3e4d",
    "ffa719867c77cfd8fc661fdb6d8c8d266746e15f",
]

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


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """\
Use the following story to answer the question.

Story:
{story}

Question: {question}

Instructions:
- Use the story to answer the question directly.
- Think step by step before answering.
- Be concise and direct in your final answer.
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_questions(data_dir: str, split: str) -> list[dict]:
    qaps_path = os.path.join(data_dir, "qaps.csv")
    questions = []
    with open(qaps_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["set"] != split:
                continue
            questions.append({
                "document_id": row["document_id"],
                "question": row["question"],
                "answer1": row["answer1"],
                "answer2": row["answer2"],
            })
    return questions


def load_story(data_dir: str, document_id: str) -> str | None:
    path = os.path.join(data_dir, "tmp", f"{document_id}.content")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_answer_prompt(question: str, story: str) -> str:
    return ANSWER_PROMPT.format(story=story, question=question)


# ---------------------------------------------------------------------------
# Single-example async processing
# ---------------------------------------------------------------------------


async def process_entry(
    client: AsyncOpenAI,
    model_id: str,
    entry: dict,
    story: str | None,
    idx: int,
    max_tokens: int,
    answer_semaphore: asyncio.Semaphore,
) -> dict:
    question = entry["question"]

    result = {
        "idx": idx,
        "document_id": entry["document_id"],
        "question": question,
        "answer1": entry["answer1"],
        "answer2": entry["answer2"],
        "model_response_raw": None,
        "score": None,
        "judge_response_raw": None,
        "error": None,
    }

    if story is None:
        result["error"] = "story_file_not_found"
        result["score"] = 0.0
        print(f"  [{idx}] ERROR: story file not found for {entry['document_id']}", file=sys.stderr)
        return result

    prompt = build_answer_prompt(question, story)

    try:
        async with answer_semaphore:
            response = await client.chat.completions.create(
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


async def collect_answers(
    args, questions: list[dict], stories: dict[str, str | None]
) -> list[dict]:
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or "no-key"
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=openrouter_key if args.base_url == "https://openrouter.ai/api/v1" else "EMPTY",
    )
    answer_semaphore = asyncio.Semaphore(args.max_concurrent)

    tasks = [
        process_entry(
            client, args.model, entry,
            stories.get(entry["document_id"]),
            idx, args.max_tokens,
            answer_semaphore,
        )
        for idx, entry in enumerate(questions)
    ]

    results = []
    for coro in async_tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Collecting answers from {args.model} on NarrativeQA (full story)",
    ):
        results.append(await coro)

    results.sort(key=lambda r: r["idx"])
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(results: list[dict]) -> dict:
    errored = [r for r in results if r.get("error") is not None]
    scored = [r for r in results if r.get("score") is not None and not r.get("error")]
    avg_score = sum(r["score"] for r in scored) / len(scored) if scored else None

    return {
        "benchmark": "narrativeqa",
        "context": "full_story",
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
        description="Evaluate an OpenRouter model on NarrativeQA (full story context)"
    )
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini",
                        help="Model ID, or 'auto' to detect from the base_url /v1/models endpoint.")
    parser.add_argument(
        "--base_url", type=str, default="https://openrouter.ai/api/v1",
        help="OpenAI-compatible API base URL (e.g. http://localhost:4325/v1 for vLLM).",
    )
    parser.add_argument(
        "--split", type=str, default="all", choices=["train", "valid", "test", "all"],
    )
    parser.add_argument("--data_dir", type=str, default=NARRATIVEQA_DIR)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--max_concurrent", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()
    args.model = resolve_model(args.model, args.base_url)

    splits_to_run = ["train", "valid", "test"] if args.split == "all" else [args.split]

    model_dir = os.path.basename(args.model.rstrip("/"))
    if args.base_url != "https://openrouter.ai/api/v1":
        model_dir += "_vllm"
    else:
        model_dir = args.model.replace("/", "_")
    os.makedirs(model_dir, exist_ok=True)

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment or .env file.")
    judge_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)

    all_metrics: dict[str, dict] = {}

    for split in splits_to_run:
        print(f"\n{'=' * 60}")
        print(f"NarrativeQA split: {split}  (full story)")
        print(f"{'=' * 60}")

        output_path = os.path.join(model_dir, f"narrativeqa_{split}_results.json")
        metrics_path = os.path.join(model_dir, f"narrativeqa_{split}_metrics.json")

        questions = load_questions(args.data_dir, split)
        print(f"  Questions loaded: {len(questions):,}")

        n_docs = args.max_docs if args.max_docs is not None else len(DOC_ID_SUBSET)
        if n_docs > len(DOC_ID_SUBSET):
            print(f"  WARNING: --max_docs {n_docs} exceeds DOC_ID_SUBSET size ({len(DOC_ID_SUBSET)}); clamping to {len(DOC_ID_SUBSET)}", file=sys.stderr)
            n_docs = len(DOC_ID_SUBSET)
        allowed_ids = set(DOC_ID_SUBSET[:n_docs])
        questions = [q for q in questions if q["document_id"] in allowed_ids]
        print(f"  Filtered to {len(questions)} questions from {n_docs} doc ID(s)")

        if args.max_questions:
            questions = questions[: args.max_questions]
            print(f"  Capped to {len(questions)} (--max_questions)")

        doc_ids = {q["document_id"] for q in questions}
        stories = {doc_id: load_story(args.data_dir, doc_id) for doc_id in doc_ids}
        missing = sum(1 for v in stories.values() if v is None)
        print(f"  Stories loaded: {len(stories) - missing}  Missing: {missing}")

        print(f"\nModel           : {args.model}")
        print(f"Judge model     : {JUDGE_MODEL}")
        print(f"Max concurrent  : {args.max_concurrent}")
        print(f"Max resp tokens : {args.max_tokens}\n")

        # Step 1: collect model answers
        results = asyncio.run(collect_answers(args, questions, stories))

        # Step 2: batch judge with deepeval
        scoreable = [(i, r) for i, r in enumerate(results) if r.get("model_response_raw") and not r.get("error")]
        data_pairs = [
            {"question": r["question"], "ans1": r["model_response_raw"], "ans2": [r["answer1"], r["answer2"]]}
            for _, r in scoreable
        ]

        print(f"\nJudging {len(data_pairs)} answers with {JUDGE_MODEL} ...")
        eval_results = run_evaluation_multi_answer(judge_client, JUDGE_MODEL, data_pairs)

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
        metrics["split"] = split
        all_metrics[split] = metrics

        print(f"\nRESULTS — NarrativeQA-{split}  (full story)")
        print(f"Model              : {args.model}")
        print(f"Total attempted    : {metrics['total_attempted']}")
        print(f"Total scored       : {metrics['total_scored']}")
        print(f"Errors             : {metrics['total_errored']}")
        if metrics["avg_score_pct"] is not None:
            print(f"Avg score          : {metrics['avg_score_pct']}%")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nPer-query results saved to  : {output_path}")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Aggregated metrics saved to : {metrics_path}")

    if len(splits_to_run) > 1:
        print(f"\n{'=' * 60}")
        print(f"OVERALL SUMMARY  (full story)")
        print(f"{'=' * 60}")
        print(f"{'Split':<8} {'Score':>10} {'Scored':>8} {'Errors':>8}")
        print(f"{'-' * 40}")
        for split, m in all_metrics.items():
            score_str = f"{m['avg_score_pct']:.2f}%" if m["avg_score_pct"] is not None else "N/A"
            print(
                f"  {split:<6} {score_str:>10}"
                f" {m['total_scored']:>8} {m['total_errored']:>8}"
            )
        combined_path = os.path.join(model_dir, "narrativeqa_all_metrics.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "judge": JUDGE_MODEL, "splits": all_metrics},
                      f, indent=2, ensure_ascii=False)
        print(f"\nCombined metrics saved to : {combined_path}")


if __name__ == "__main__":
    main()
