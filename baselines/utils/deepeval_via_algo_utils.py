"""Wrapper that evaluates baseline-generated QA outputs using the
`run_evaluation` correctness metric defined in
`evaluation_pipeline/deepeval_utils.py`.

Invoked from `run_deepeval.sh`. Does NOT score justification — the algo file
has no justification metric.
"""
import argparse
import importlib.util
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ALGO_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "evaluation_pipeline"))

# Load the baselines `third_party_api_client` explicitly by path so we don't
# pick up the algo dir's same-named module (which uses different env vars).
_baselines_client_path = os.path.join(_THIS_DIR, "third_party_api_client.py")
_spec = importlib.util.spec_from_file_location("baselines_third_party_api_client", _baselines_client_path)
_baselines_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baselines_client)
ASYNC_ROUTER_CLIENT = _baselines_client.ASYNC_ROUTER_CLIENT

# Now make `deepeval_utils` from the algo package importable.
sys.path.insert(0, _ALGO_DIR)
from deepeval_utils import run_evaluation  # noqa: E402


def _extract_groundtruth(item):
    if "groundtruth" in item:
        return item["groundtruth"]
    if "answers" in item:
        ans = item["answers"]
        if isinstance(ans, list) and ans:
            return ans[0]
        return ans or ""
    if "answer" in item:
        return item["answer"]
    return ""


def _score_for(test_result):
    """Pull (score, reason, success) from a deepeval TestResult."""
    md = getattr(test_result, "metrics_data", None) or []
    if md:
        m = md[0]
        return float(getattr(m, "score", 0.0) or 0.0), getattr(m, "reason", "") or "", bool(getattr(m, "success", False))
    return 0.0, "No metric data", False


def main():
    parser = argparse.ArgumentParser(description="Baseline QA eval via algo deepeval_utils")
    parser.add_argument("--generated_file_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--summary_file_path", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="google/gemini-2.5-flash-lite")
    parser.add_argument("--run_id", type=str, default="baselines-algo-eval")
    args = parser.parse_args()

    with open(args.generated_file_path, "r", encoding="utf-8") as f:
        qa_results = json.load(f)
    print(f"Loaded {len(qa_results)} QA results from {args.generated_file_path}")

    data_pairs = []
    for item in qa_results:
        data_pairs.append({
            "question": item["question"],
            "ans1": str(item.get("model_response", "")),
            "ans2": str(_extract_groundtruth(item)),
        })

    eval_results = run_evaluation(
        client=ASYNC_ROUTER_CLIENT,
        client_model_name=args.judge_model,
        data_pairs=data_pairs,
        run_id=args.run_id,
    )

    # `EvaluationResult` exposes `.test_results`; fall back to iteration if needed.
    test_results = getattr(eval_results, "test_results", None)
    if test_results is None:
        # The legacy iteration form: list of (key, [TestResult, ...]) tuples.
        flattened = []
        for entry in eval_results:
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], list):
                flattened.extend(entry[1])
                break
        test_results = flattened

    if len(test_results) != len(qa_results):
        print(f"WARNING: got {len(test_results)} test results for {len(qa_results)} inputs")

    detailed_results = []
    accuracy_scores = []
    question_types = {}
    for idx, original in enumerate(qa_results):
        if idx < len(test_results):
            score, reason, success = _score_for(test_results[idx])
        else:
            score, reason, success = 0.0, "No score recorded", False

        groundtruth = _extract_groundtruth(original)
        result_dict = {
            "question_no": original.get("question_no", idx + 1),
            "question": original.get("question", ""),
            "question_type": original.get("question_type", "Unknown"),
            "model_answer": original.get("model_response", ""),
            "model_justification": original.get("model_justification", "N/A"),
            "groundtruth": groundtruth,
            "groundtruth_justification": original.get("groundtruth_justification", ""),
            "accuracy_score": score,
            "accuracy_reason": reason,
            "is_correct": score == 1.0 or success,
            "sub_questions": original.get("sub_questions", []),
            "used_memory_module": len(original.get("sub_questions", [])) > 0,
        }
        detailed_results.append(result_dict)
        accuracy_scores.append(score)

        qtype = result_dict["question_type"]
        bucket = question_types.setdefault(qtype, {"total": 0, "correct": 0, "accuracy_scores": []})
        bucket["total"] += 1
        bucket["accuracy_scores"].append(score)
        if score == 1.0:
            bucket["correct"] += 1

    total = len(accuracy_scores)
    avg_accuracy = (sum(accuracy_scores) / total) if total else 0.0
    perfect = sum(1 for s in accuracy_scores if s == 1.0)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY (deepeval_utils correctness metric)")
    print("=" * 80)
    print(f"Total Questions: {total}")
    if total:
        print(f"Average Accuracy: {avg_accuracy:.3f}")
        print(f"Perfect (1.0):   {perfect} ({perfect / total * 100:.1f}%)")
    print("\nBreakdown by question type:")
    for qtype, stats in sorted(question_types.items()):
        avg_acc = sum(stats["accuracy_scores"]) / len(stats["accuracy_scores"])
        pct = stats["correct"] / stats["total"] * 100 if stats["total"] else 0.0
        print(f"  {qtype}: {stats['correct']}/{stats['total']} ({pct:.1f}%), avg={avg_acc:.3f}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)
    print(f"\nDetailed results -> {args.output_path}")

    summary = {
        "total_questions": total,
        "judge_model": args.judge_model,
        "metric_source": "evaluation_pipeline/deepeval_utils.py:run_evaluation",
        "accuracy": {
            "average_score": avg_accuracy,
            "perfect_answers": perfect,
            "perfect_rate": (perfect / total) if total else 0.0,
            "score_distribution": {
                "1.0": sum(1 for s in accuracy_scores if abs(s - 1.0) < 0.05),
                "0.0": sum(1 for s in accuracy_scores if abs(s - 0.0) < 0.05),
            },
        },
        "question_type_breakdown": {
            qtype: {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy_rate": (s["correct"] / s["total"]) if s["total"] else 0.0,
                "average_accuracy": (sum(s["accuracy_scores"]) / len(s["accuracy_scores"])) if s["accuracy_scores"] else 0.0,
            }
            for qtype, s in question_types.items()
        },
    }
    with open(args.summary_file_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"Summary -> {args.summary_file_path}")


if __name__ == "__main__":
    # `run_evaluation` internally drives async via deepeval; main is sync.
    main()
