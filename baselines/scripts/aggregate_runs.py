"""Aggregate per-run deepeval summaries into mean and sample std.

Reads N summary JSON files (each produced by `baselines/utils/deepeval_via_algo_utils.py`),
and computes accuracy as perfect_answers / total_questions, where total_questions
is the full benchmark size (successful + failed entries from the generated file),
not just the successful subset that deepeval scored.

Usage:
    python baselines/scripts/aggregate_runs.py \\
        --summary_files <s1.json> <s2.json> <s3.json> \\
        --output <combined.json>
"""
import argparse
import json
import statistics
from pathlib import Path


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_score(summary_path):
    """accuracy = perfect_answers / (len(success) + len(failed))."""
    summary = _load_json(summary_path)
    acc = summary.get("accuracy", {})
    perfect = acc.get("perfect_answers")
    if perfect is None:
        raise ValueError(f"{summary_path}: missing accuracy.perfect_answers")

    sp = str(summary_path)
    if not sp.endswith("_summary.json"):
        raise ValueError(f"Expected '_summary.json' suffix: {summary_path}")
    base = sp[: -len("_summary.json")]
    success_path = Path(base + ".json")
    failed_path = Path(base + "_failed.json")

    if not success_path.exists():
        raise FileNotFoundError(f"Generated file not found: {success_path}")
    n_success = len(_load_json(success_path))
    n_failed = len(_load_json(failed_path)) if failed_path.exists() else 0
    total = n_success + n_failed
    if total == 0:
        raise ValueError(f"{summary_path}: total_questions == 0")
    return {
        "perfect_answers": int(perfect),
        "n_success": n_success,
        "n_failed": n_failed,
        "total_questions": total,
        "average_score": float(perfect) / total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_files", nargs="+", required=True,
                    help="Per-run deepeval summary JSON files.")
    ap.add_argument("--output", required=True, help="Combined output JSON path.")
    args = ap.parse_args()

    if len(args.summary_files) < 1:
        raise ValueError("Need at least 1 summary file.")

    per_run = []
    for p in args.summary_files:
        rec = compute_score(p)
        rec["summary_file"] = str(p)
        per_run.append(rec)

    scores = [r["average_score"] for r in per_run]
    mean = statistics.fmean(scores)
    sstd = statistics.stdev(scores) if len(scores) >= 2 else None  # sample std (n-1); undefined for n<2

    out = {
        "n": len(scores),
        "mean": mean,
        "sample_std": sstd,
        "min": min(scores),
        "max": max(scores),
        "metric": "perfect_answers / total_questions (success + failed)",
        "per_run": per_run,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Aggregated {len(scores)} runs -> {args.output}")
    sstd_str = f"{sstd:.4f}" if sstd is not None else "n/a (single run)"
    print(f"  mean={mean:.4f}, sample_std={sstd_str}, scores={scores}")


if __name__ == "__main__":
    main()
