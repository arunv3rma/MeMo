#!/usr/bin/env python3
"""
Run a sweep of all merge methods across N (>=2) fine-tuned MEMORY checkpoints
trained on disjoint corpora. Uploads each merged model to HuggingFace and
deletes the local copy.

Sweep:
  linear      — all N models, equal weights
  slerp       — every pair of models, t in {0.3, 0.5, 0.7}      (2 models => 1 pair)
  task        — all N models, equal weights
  ties        — all N models, density in {0.3, 0.5, 0.7}
  dare_ties   — all N models, density in {0.3, 0.5, 0.7}
  dare_linear — all N models, density in {0.3, 0.5, 0.7}

Run `huggingface-cli login` before invoking.

Usage:
  python run_sweep.py \\
      --hf-user <hf-user> \\
      --hf-base-name <merge-name-prefix> \\
      --base <base-model-id> \\
      --models <hf-user>/<m1> <hf-user>/<m2> [<hf-user>/<m3> ...] \\
      [--labels corpus1 corpus2 ...]
"""

import argparse
import itertools
import os
import shutil
import sys
import traceback
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_merge.model import Model

DENSITIES = [0.3, 0.5, 0.7]
SLERP_T   = [0.3, 0.5, 0.7]


def make_repo_name(cfg: dict, hf_base: str, labels: dict) -> str:
    method  = cfg["method"]
    density = cfg.get("density")
    t       = cfg.get("t")
    models  = cfg["models"]

    if method == "linear":
        return f"{hf_base}-linear"
    if method == "task":
        return f"{hf_base}-task"
    if method == "slerp":
        pair = "_".join(labels[m] for m in models)
        return f"{hf_base}-slerp-{pair}-t{t}"
    if density:
        return f"{hf_base}-{method}-d{density[0]}"
    return f"{hf_base}-{method}"


def build_configs(models: list, base: str) -> list:
    pairs = list(itertools.combinations(models, 2))
    w = [1] * len(models)

    configs = []
    configs.append(dict(method="linear", models=models, weights=w))

    for (ma, mb) in pairs:
        for t in SLERP_T:
            configs.append(dict(method="slerp", models=[ma, mb], t=t))

    configs.append(dict(method="task", models=models, weights=w, base=base))

    for method in ("ties", "dare_ties", "dare_linear"):
        for den in DENSITIES:
            d = [den] * len(models)
            configs.append(dict(method=method, models=models, weights=w, density=d, base=base))

    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True,
                        help="Source MEMORY checkpoints (HF repo IDs or local paths). Need at least 2.")
    parser.add_argument("--base", required=True,
                        help="Base model used for task-vector methods (task / ties / dare_*).")
    parser.add_argument("--hf-user", required=True,
                        help="HuggingFace username/org to upload merged models under.")
    parser.add_argument("--hf-base-name", required=True,
                        help="Prefix for merged-model repo names, e.g. 'MyMemoryModel-corpus_1_2_3-merge'.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Short labels for each source model (used in SLERP repo names). "
                             "Defaults to model1, model2, ...")
    parser.add_argument("--temp-dir", default=None,
                        help="Local scratch directory for intermediate merged models. "
                             "Defaults to ./_merge_tmp next to this script.")
    args = parser.parse_args()

    if len(args.models) < 2:
        parser.error("--models needs at least 2 entries.")
    if args.labels is None:
        labels_list = [f"model{i+1}" for i in range(len(args.models))]
    else:
        if len(args.labels) != len(args.models):
            parser.error("--labels must have the same number of entries as --models.")
        labels_list = args.labels
    labels = dict(zip(args.models, labels_list))

    temp_dir = args.temp_dir or str(Path(__file__).resolve().parent / "_merge_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    configs = build_configs(args.models, args.base)
    api     = HfApi()
    total   = len(configs)
    print(f"Total merges to run: {total}\n")

    results = []
    for i, cfg in enumerate(configs, 1):
        method    = cfg["method"]
        models    = cfg["models"]
        weights   = cfg.get("weights")
        density   = cfg.get("density")
        t         = cfg.get("t", 0.5)
        base      = cfg.get("base")
        repo_name = make_repo_name(cfg, args.hf_base_name, labels)
        hf_repo   = f"{args.hf_user}/{repo_name}"

        print(f"\n{'='*60}\n[{i}/{total}] {hf_repo}\n{'='*60}")
        try:
            model = Model(models, method=method, output_dir=temp_dir, base_model=base)
            model.merge(t=t, weights=weights, density=density)
            local_path = model.save()

            print(f"Uploading to https://huggingface.co/{hf_repo} ...")
            api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="model")
            api.upload_folder(folder_path=local_path, repo_id=hf_repo, repo_type="model")
            print(f"Uploaded: https://huggingface.co/{hf_repo}")

            shutil.rmtree(local_path)
            results.append((hf_repo, "OK"))
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()
            results.append((hf_repo, f"FAILED: {str(e)[:80]}"))

    print(f"\n{'='*60}\nSWEEP SUMMARY\n{'='*60}")
    for repo, status in results:
        print(f"  {status[:6]:6}  {repo}")
    ok = sum(1 for _, s in results if s == "OK")
    print(f"\n{ok}/{total} merges uploaded successfully.")


if __name__ == "__main__":
    main()
