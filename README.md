# MeMo: Memory as a Model

[![arXiv](https://img.shields.io/badge/arXiv-2605.15156-b31b1b.svg)](https://arxiv.org/abs/2605.15156)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Collection-yellow)](https://huggingface.co/collections/Glow-AI/memo-memory-as-a-model)

This repository contains the official codebase for **MeMo**, a modular framework that encodes new knowledge into a dedicated **memory model (MEM)** while keeping the LLM parameters unchanged. MeMo addresses the core limitation of frozen LLMs by enabling plug-and-play integration of domain-specific or up-to-date knowledge — without retraining, retrieval pipelines, or access to the LLM's weights.

## Key Advantages

Compared to RAG and continual fine-tuning baselines, MeMo:

- **Captures cross-document relationships** — learns connections across the corpus, not just individual passages
- **Is robust to retrieval noise** — avoids brittleness from imperfect retrieval
- **Prevents catastrophic forgetting** — the LLM backbone is never modified
- **Works with closed-source LLMs** — requires no access to weights or output logits
- **Has inference-time cost independent of corpus size** — retrieval overhead does not scale with the knowledge base

## Overview

The core idea is a two-model inference protocol:

- **Executive Model (EM)** — a frozen, powerful model (e.g. Qwen2.5-32B-Instruct) that decomposes a complex question into sub-questions and synthesises a final answer.
- **Memory Model (MEM)** — a fine-tuned small model that serves as an external knowledge source, answering targeted sub-questions from what it memorised during SFT training.

The pipeline has four stages:

```
Raw Corpus
    │
    ▼
[1] Data Synthesis   — Generate synthetic QA pairs (5-step pipeline)
    │
    ▼
[2] SFT Training     — Fine-tune MEM on each corpus independently
    │
    ▼
[3] Model Merging    — Merge corpus-specific checkpoints into one MEM  (optional)
    │
    ▼
[4] Evaluation       — Benchmark EM+MEM against baselines
```

---

## Reproducing with Pre-generated Data

Pre-generated subset IDs and hard-negative document IDs used in the paper's experiments are in `data_synthesis_pipeline/data_subsets/`. These can be passed directly to the synthesis or evaluation scripts.

Other pre-generated data is available on HuggingFace from the original dataset providers.

---

## Repository Structure

```
MeMo/
├── baselines_icl/              # ICL oracle and closed-book baselines (bcp/, nqa/, msq/)
├── data_processing_utils/      # Download & preprocess raw datasets
├── data_synthesis_pipeline/    # 5-step synthetic QA generation
│   ├── data_subsets/           # Pre-generated subset IDs & negative doc IDs
│   ├── datasynth_pipeline/     # End-to-end pipeline shell scripts per dataset
│   └── loo_data_ablation/      # Leave-one-out data ablation scripts
├── sft_training/               # Supervised fine-tuning (full, LoRA, Gemma variants)
├── model_merging_scripts/      # Parameter-space model merging (optional)
├── evaluation_pipeline/        # MEMO evaluation (single-turn, unstructured, structured)
├── memo_requirements.txt
└── lfm_requirements.txt
```

---

## Benchmarks

| Dataset | Abbrev | Size | Task |
|---------|--------|------|------|
| BrowseComp-Plus | BCP | 300 questions | Long-context web document QA |
| NarrativeQA | NQA | 293 questions | Narrative document comprehension |
| MusiQue | MSQ | 1 000 questions | Multi-hop reasoning across passages |

---

## Setup

### 1. Install dependencies

```bash
conda create -n memo python=3.10.19 -y
conda activate memo
pip install -r memo_requirements.txt

# only required as a separate env if training LFM memory models
conda create -n lfm python=3.10.20 -y
conda activate lfm
pip install -r lfm_requirements.txt
```

### 2. Environment variables

Copy `.env_sample` to `.env` and fill in your keys:

```bash
cp .env_sample .env
```

```
OPENAI_API_KEY=...        # Used by DeepEval for LLM-based scoring
OPENROUTER_API_KEY=...    # Optional: for routing API calls
WANDB_API_KEY=...         # Optional: W&B experiment tracking
```

---

## Stage 1 — Data Preparation

Download and preprocess the raw corpora. See [`data_processing_utils/README.md`](data_processing_utils/README.md) for per-dataset instructions.

---

## Stage 2 — Data Synthesis Pipeline

Generates synthetic QA pairs from the raw corpus using a 5-step pipeline. Each dataset has a self-contained shell script in `data_synthesis_pipeline/datasynth_pipeline/` that runs all steps sequentially. The pipeline can be sped up with mulitple running vLLM inference servers; a sample launch script is provided at `vllm_serve_qwen2_5_32b_instruct.sh`.

Pre-generated subset IDs and hard-negative document IDs are in `data_synthesis_pipeline/data_subsets/`. Leave-one-out ablation scripts are in `data_synthesis_pipeline/loo_data_ablation/`.

See [`data_synthesis_pipeline/README.md`](data_synthesis_pipeline/README.md) for full details.

---

## Stage 3 — SFT Training

Fine-tunes a separate MEM checkpoint per dataset. Supports full SFT and LoRA variants across Qwen2.5, Gemma3, and LFM base models, using DeepSpeed ZeRO-2.

See [`sft_training/`](sft_training/) for launch scripts and hyperparameter defaults.

---

## Stage 4 — Model Merging (Optional)

Merges corpus-specific MEM checkpoints into a single generalised MEM using parameter-space merge methods (linear, SLERP, task vectors, TIES, DARE-linear, DARE-TIES). Can be skipped if evaluating dataset-specific checkpoints independently.

See [`model_merging_scripts/README.md`](model_merging_scripts/README.md) for usage and method details.

---

## Stage 5 — Evaluation

All MEMO evaluation scripts use two vLLM servers (one EM, one MEM). Four evaluation paradigms are available:

| Paradigm | Directory | Description |
|----------|-----------|-------------|
| Single-turn baseline | `single_turn_baseline/` | EM only, no MEM |
| Unstructured multi-turn | `unstructured_multi_turn_baseline/` | Naive Multi-turn loop |
| Structured multi-turn | `structured_multi_turn/` | Full MEMO protocol |
| ICL baselines | `baselines_icl/` | Oracle retrieval and closed-book baselines (no MEM) |

See [`evaluation_pipeline/README.md`](evaluation_pipeline/README.md) and each `baselines_icl/<dataset>/README.md` for script-level details.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{quek2026memomemorymodel,
      title={MeMo: Memory as a Model}, 
      author={Ryan Wei Heng Quek and Sanghyuk Lee and Alfred Wei Lun Leong and Arun Verma and Alok Prakash and Nancy F. Chen and Bryan Kian Hsiang Low and Daniela Rus and Armando Solar-Lezama},
      year={2026},
      eprint={2605.15156},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.15156}, 
}
```


