# Baseline run scripts

The minimal set needed to reproduce the baseline numbers. All paths inside these scripts are resolved relative to the MeMo repo root — each script `cd`s there using `dirname "$0"`, so you can invoke them from anywhere as `bash baselines/scripts/<name>.sh`.

## Assumed directory layout

The scripts expect benchmark inputs at:

- `baselines/data/decrypted.jsonl` — BrowseComp-Plus questions
- `baselines/data/browsecomp_plus/full_corpus_train.jsonl` — BCP corpus
- `baselines/data/musique_corpus_chunks_1000.jsonl`, `baselines/data/musique_questions_chunks_1000.jsonl` — MuSiQue
- `baselines/data/narrativeqa_dev_10_doc_corpus.json`, `baselines/data/nqa_question.json` — NarrativeQA

Outputs land in `baselines/output_runs/` and logs in `baselines/output_runs/logs/` — both auto-created.

## Python environment

Scripts call plain `python` and assume your env (the one with the MeMo requirements installed) is already activated. Activate it before running, e.g.:

```bash
conda activate <your-env>
bash baselines/scripts/run_bm25_bcp_qwen.sh
```

You will also need to tweak the `API_BASE` / `MODEL_ID` / `API_KEY` block near the top of each script to match the vLLM (or other OpenAI-compatible) server you have running.

## What's here

**Per-method, per-benchmark independent runs (Qwen2.5-32B-Instruct serving):**
- BM25: `run_bm25_{bcp,musique,nqa}_qwen.sh`
- HippoRAG2: `run_hipporag2_{bcp,musique,nqa}_qwen.sh`
- NV-Embed: `run_nv_embed_{bcp,musique,nqa}_qwen.sh`
- Cartridges: `run_cartridges_{bcp,musique,nqa}_qwen.sh`

**Cartridges-specific:**
- `index_cartridges.sh` — inference loop after KV-cache distillation

**Evaluation:**
- `run_deepeval.sh` — calls `baselines/utils/deepeval_via_algo_utils.py` against generated answer JSONs

**Helpers:**
- `aggregate_runs.py` — combines per-run `*_run{1,2,3,...}_*.json` outputs into a single `combined_<method>_<bench>_<model>_<timestamp>.json`. Called automatically at the end of each run script.
