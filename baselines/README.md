# MeMo Baselines

This directory contains baseline methods used in the MeMo paper. Five methods are implemented:

| Method | Directory | Description |
|---|---|---|
| BM25 | `bm25/` | Lexical retrieval |
| NV-Embed-V2 | `nv_embed/` | Dense embedding retrieval |
| HippoRAG2 | `hipporag2/` | Graph-augmented RAG |
| In-Context Learning | `icl/` | Full-context QA with no retrieval |
| Cartridges | `cartridges/` | KV-cache distillation |

Three benchmarks are supported across the methods:

- **BCP** — BrowseComp-Plus
- **MSQ** — MuSiQue
- **NQA** — NarrativeQA

## Layout

```
baselines/
├── bm25/, nv_embed/, hipporag2/    # one main_for_<bench>.py per method/bench
├── icl/{bcp,msq,nqa}/              # ICL runners
├── cartridges/                     # upstream cartridges source + main.py entrypoint
├── utils/                          # shared helpers (generate, read, parse, llm_server,
│                                   #   third_party_api_client, deepeval_via_algo_utils)
├── scripts/                        # shell runners, seed aggregation, deepeval orchestration
└── requirements.txt                # baseline-only extras (bm25s, hipporag, pydrantic)
```

## Prerequisites

1. **Python env** — from the repo root, install:
   ```bash
   pip install -r memo_requirements.txt
   pip install -r baselines/requirements.txt   # adds bm25s, hipporag, pydrantic
   ```
   Cartridges additionally needs the upstream `cartridges` and `tokasaurus` packages — install those separately following `baselines/cartridges/README.md`.
2. **vLLM server** — most baselines call a Qwen2.5-32B-Instruct vLLM server. From the repo root, start it with `bash vllm_serve_qwen2_5_32b_instruct.sh`. Note its base URL/port; the per-method `main_for_*.py` scripts and the `run_*.sh` scripts expect it.
3. **Repo layout** — the per-method `main_for_*.py` files load `bcp_data_utils` / `musique_data_utils` from the sibling `data_synthesis_pipeline/`, and the ICL runners load `deepeval_utils` from the sibling `evaluation_pipeline/`. Both are reached via relative `sys.path.insert` from the file location, so the `baselines/` directory only works as a subdirectory of the MeMo repo, not standalone.
4. **Run from the repo root** — most scripts assume `cwd == MeMo/` and call modules as `baselines.<method>.main_for_<bench>`.

## Data preparation

All benchmark inputs land under `baselines/data/`. The scripts and ICL runners read from there directly.

### BrowseComp-Plus (BCP)

BCP question/corpus text is distributed in **encrypted form** on Hugging Face (via Tevatron) to prevent benchmark contamination; the auxiliary subset / negative-doc files live in the [MeMo HF repo](https://huggingface.co/datasets/Glow-AI/MeMo) under `corpus_documents/browsecomp_plus/`. The download/decrypt scripts in `data_processing_utils/` are adapted from [Tevatron/browsecomp-plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus) and [Tevatron/browsecomp-plus-corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus).

> **Important:** Decrypted BCP data must never be committed to a public repository or shared online in plaintext.

```bash
# 1. Retrieval corpus → output/full_corpus_<split>.jsonl  (from Tevatron)
python data_processing_utils/download_browsecomplus_corpus.py

# 2. Questions (downloads + decrypts) → browsecomp_plus_questions.jsonl  (from Tevatron)
python data_processing_utils/download_browsecomplus_questions.py
```

From the MeMo HF repo (`corpus_documents/browsecomp_plus/`):
- `bcp_subset300_queries_id.json` — 300-question subset IDs used by the ICL runners (default `--query_ids_file`)
- `bcp_negative_N_doc_ids_per_query.json` — negative doc IDs per query (consumed by `--include_negatives`)

Move all outputs into `baselines/data/` so the runners find them:
- `baselines/data/browsecomp_plus_questions.jsonl` — used by ICL (`--questions_file`) and the seed-sweep scripts (`QUESTIONS`)
- `baselines/data/browsecomp_plus/full_corpus_train.jsonl` — used by the seed-sweep scripts (`CORPUS`)
- `baselines/data/browsecomp_plus/bcp_subset300_queries_id.json`
- `baselines/data/browsecomp_plus/bcp_negative_N_doc_ids_per_query.json`

Each question is a JSON object with `query_id`, `query`, `answer`, `evidence_docs`, `gold_docs`. **Expected question count: 300**

### MuSiQue (MSQ)

Pull directly from the [MeMo dataset on Hugging Face](https://huggingface.co/datasets/Glow-AI/MeMo), under `corpus_documents/musique/`:

| HF file | Local path | Purpose |
|---|---|---|
| `corpus_documents/musique/musique_corpus_chunks_1000.jsonl` | `baselines/data/musique_corpus_chunks_1000.jsonl` | Chunked corpus (seed-sweep `CORPUS`) |
| `corpus_documents/musique/musique_questions_chunks_1000.jsonl` | `baselines/data/musique_questions_chunks_1000.jsonl` | Chunked questions (seed-sweep `QUESTIONS`) |
| `corpus_documents/musique/musique_negative_N_doc_ids_per_query.json` | `baselines/data/musique_negative_N_doc_ids_per_query.json` | Negative doc IDs per query |

Chunking is technically optional for MSQ — all documents are under ~8.5k tokens so each paragraph produces exactly one chunk at default settings — but we use the chunked format for consistency with other baselines. **Expected question count: 1000**

### NarrativeQA (NQA)

Pull directly from the [MeMo HF repo](https://huggingface.co/datasets/Glow-AI/MeMo), under `corpus_documents/narrativeqa/`:

| HF file | Local path | Purpose |
|---|---|---|
| `corpus_documents/narrativeqa/narrativeqa_valid_corpus_chunks.jsonl` | `baselines/data/narrativeqa_valid_corpus_chunks.jsonl` | Corpus (seed-sweep `CORPUS`) |
| `corpus_documents/narrativeqa/narrativeqa_valid_questions_chunks.jsonl` | `baselines/data/narrativeqa_valid_questions_chunks.jsonl` | Questions (seed-sweep `QUESTIONS`) |

Evaluation is restricted to the 10-document `DOC_ID_SUBSET` baked into each script — the same 10 docs used by the HippoRAG2 paper. **Expected question count: 293**: the original NarrativeQA repo yields 294 questions across these IDs, but we drop one exact duplicate (`"Where does Hi get a job at?"`) to match the HippoRAG2 paper.

## Quick start: per-method seed sweeps

1. Start the Qwen vLLM server (step 2 above).
2. Place benchmark inputs under `baselines/data/` (see `scripts/README.md` for expected filenames) and confirm the `API_BASE` / `MODEL_ID` / `API_KEY` block at the top of each script matches your server.
3. From the repo root, run any combination of:
   ```bash
   bash baselines/scripts/run_bm25_{bcp,musique,nqa}_qwen_seeds.sh
   bash baselines/scripts/run_hipporag2_{bcp,musique,nqa}_qwen_seeds.sh
   bash baselines/scripts/run_nv_embed_{bcp,musique,nqa}_qwen_seeds.sh
   bash baselines/scripts/run_cartridges_{bcp,musique,nqa}_qwen_seeds.sh
   ```
   Each sweeps 3 independent runs for that method/benchmark and calls `aggregate_seeds.py` at the end.

## Running ICL

The `icl/` directory has its own `main_for_<bench>.py` and accompanying `.sh` per benchmark (`bcp/`, `msq/`, `nqa/`), plus `_no_context` variants for the no-retrieval ablations. See the README in each subdir.

## Running Cartridges

Cartridges is more involved because it requires a tokasaurus inference server and a per-document KV-cache distillation step.

1. **Configure `cartridges/.env`** — set `CARTRIDGES_OUTPUT_DIR` (where trained caches land) and any other env vars the upstream cartridges code reads.
2. **Start tokasaurus** serving Qwen2.5-32B-Instruct from `cartridges/tokasaurus/` (note: the `tokasaurus/` dependency is not bundled in this directory — install it separately following the upstream cartridges README):
   ```bash
   CUDA_VISIBLE_DEVICES=6,7 tksrs model=Qwen/Qwen2.5-32B-Instruct \
     kv_cache_num_tokens='(512 * 1024)' max_topk_logprobs=20 tp_size=2
   ```
3. **Stage data** — put per-document files (JSON or TXT) under `cartridges/data/<dataset_name>/`. For BCP subsets, see the upstream cartridges helpers (`prepare_bcp_subsets.py`) in the original cartridges checkout.
4. **Synthesize training data** — set `folder_path` in `cartridges/synthesize.py` to your data dir, then from `cartridges/`:
   ```bash
   python synthesize.py
   ```
   This calls the tokasaurus server.
5. **Train the KV caches** — set `parent_dir` and `text_file` in `cartridges/train.py` to match `CARTRIDGES_OUTPUT_DIR`, then:
   ```bash
   python train.py
   ```
   Tokasaurus is not used during training.
6. **Run inference** — from the repo root:
   ```bash
   bash baselines/scripts/index_cartridges.sh
   ```
   Add `--need_to_move_cartridges` on the first run after training so the trained caches get sharded and copied into `cartridges/tokasaurus/cartridges/`.

The `cartridges/main.py` entrypoint runs the single-turn cartridge inference loop. Multi-turn grounding-and-followup mode was removed during migration to drop an internal-prompt-utils dependency.

## Evaluation

Correctness scoring uses the [DeepEval](https://github.com/confident-ai/deepeval) framework with OpenAI's **gpt-5-nano** as the LLM-as-a-judge model (accessed via OpenRouter). Concretely, `baselines/utils/deepeval_via_algo_utils.py` wraps `evaluation_pipeline/deepeval_utils.run_evaluation`, which is the MeMo paper's correctness metric built on top of DeepEval. See [`evaluation_pipeline/README.md`](../evaluation_pipeline/README.md) for the full description of the judge setup, prompt templates, and the required env vars (`OPENROUTER_BASE_URL`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL_NAME=gpt-5-nano`).

You don't have to invoke the evaluator directly as the seed-sweep scripts call it per seed, and `baselines/scripts/run_deepeval.sh` is the standalone orchestrator if you want to re-evaluate generated answer JSONs without re-running inference.

For per-seed result aggregation, `scripts/aggregate_seeds.py` combines individual seed JSONs into a `combined_<method>_<bench>_<model>_<timestamp>.json` summary. The seed-sweep scripts call this automatically at the end.
