## Overview of scripts

### Data requirement

Skip this section if pulling data directly from the [MeMo Huggingface Repo]((https://huggingface.co/datasets/Glow-AI/MeMo)).

The evaluation scripts require the NarrativeQA dataset cloned locally. See `data_processing_utils/README.md` for setup instructions (clone, pip install, `download_stories.py`).

The scripts default to `~/narrativeqa-master` and can be overridden with `--data_dir`.

Questions and answers are read from `qaps.csv`; full story text is read from `tmp/{document_id}.content` (full-story script only).

Evaluations are restricted to the 10-document subset (`DOC_ID_SUBSET`) baked into each script — the same 10 docs used by the NarrativeQA benchmark (`corpus_documents/narrativeqa/`).

---

### Data preparation

**Expected question count: 293.** The original NarrativeQA repo yields 294 questions across the same 10 document IDs, but this repo removes one exact duplicate (`"Where does Hi get a job at?"`) for a total of 293. There are additional near-duplicates if question strings are cleaned, but only the single exact duplicate is removed here to keep consistent with the HippoRAG2 paper.

---

### Evaluation

**`main_for_narrativeqa.py`** / **`main_for_narrativeqa.sh`**
Evaluates a model on NarrativeQA **with the full story supplied as context** (open-book setting). For each question, the complete story text is injected into the prompt and the model is asked to answer. A separate LLM judge (`google/gemini-2.5-flash-lite` via OpenRouter) scores whether the answer is correct against both reference answers using `run_evaluation_multi_answer` from `evaluation_pipeline/deepeval_utils.py`.

Supports any OpenAI-compatible API (OpenRouter for hosted models, or a local vLLM server via `--base_url`).

Outputs per model into a `{model_name}/` directory: `narrativeqa_{split}_results.json`, `narrativeqa_{split}_metrics.json`.

```
# OpenRouter
python main_for_narrativeqa.py \
    --model "google/gemini-2.5-flash-lite" \
    --split valid \
    --max_concurrent 8 \
    --max_docs 10

# vLLM
python main_for_narrativeqa.py \
    --base_url http://localhost:4325/v1 \
    --model auto \
    --split valid \
    --max_concurrent 10
```

**`main_for_narrativeqa_no_context.py`** / **`main_for_narrativeqa_no_context.sh`**
Variant of `main_for_narrativeqa.py` that sends **only the raw question** to the model with no story context (closed-book / parametric-knowledge baseline). Same LLM judge and output format.

Outputs per model: `narrativeqa_{split}_results_no_context.json`, `narrativeqa_{split}_metrics_no_context.json`.

```
# OpenRouter
python main_for_narrativeqa_no_context.py \
    --model "google/gemini-2.5-flash-lite" \
    --split valid \
    --max_concurrent 30 \
    --max_docs 10

# vLLM
python main_for_narrativeqa_no_context.py \
    --base_url http://localhost:4323/v1 \
    --model auto \
    --split valid \
    --max_concurrent 100
```

The `.sh` scripts loop over a list of OpenRouter models and include a commented-out vLLM block at the bottom.

---

## Remarks

- Set `OPENROUTER_API_KEY` in your environment or a `.env` file before running any evaluation script. This is required even for vLLM answer generation, since the judge model always runs via OpenRouter.
- `--max_docs` filters questions to the first N doc IDs in `DOC_ID_SUBSET` (max 10). `--max_questions` then caps within that filtered set.
- `--model auto` auto-detects the loaded model from the vLLM server's `/v1/models` endpoint.
- Full story context runs require significantly lower `--max_concurrent` than no-context runs due to the large prompt size.
