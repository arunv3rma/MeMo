## Overview of scripts

### Data preparation

The BrowseComp-Plus dataset is distributed in **encrypted form** on Hugging Face to prevent benchmark contamination. Use the scripts in `data_processing_utils/` to download and decrypt it locally.

> **Note:** These scripts are adapted from the scripts provided in their respective Hugging Face repositories: [Tevatron/browsecomp-plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus) and [Tevatron/browsecomp-plus-corpus](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus).

> **Important:** Decrypted BrowseComp-Plus data must never be committed to a public repository or shared online in plaintext form.

**1. Download the retrieval corpus**

```bash
python data_processing_utils/download_browsecomplus_corpus.py
# Output: output/full_corpus_<split>.jsonl
```

**2. Download and decrypt the questions**

```bash
python data_processing_utils/download_browsecomplus_questions.py
# Output: browsecomp_plus_questions.jsonl
```

---

### Data requirement

The evaluation scripts require the BrowseComp+ questions JSONL file, passed via `--questions_file`. There is no bundled copy — set the path to your local file:

```
/path/to/browsecomp_plus_questions.jsonl
```

Each line is a JSON object with fields: `query_id`, `query`, `answer`, `evidence_docs`, `gold_docs`.

By default, evaluation is restricted to the 300-question subset defined in:
```
data_synthesis_pipeline/data_subsets/bcp_300_queries_id.json
```
This can be overridden with `--query_ids_file`.

---

### Evaluation

**`main_for_bcp.py`** / **`main_for_bcp.sh`**
Evaluates a model on BrowseComp+ **with all associated evidence documents** supplied as context (open-book / retrieval-oracle setting). For each question, the evidence documents are injected into the prompt and the model is asked to answer. A separate LLM judge (`google/gemini-2.5-flash-lite` via OpenRouter) scores whether the answer is correct using `run_evaluation` from `evaluation_pipeline/deepeval_utils.py`.

Supports any OpenAI-compatible API (OpenRouter for hosted models, or a local vLLM server via `--base_url`).

Outputs per model into a `{model_name}/` directory: `bcp_results.json`, `bcp_metrics.json`.

```
# OpenRouter
python main_for_bcp.py \
    --model "anthropic/claude-3.5-sonnet" \
    --questions_file /path/to/browsecomp_plus_questions.jsonl \
    --max_concurrent 8

# vLLM
python main_for_bcp.py \
    --base_url http://localhost:4325/v1 \
    --model auto \
    --questions_file /path/to/browsecomp_plus_questions.jsonl \
    --max_concurrent 10
```

**`main_for_bcp_no_context.py`** / **`main_for_bcp_no_context.sh`**
Variant of `main_for_bcp.py` that sends **only the raw question** to the model with no document context (closed-book / parametric-knowledge baseline). Same LLM judge and output format.

Outputs per model: `bcp_results_no_context.json`, `bcp_metrics_no_context.json`.

```
# OpenRouter
python main_for_bcp_no_context.py \
    --model "anthropic/claude-3.5-sonnet" \
    --questions_file /path/to/browsecomp_plus_questions.jsonl \
    --max_concurrent 30

# vLLM
python main_for_bcp_no_context.py \
    --base_url http://localhost:4325/v1 \
    --model auto \
    --questions_file /path/to/browsecomp_plus_questions.jsonl \
    --max_concurrent 32
```

The `.sh` scripts loop over a list of OpenRouter models and include an active vLLM block at the bottom. Update `QUESTIONS_FILE` at the top of each script before running.

---

## Remarks

- Set `OPENROUTER_API_KEY` in your environment or a `.env` file before running. Required even for vLLM answer generation, since the judge model always runs via OpenRouter.
- `--query_ids_file` defaults to the 300-question subset in `data_synthesis_pipeline/data_subsets/bcp_300_queries_id.json`. Pass a different file to evaluate on a custom subset.
- `--max_questions` caps within the filtered subset. A warning is printed if it exceeds the subset size.
- `--model auto` auto-detects the loaded model from the vLLM server's `/v1/models` endpoint.
- Evidence-doc runs require lower `--max_concurrent` than no-context runs due to larger prompt sizes.
