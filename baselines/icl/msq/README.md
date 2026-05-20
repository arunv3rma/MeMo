## Overview of scripts

Data preparation, source files, and the chunking step are documented in `baselines/README.md` (the "Data preparation → MuSiQue" section). This README only covers the evaluation entrypoints.

**`main_for_musique.py`** / **`main_for_musique.sh`**
Evaluates a model on MuSiQue **with gold supporting paragraphs** supplied as context (open-book / retrieval-oracle setting). For each question, the gold passages are injected into the prompt and the model is asked to answer. A separate LLM judge (`google/gemini-2.5-flash-lite` via OpenRouter) scores whether the answer is correct.

Supports any OpenAI-compatible API (OpenRouter for hosted models, or a local vLLM server via `--base_url`). Results are broken down by hop count (2-hop / 3-hop / 4-hop) and token usage is tracked per question.

Outputs per-model: `musique_results.json`, `musique_metrics.json`, `musique_results_token_summary.json`.

```
python main_for_musique.py \
    --model "google/gemini-2.5.-flash-lite" \
    --max_concurrent 8 \
    --max_questions 20   # optional cap for testing
```

**`main_for_musique_no_context.py`** / **`main_for_musique_no_context.sh`**
Variant of `main_for_musique.py` that sends **only the raw question** to the model with no passage context (closed-book / parametric-knowledge baseline). The `--include_para_titles` flag adds supporting paragraph titles as lightweight hints without the full text. Same LLM judge and output format as the with-context version.

Outputs per-model: `musique_results_no_context.json`, `musique_metrics_no_context.json`.

```
python main_for_musique_no_context.py \
    --model "google/gemini-2.5.-flash-lite" \
    --max_concurrent 30

# With paragraph title hints only:
python main_for_musique_no_context.py \
    --model "google/gemini-2.5.-flash-lite" \
    --include_para_titles
```

The `.sh` scripts loop over a list of OpenRouter models and then run the same evaluation against a local vLLM server at `http://localhost:4325/v1`.

---

## Remarks
- Set `OPENROUTER_API_KEY` in your environment or a `.env` file before running any evaluation script.
