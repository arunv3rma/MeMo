# Evaluation Pipeline

This directory contains three evaluation setups for benchmarking MeMo across three datasets: **NarrativeQA (NQA)**, **MuSiQue (MSQ)**, and **BrowseComp-Plus (BCP)**. Each setup differs in how the large model (LM) interacts with the small memory model (SM) across turns.

All setups require two vLLM servers to be running: an LM server and an SM server. Evaluation is scored using [DeepEval](https://github.com/confident-ai/deepeval) with an external judge model configured via environment variables.

---

## Prerequisites

Both vLLM servers must be running before launching any eval script. The scripts expect:

- **LM server** on port `4325` (configurable via `--lm_port`)
- **SM server** on port `4324` (configurable via `--sm_port`)

If an LFM Memory Model is being used, the `lfm` python env needs to installed first as the lfm architecture uses a different version of vllm and transformers compared to Qwen and Gemma models.

Set the following environment variables (e.g. in a `.env` file) for the DeepEval judge:

```
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=<your_key>
OPENROUTER_MODEL_NAME=<judge_model_name>
```

---

## 1. Single-Turn (`single_turn_baseline/`)

The LM receives the question, decides whether memory retrieval is needed, generates sub-questions, queries the SM once, and synthesises a final answer — all in a single round of interaction.

**Flow:**
1. LM generates sub-questions (or answers directly if no memory needed)
2. SM answers sub-questions in parallel
3. LM synthesises a final answer from SM responses

**Scripts:**

| Dataset | Shell script | Python script |
|---------|-------------|---------------|
| NarrativeQA | `eval_nqa.sh` | `eval_nqa_trng.py` |
| MuSiQue | `eval_msq.sh` | `eval_msq_trng.py` |
| BrowseComp-Plus | `eval_bcp.sh` | `eval_bcp_trng.py` |

**Key parameters:**

```bash
--lm_port        # LM vLLM server port (default: 4325)
--sm_port        # SM vLLM server port (default: 4324)
--nqa_qns_path   # Path to NQA questions JSONL (or --msq_qns_path / --bcp_qns_path)
--max_num_qns/max_num_docs   # Limit number of questions evaluated
--output_path    # Path for JSON output
--max_concurrent # Max concurrent requests (default: 50)
```

**Example (NQA):**
```bash
cd single_turn_baseline
bash eval_nqa.sh
```

Configure `NQA_QNS_PATH`, `EVAL_DIR`, and `NUM_RUNS` at the top of the shell script before running.

---

## 2. Unstructured Multi-Turn (`unstructured_multi_turn_baseline/`)

An extension of single-turn where the LM iteratively reviews the full Q&A history and decides whether to ask more sub-questions. State tracking is implicit — the LM reasons over the raw conversation history with no structured guidance.

**Flow:**
1. Round 0: identical to single-turn (LM generates sub-questions, SM answers)
2. Loop (up to `--max_turns`): LM reviews full history, summarises what is known, identifies gaps, and generates the next round of sub-questions
3. Loop exits early when the LM decides it has enough information
4. Final: LM synthesises an answer from all collected SM responses

**Scripts:**

| Dataset | Shell script | Python script |
|---------|-------------|---------------|
| NarrativeQA | `eval_nqa_trng_naive_multi_step.sh` | `eval_nqa_trng_naive_multi_step.py` |
| MuSiQue | `eval_msq_trng_naive_multi_step.sh` | `eval_msq_trng_naive_multi_step.py` |
| BrowseComp-Plus | `eval_bcp_trng_naive_multi_step.sh` | `eval_bcp_trng_naive_multi_step.py` |

**Key parameters (in addition to single-turn parameters):**

```bash
--max_turns          # Maximum number of retrieval loops (default: 5)
--lm_temperature     # LM temperature for round 0 (default: 1.1)
--sm_temperature     # SM temperature (default: 0.7)
--loop_temperature   # LM temperature during loop turns (default: 1.0)
--final_temperature  # LM temperature for final synthesis (default: 0.3)
```

**Example (NQA):**
```bash
cd unstructured_multi_turn_baseline
bash eval_nqa_trng_naive_multi_step.sh
```

Configure `NQA_QNS_PATH`, `EVAL_DIR`, `MAX_TURNS`, and temperature variables at the top of the shell script.

---

## 3. Structured Multi-Turn (`structured_multi_turn/`)

A multi-turn setup with explicit phase structure. The LM progresses through defined phases — grounding, entity pinning, and answer seeking — with dead-end detection to skip candidates the SM consistently cannot answer.

**Flow:**
1. **Grounding**: LM identifies candidate entities/facts relevant to the question
2. **Entity pinning** (up to `--max_entity_turns`): LM asks targeted sub-questions to pin down the correct entity; candidates with repeated "I don't know" responses are pruned after `--dead_end_threshold` consecutive failures
3. **Answer seeking** (up to `--max_answer_turns`): LM asks sub-questions to gather the specific answer using the pinned entity
4. **Final synthesis**: LM produces a final answer

The structured shell script also handles spinning up/down the SM vLLM server between model checkpoints, making it suitable for evaluating multiple SM checkpoints in sequence.

**Scripts:**

| Dataset | Shell script | Python script |
|---------|-------------|---------------|
| NarrativeQA | `eval_nqa_trng_structured_multi_step.sh` | `eval_nqa_trng_structured_multi_step.py` |
| MuSiQue | `eval_msq_trng_structured_multi_step.sh` | `eval_msq_trng_structured_multi_step.py` |
| BrowseComp-Plus | `eval_bcp_trng_structured_multi_step.sh` | `eval_bcp_trng_structured_multi_step.py` |

**Key parameters (in addition to single-turn parameters):**

```bash
--max_entity_turns           # Max turns for entity pinning phase (default: 7)
--max_answer_turns           # Max turns for answer seeking phase (default: 8)
--dead_end_threshold         # Consecutive IDK responses before pruning a candidate (default: 3)
--lm_grounding_temperature   # LM temperature during grounding (default: 0.4)
--sm_grounding_temperature   # SM temperature during grounding (default: 0.1)
--lm_entity_temperature      # LM temperature during entity pinning (default: 0.4)
--sm_entity_temperature      # SM temperature during entity pinning (default: 0.1)
--lm_answer_temperature      # LM temperature during answer seeking (default: 1.0)
--sm_answer_temperature      # SM temperature during answer seeking (default: 0.3)
--lm_final_temperature       # LM temperature for final synthesis (default: 0.3)
--lm_model_name              # Optional: use an OpenRouter model instead of local vLLM
```

**Example (NQA):**
```bash
cd structured_multi_turn
bash eval_nqa_trng_structured_multi_step.sh
```

Configure `SM_MODELS` and `SM_EXPR_PREFIXES` arrays, `NQA_QNS_PATH`, `EVAL_DIR`, and temperature variables at the top of the shell script. The script loops over each SM model in the arrays, starts and stops the SM server between runs, and supports both local vLLM and OpenRouter LMs.

---

## Output

Each eval script writes two files per run:

- `<output_path>.json` — per-question results including sub-questions, SM responses, and the model's final answer
- `<output_path>_deepeval_summary.json` — aggregated accuracy and per-question judge scores

---

## Shared Utilities

| File | Purpose |
|------|---------|
| `deepeval_utils.py` | DeepEval integration and judge-model wrapper |
| `general_eval_prompt_utils.py` | Prompt formatting helpers shared across all setups |
