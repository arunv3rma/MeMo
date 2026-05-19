# Data Synthesis Pipeline

This directory contains the code for synthesising the MeMo training dataset. It supports two tasks:

1. **Main data synthesis** — end-to-end QA generation across three benchmark datasets (`datasynth_pipeline/`)
2. **Leave-One-Out (LOO) ablation** — re-run the same pipeline with one step removed at a time to measure each step's contribution (`loo_data_ablation/`). The importance of each step is highlighted in the **Appendix E** of the paper.

---

## Tasks

### 1. Main Data Synthesis (`datasynth_pipeline/`)

Three pipeline scripts, one per dataset:

| Script | Dataset |
|---|---|
| `bcp_datasynth_pipeline.sh` | BrowseComp-Plus (BCP) |
| `musique_datasynth_pipeline.sh` | MuSiQue |
| `narrativeqa_datasynth_pipeline.sh` | NarrativeQA |

Each script runs the same five-step pipeline using multiple locally-served LLM (Qwen2.5-32B-Instruct via vLLM, addressed over `--ports`):

| Step | Script | Description |
|---|---|---|
| 1a | `generate_directfact_qa.py` / `with_neg_generate_directfact_qa.py` | Extract direct, explicitly-stated QA pairs from each document chunk |
| 1b | `generate_indirect_fact_qa.py` / `with_neg_generate_indirect_fact_qa.py` | Extract indirect QA pairs requiring multi-sentence reasoning |
| — | `combine_direct_and_indirect.py` | Merge step 1a and 1b outputs (no LLM calls) |
| 2 | `generate_consolidation_cache.py` | Consolidate permutations of related QA pairs per document |
| 3 | `check_self_containment_post_combination.py` | Check and fix QA pairs that are not self-contained; retries failed pairs |
| 4 | `generate_surface_entity_cache.py` | Generate surface-level entity QA pairs from the verified set |
| 5 | `generate_crossdoc_entity_combination_cache.py` / `with_neg_generate_crossdoc_entity_combination_cache.py` | Generate cross-document entity QA pairs (converging clues and parallel properties) |

All steps support `--checkpoint_iter_freq` for fault-tolerant resumption and write intermediate `.json` caches to a configurable `OUTPUT_DIR`.

BCP and MuSiQue use `with_neg_*` variants of steps 1 and 5 that also sample hard negative documents for each query.

### 2 LOO Ablation (`loo_data_ablation/`)

Three ablation scripts, one per dataset:

| Script | Dataset |
|---|---|
| `bcp_LOO_data_ablation.sh` | BrowseComp-Plus (BCP) |
| `musique_LOO_data_ablation.sh` | MuSiQue |
| `narrativeqa_LOO_data_ablation.sh` | NarrativeQA |

Each script accepts a single argument `ABLATE_STEP` (values: `1a`, `1b`, `2`, `3`, `4`, `5`, or `none`) and runs the full pipeline with the specified step replaced by a `--passthrough` run — the step reads its normal input, writes the expected output format, and returns without making any LLM calls. This isolates the contribution of each step to downstream quality.

```bash
# Skip the consolidation step
./bcp_LOO_data_ablation.sh 2

# Skip the cross-doc combination step
./bcp_LOO_data_ablation.sh 5

```

Output lands in `ablation_with_neg_bcp<N>_skip_step<ABLATE_STEP>/`.

---

## Helper Files

### Dataset utilities

| File | Purpose |
|---|---|
| `bcp_data_utils.py` | BCP corpus loading, query ID filtering, skip-list |
| `musique_data_utils.py` | MuSiQue corpus and question loading |
| `nqa_data_utils.py` | NarrativeQA corpus loading and doc-subset maps |
| `bcp_query_negatives_utils.py` | Hard-negative doc IDs for BCP queries |
| `bcp_query_subset_utils.py` | BCP query subsetting helpers |
| `musique_query_negatives_utils.py` | Hard-negative doc IDs for MuSiQue queries |
| `nqa_subset_utils.py` | NarrativeQA doc-subset definitions |

### Negative document generation

| File | Purpose |
|---|---|
| `generate_bcp_negative_doc_ids.py` | Builds the hard-negative doc ID list for BCP |
| `generate_musique_negative_doc_ids.py` | Builds the hard-negative doc ID list for MuSiQue |

### Model and prompt utilities

| File | Purpose |
|---|---|
| `model_utils.py` | LLM inference wrappers (async, multi-port load balancing) |
| `general_prompt_utils.py` | Shared prompt templates for fact extraction and QA generation |
| `mam_general_utils.py` | Answer formatting helpers used across prompts |
| `third_party_api_client.py` | OpenAI / OpenRouter async clients initialised from `.env` |

### Pre-computed data

| Path | Contents |
|---|---|
| `data_subsets/` | Pre-computed query ID lists and negative doc ID mappings (`.json`) for BrowseComp-Plus and MuSiQue. NarrativeQA has only 10 and is left inline. |
| `langdetect_processing/` | Language-detection scripts and logs for corpus filtering of BrowseComp-Plus |
