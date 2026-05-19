# data_processing_utils

Utilities for downloading and preprocessing benchmark datasets used in MeMo experiments.

## Scripts

### `download_browsecomplus_corpus.py`

Downloads the `Tevatron/browsecomp-plus-corpus` retrieval corpus from Hugging Face and writes each split as a JSONL file under `output/full_corpus_<split>.jsonl`.

**Important — do not publish BrowseComp-Plus data as plain text.** The BrowseComp-Plus benchmark is distributed in encrypted form to prevent benchmark contamination. Decrypted question text (see below) must never be committed to a public repository, posted online, or otherwise made publicly available in plaintext.

### `download_browsecomplus_questions.py`

Downloads the `Tevatron/browsecomp-plus` question set (encrypted on HuggingFace), decrypts it using the embedded canary key, and writes the result to `browsecomp_plus_questions.jsonl`.

**Important — same restriction as above.** The decrypted output file must be kept local and must not be shared publicly in any plaintext form.

### `convert_narrativeqa_to_chunks_jsonl.py`

Script used to convert [NarrativeQA](https://github.com/deepmind/narrativeqa) dataset into chunked JSONL files for the data synthesis pipeline. Chunked files available on Huggingface.

### `convert_musique_to_chunks_jsonl.py`

Script used to convert [MuSiQue](https://github.com/stonybrooknlp/musique) dataset into chunked JSONL files for the data synthesis pipeline. Chunked files available on Huggingface.

*Note that MuSiQue does not require chunking as each document is rather short (less than ~8k tokens).

