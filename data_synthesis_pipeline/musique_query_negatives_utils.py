import os
import json

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_subsets')

with open(os.path.join(data_dir, 'musique_negative_N_doc_ids_per_query.json'), 'r', encoding='utf-8') as f:
    MUSIQUE_NEGATIVE_DOC_IDS_PER_QUERY = json.load(f)

MUSIQUE_NEGATIVE_DOCS_MAP = {
    1: MUSIQUE_NEGATIVE_DOC_IDS_PER_QUERY,
}

# The 2N file is optional (not shipped in data_subsets/); generate it with
# generate_musique_negative_doc_ids.py if you need neg_n=2.
_path_2n = os.path.join(data_dir, 'musique_negative_2N_doc_ids_per_query.json')
if os.path.exists(_path_2n):
    with open(_path_2n, 'r', encoding='utf-8') as f:
        MUSIQUE_NEGATIVE_2N_DOC_IDS_PER_QUERY = json.load(f)
    MUSIQUE_NEGATIVE_DOCS_MAP[2] = MUSIQUE_NEGATIVE_2N_DOC_IDS_PER_QUERY
