import os
import json

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_generated_data')

with open(os.path.join(data_dir, 'musique_negative_N_doc_ids_per_query.json'), 'r', encoding='utf-8') as f:
    MUSIQUE_NEGATIVE_DOC_IDS_PER_QUERY = json.load(f)

with open(os.path.join(data_dir, 'musique_negative_2N_doc_ids_per_query.json'), 'r', encoding='utf-8') as f:
    MUSIQUE_NEGATIVE_2N_DOC_IDS_PER_QUERY = json.load(f)

MUSIQUE_NEGATIVE_DOCS_MAP = {
    1: MUSIQUE_NEGATIVE_DOC_IDS_PER_QUERY,
    2: MUSIQUE_NEGATIVE_2N_DOC_IDS_PER_QUERY,
}
