import os
import json

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_generated_data')

with open(os.path.join(data_dir, 'bcp_negative_N_doc_ids_per_query.json'), 'r', encoding='utf-8') as f:
    NEGATIVE_DOC_IDS_PER_QUERY = json.load(f)

NEGATIVE_DOCS_MAP = {
    1: NEGATIVE_DOC_IDS_PER_QUERY,
}
