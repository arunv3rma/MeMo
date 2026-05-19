import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'final_generated_data')

with open(os.path.join(data_dir, 'bcp_300_queries_id.json'), 'r', encoding='utf-8') as j:
    SUBSET_300_QUERY_IDS = json.load(j)

