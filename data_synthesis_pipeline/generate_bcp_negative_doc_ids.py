import json
import random
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bcp_data_utils import BROWSECOMP_QUERY_IDS_TO_SKIP

def generate_n(qns_path, output_path, seed):
    rng = random.Random(seed)
    negative_doc_ids_per_query = {}
    total_docs = 0
    shortfall_counts = []

    with open(qns_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            query_id = data['query_id']

            if query_id in BROWSECOMP_QUERY_IDS_TO_SKIP:
                continue

            evidence_docs = data.get('evidence_docs', [])
            negative_docs = data.get('negative_docs', [])

            n = len(evidence_docs)
            sampled = rng.sample(negative_docs, min(n, len(negative_docs)))
            negative_doc_ids_per_query[query_id] = [doc['docid'] for doc in sampled if doc.get('docid')]
            total_docs += len(negative_doc_ids_per_query[query_id])

            if len(negative_docs) < n:
                shortfall_counts.append((query_id, n, len(negative_docs), n - len(negative_docs)))

    # with open(output_path, 'w', encoding='utf-8') as f:
    #     json.dump(negative_doc_ids_per_query, f)

    total_queries = len(negative_doc_ids_per_query)
    print(f"Written {total_queries} query entries (N) to {output_path}")
    print(f"Total negative docs across all queries: {total_docs}")
    print(f"Mean negative docs per query: {total_docs / total_queries:.1f}" if total_queries else "")

    print(f"\nN negative doc availability check (N = number of evidence docs):")
    has_enough = total_queries - len(shortfall_counts)
    print(f"  Queries with enough for N: {has_enough}/{total_queries} ({100*has_enough/total_queries:.1f}%)" if total_queries else "")
    if shortfall_counts:
        print(f"  Queries short of N: {len(shortfall_counts)}")
        for qid, n, have, short in shortfall_counts[:10]:
            print(f"    {qid}: N={n}, have={have}, short by {short}")
        if len(shortfall_counts) > 10:
            print(f"    ... and {len(shortfall_counts) - 10} more")


def generate_2n(qns_path, n_path, output_path_2n, seed):
    rng = random.Random(seed)

    with open(n_path, 'r', encoding='utf-8') as f:
        n_data = json.load(f)

    negative_doc_ids_per_query_2n = {}
    total = 0
    has_enough_for_2n = 0
    shortfall_counts = []

    with open(qns_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            query_id = data['query_id']

            if query_id in BROWSECOMP_QUERY_IDS_TO_SKIP:
                continue
            if query_id not in n_data:
                continue

            evidence_docs = data.get('evidence_docs', [])
            negative_docs = data.get('negative_docs', [])

            n = len(evidence_docs)
            ids_n = n_data[query_id]
            ids_n_set = set(ids_n)
            remaining = [doc for doc in negative_docs if doc.get('docid') and doc['docid'] not in ids_n_set]
            sampled_extra = rng.sample(remaining, min(n, len(remaining)))
            ids_extra = [doc['docid'] for doc in sampled_extra]

            negative_doc_ids_per_query_2n[query_id] = ids_n + ids_extra

            total += 1
            if len(negative_docs) >= 2 * n:
                has_enough_for_2n += 1
            else:
                shortfall_counts.append((query_id, n, len(negative_docs), 2 * n - len(negative_docs)))

    # with open(output_path_2n, 'w', encoding='utf-8') as f:
    #     json.dump(negative_doc_ids_per_query_2n, f)

    print(f"Written {len(negative_doc_ids_per_query_2n)} query entries (2N) to {output_path_2n}")
    print(f"\n2N negative doc availability check (N = number of evidence docs):")
    print(f"  Queries with enough for 2N: {has_enough_for_2n}/{total} ({100*has_enough_for_2n/total:.1f}%)")
    if shortfall_counts:
        print(f"  Queries short of 2N: {len(shortfall_counts)}")
        for qid, n, have, short in shortfall_counts[:10]:
            print(f"    {qid}: N={n}, have={have}, short by {short}")
        if len(shortfall_counts) > 10:
            print(f"    ... and {len(shortfall_counts) - 10} more")


if __name__ == '__main__':
    _default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_generated_data')
    parser = argparse.ArgumentParser()
    parser.add_argument('--qns_path', default='/path/to/browsecomp_plus_questions.jsonl')
    parser.add_argument('--output_path', default=os.path.join(_default_dir, 'negative_N_doc_ids_per_query.json'))
    parser.add_argument('--output_path_2n', default=os.path.join(_default_dir, 'negative_2N_doc_ids_per_query.json'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_n', action='store_true', help='Skip N generation and use existing output_path file')
    args = parser.parse_args()

    # if not args.skip_n:
    #     generate_n(args.qns_path, args.output_path, args.seed)

    generate_n(args.qns_path, args.output_path, args.seed)
    # generate_2n(args.qns_path, args.output_path, args.output_path_2n, args.seed)
