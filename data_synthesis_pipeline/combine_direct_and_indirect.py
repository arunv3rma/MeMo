import json


def extract_qa_list(raw) -> list[dict]:
    """
    Safely extract a flat list of {"question", "answer"} dicts from a qa_pairs field.
    Handles both the nested {"qa_pairs": [...]} format from older scripts and
    already-flat list format.
    """
    if isinstance(raw, dict):
        raw = raw.get("qa_pairs", [])
    if not isinstance(raw, list):
        return []
    return [qa for qa in raw if isinstance(qa, dict) and "question" in qa and "answer" in qa]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct_json_path", type=str, required=True)
    parser.add_argument("--indirect_json_path", type=str, required=True)
    # parser.add_argument("--consolidation_json_path", type=str, required=True)
    parser.add_argument("--output_json_path", type=str, required=True)
    parser.add_argument("--passthrough", action="store_true",
                        help="Write an empty combined cache without combining (for ablation studies)")

    args = parser.parse_args()

    # ---- Passthrough mode --------------------------------------------------
    if args.passthrough:
        print("[PASSTHROUGH] Skipping combination — writing empty combined cache...")
        output = {"total_entries": 0, "total_qa_pairs": 0, "qa_pairs_cache": []}
        with open(args.output_json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"[PASSTHROUGH] Wrote empty combined cache → {args.output_json_path}")
        return
    
    direct_json_path = args.direct_json_path #"/path/to/bcp_subset5_numepochs1_fact_extraction_cache_v3.json"
    with open(direct_json_path, 'r', encoding='utf-8') as f:
        direct_data = json.load(f)

    indirect_json_path = args.indirect_json_path #"/path/to/bcp_subset5_numepochs1_indirectfact_extraction_cache_v3.json"
    with open(indirect_json_path, 'r', encoding='utf-8') as g:
        indirect_data = json.load(g)

    # consolidation_json_path = args.consolidation_json_path #"/path/to/bcp_subset5_numepochs_consolidation_extraction_cache_v3.json"
    # with open(consolidation_json_path, 'r', encoding='utf-8') as h:
    #     consolidation_data = json.load(h)


    output_json_path = args.output_json_path #"/path/to/bcp_subset5_numepochs1_combined_extraction_cache_v3.json"

    # Accumulate all QA pairs per doc_id from all three sources into one list
    combined: dict[str, list[dict]] = {}

    for item in direct_data:
        if "error" in item:
            continue
        doc_id = item.get("doc_id")
        if not doc_id:
            continue
        qa_pairs = extract_qa_list(item.get("qa_pairs", []))
        if qa_pairs:
            combined.setdefault(doc_id, []).extend(qa_pairs)

    for item in indirect_data:
        if "error" in item:
            continue
        doc_id = item.get("doc_id")
        if not doc_id:
            continue
        qa_pairs = extract_qa_list(item.get("qa_pairs", []))
        if qa_pairs:
            combined.setdefault(doc_id, []).extend(qa_pairs)

    # Consolidation entries use "consolidated_qa_pairs" instead of "qa_pairs"
    # for item in consolidation_data:
    #     if "error" in item:
    #         continue
    #     doc_id = item.get("doc_id")
    #     if not doc_id:
    #         continue
    #     qa_pairs = extract_qa_list(item.get("consolidated_qa_pairs", []))
    #     if qa_pairs:
    #         combined.setdefault(doc_id, []).extend(qa_pairs)

    # Build output in the format expected by downstream scripts (qa_pairs_cache key)
    qa_pairs_cache = [
        {"doc_id": doc_id, "qa_pairs": qa_pairs}
        for doc_id, qa_pairs in combined.items()
    ]

    output = {
        "total_entries":  len(qa_pairs_cache),
        "total_qa_pairs": sum(len(e["qa_pairs"]) for e in qa_pairs_cache),
        "qa_pairs_cache": qa_pairs_cache,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(qa_pairs_cache)} docs ({output['total_qa_pairs']} total QA pairs) → {output_json_path}")

if __name__ == "__main__":
    main()
