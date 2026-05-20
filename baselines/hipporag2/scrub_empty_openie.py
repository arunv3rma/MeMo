"""Strip empty/failed entries out of HippoRAG's openie_results JSON so the next
resume re-attempts them with vLLM.

Usage:
    python -m baselines.hipporag2.scrub_empty_openie \
        --save_dir ./baselines/hipporag2/output_musique/musique_corpus_chunks_1000_qwen32idx_20260428_165305 \
        [--also_empty_entities] [--dry_run]

Backs up the original JSON to <name>.bak_<ts>.json before rewriting.
"""
import argparse
import glob
import json
import os
import shutil
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True, help="HippoRAG save_dir holding openie_results_ner_*.json")
    ap.add_argument("--also_empty_entities", action="store_true",
                    help="Also drop entries with empty extracted_entities (default: only empty triples).")
    ap.add_argument("--dry_run", action="store_true", help="Report what would be removed; don't write.")
    args = ap.parse_args()

    pattern = os.path.join(args.save_dir, "openie_results_ner_*.json")
    matches = glob.glob(pattern)
    if not matches:
        raise SystemExit(f"No openie_results_ner_*.json under {args.save_dir}")
    if len(matches) > 1:
        print(f"Multiple openie files found, processing all: {matches}")

    for path in matches:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        docs = data.get('docs', [])
        kept, dropped = [], []
        for d in docs:
            triples_empty = not d.get('extracted_triples')
            ents_empty = not d.get('extracted_entities')
            should_drop = triples_empty if not args.also_empty_entities else (triples_empty or ents_empty)
            (dropped if should_drop else kept).append(d)

        print(f"{path}: total={len(docs)}, drop={len(dropped)}, keep={len(kept)}")
        if dropped[:3]:
            print("  sample dropped idx:", [d.get('idx') for d in dropped[:3]])

        if args.dry_run:
            continue

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.replace('.json', f'.bak_{ts}.json')
        shutil.copy(path, backup)
        print(f"  backup -> {backup}")

        data['docs'] = kept
        # Recompute aggregate stats if present so they aren't stale.
        for stat_key in ('avg_ent_chars', 'avg_ent_words'):
            data.pop(stat_key, None)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  rewrote {path}")


if __name__ == "__main__":
    main()
