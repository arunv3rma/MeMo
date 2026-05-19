"""
convert_musique_to_chunks_jsonl.py

Splits each paragraph into overlapping chunks, each becoming its own corpus entry. 
Questions are mapped to all chunks of their supporting / negative paragraphs.

musique_corpus_chunks_1000.jsonl    — one line per chunk:
    {"docid": "2hop__460946_294723_para0_chunk0", "text": "...", "url": "Miquette Giraudy"}

musique_questions_chunks_1000.jsonl — one line per question:
    {"query_id": "2hop__460946_294723", "question": "...", "answers": [...],
     "document_id": "2hop__460946_294723", "hop": "2hop",
     "evidence_docs": [{"docid": "..."}], "gold_docs": [{"docid": "..."}],
     "negative_docs": [{"docid": "..."}]}

Usage:
    python convert_musique_to_chunks_jsonl.py \\
        [--output_dir .] \\
        [--chunk_size 400] \\
        [--overlap 40] \\
        [--input_json PATH]
"""

import argparse
import json
import os
import re

DEFAULT_INPUT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "./hipporag2_dataset/musique.json",
)


def get_hop_count(qid: str) -> str:
    m = re.match(r"(\d+hop)", qid)
    return m.group(1) if m else "unknown"


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    corpus_path = os.path.join(args.output_dir, "musique_corpus_chunks_1000.jsonl")
    questions_path = os.path.join(args.output_dir, "musique_questions_chunks_1000.jsonl")

    print(f"Loading MuSiQue examples from {args.input_json} ...")
    with open(args.input_json, encoding="utf-8") as f:
        examples = json.load(f)
    print(f"  Total examples: {len(examples)}")
    print(f"\nChunking (chunk_size={args.chunk_size} words, overlap={args.overlap} words)...")

    total_chunks = 0
    total_supporting_chunks = 0
    total_negative_chunks = 0

    with open(corpus_path, "w", encoding="utf-8") as corpus_f, \
         open(questions_path, "w", encoding="utf-8") as questions_f:

        for entry in examples:
            qid = entry["id"]
            hop = get_hop_count(qid)
            paragraphs = entry.get("paragraphs") or []

            # para_id -> list of chunk docids (for question mapping)
            para_chunk_map: dict[str, list[str]] = {}

            # --- corpus: one line per chunk ---
            for para in paragraphs:
                para_base = f"{qid}_para{para['idx']}"
                chunks = chunk_text(para["paragraph_text"], args.chunk_size, args.overlap)

                chunk_docids = []
                for i, chunk in enumerate(chunks):
                    chunk_docid = f"{para_base}_chunk{i}"
                    chunk_docids.append(chunk_docid)
                    corpus_f.write(json.dumps({
                        "docid": chunk_docid,
                        "text": chunk,
                        "url": para["title"],
                    }, ensure_ascii=False) + "\n")

                para_chunk_map[para_base] = chunk_docids
                total_chunks += len(chunks)

                if para.get("is_supporting"):
                    total_supporting_chunks += len(chunks)
                else:
                    total_negative_chunks += len(chunks)

            # --- questions ---
            evidence_docs = []
            negative_docs = []

            for para in paragraphs:
                para_base = f"{qid}_para{para['idx']}"
                chunk_refs = [{"docid": cid} for cid in para_chunk_map[para_base]]
                if para.get("is_supporting"):
                    evidence_docs.extend(chunk_refs)
                else:
                    negative_docs.extend(chunk_refs)

            answers = [entry["answer"]] + (entry.get("answer_aliases") or [])

            questions_f.write(json.dumps({
                "query_id": qid,
                "question": entry["question"],
                "answers": answers,
                "document_id": qid,
                "hop": hop,
                "evidence_docs": evidence_docs,
                "gold_docs": evidence_docs,
                "negative_docs": negative_docs,
            }, ensure_ascii=False) + "\n")

    print(f"\nCorpus")
    print(f"  Total chunks           : {total_chunks}")
    print(f"  Supporting (gold) chunks: {total_supporting_chunks}")
    print(f"  Negative chunks        : {total_negative_chunks}")
    print(f"  Avg chunks/para        : {total_chunks / max(len(examples) * 20, 1):.1f}")
    print(f"  Written → {corpus_path}")
    print(f"\nQuestions")
    print(f"  Written → {questions_path}")
    print(f"\nTo use with generate_lvl1to3_refltrace_cache_v2.py:")
    print(f"  --corpus_path {corpus_path}")
    print(f"  --qns_path    {questions_path}")
    print("\nDone.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert MuSiQue JSON to chunked JSONL")
    p.add_argument(
        "--output_dir", type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write musique_corpus_chunks_1000.jsonl and musique_questions_chunks_1000.jsonl",
    )
    p.add_argument(
        "--input_json", type=str,
        default=DEFAULT_INPUT_JSON,
        help="Path to the pre-selected MuSiQue JSON file (default: hipporag2_dataset/musique.json)",
    )
    p.add_argument("--chunk_size", type=int, default=6400,
                   help="Chunk size in words (default: 6400 ≈ ~8k tokens)")
    p.add_argument("--overlap", type=int, default=640,
                   help="Overlap between consecutive chunks in words (default: 640)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
