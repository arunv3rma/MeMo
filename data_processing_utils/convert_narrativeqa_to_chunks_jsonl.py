"""
convert_narrativeqa_to_chunks_jsonl.py

Splits each story into overlapping chunks, each becoming its own corpus entry. 
Questions are mapped to all chunks of their source document.

corpus.jsonl    — one line per chunk:    {"docid", "text", "url"}
questions.jsonl — one line per QA pair:  {"query_id", "question", "answers",
                                           "document_id", "evidence_docs", "gold_docs"}

Usage:
    python convert_narrativeqa_to_chunks_jsonl.py \
        --narrativeqa_dir /path/to/narrativeqa-master \
        --output_dir /path/to/output \
        [--split train|test|valid|all] \
        [--chunk_size 4000] \
        [--overlap 400]
"""

import argparse
import csv
import json
import os


def extract_story_text(content: str, story_start: str, story_end: str) -> str:
    start_idx = content.find(story_start)
    if start_idx == -1:
        return content.strip()
    end_idx = content.find(story_end, start_idx + len(story_start))
    if end_idx == -1:
        return content[start_idx:].strip()
    return content[start_idx : end_idx + len(story_end)].strip()


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

    split_tag = args.split
    corpus_path = os.path.join(args.output_dir, f"narrativeqa_{split_tag}_corpus_chunks.jsonl")
    questions_path = os.path.join(args.output_dir, f"narrativeqa_{split_tag}_questions_chunks.jsonl")

    tmp_dir = os.path.join(args.narrativeqa_dir, "tmp")

    # ---- Load documents and write corpus chunks ------------------------------
    print(f"Loading documents and chunking (split={split_tag}, chunk_size={args.chunk_size}, overlap={args.overlap})...")

    # doc_id -> list of chunk docids (for question mapping)
    doc_chunk_map: dict[str, list[str]] = {}

    skipped_docs = 0
    total_chunks = 0

    with open(os.path.join(args.narrativeqa_dir, "documents.csv"), encoding="utf-8") as csv_f, \
         open(corpus_path, "w", encoding="utf-8") as out_f:

        for row in csv.DictReader(csv_f):
            doc_id = row["document_id"]

            if split_tag != "all" and row["set"] != split_tag:
                continue

            content_path = os.path.join(tmp_dir, f"{doc_id}.content")
            if not os.path.exists(content_path):
                print(f"  [skip] No content file for {doc_id}")
                skipped_docs += 1
                continue

            with open(content_path, encoding="utf-8", errors="replace") as cf:
                content = cf.read()

            if not content.strip():
                print(f"  [skip] Empty content for {doc_id}")
                skipped_docs += 1
                continue

            text = extract_story_text(content, row["story_start"], row["story_end"])
            chunks = chunk_text(text, args.chunk_size, args.overlap)

            chunk_docids = []
            for i, chunk in enumerate(chunks):
                chunk_docid = f"{doc_id}_chunk{i}"
                chunk_docids.append(chunk_docid)
                out_f.write(json.dumps({
                    "docid": chunk_docid,
                    "text": chunk,
                    "url": row["story_url"],
                }, ensure_ascii=False) + "\n")

            doc_chunk_map[doc_id] = chunk_docids
            total_chunks += len(chunks)

    print(f"  Documents loaded : {len(doc_chunk_map)}")
    print(f"  Documents skipped: {skipped_docs}")
    print(f"  Total chunks     : {total_chunks}")
    print(f"  Avg chunks/doc   : {total_chunks / max(len(doc_chunk_map), 1):.1f}")
    print(f"Corpus written → {corpus_path}\n")

    # ---- Write questions, mapping each to all chunks of its document ---------
    print("Loading questions from qaps.csv...")

    written = 0
    skipped_qns = 0

    with open(os.path.join(args.narrativeqa_dir, "qaps.csv"), encoding="utf-8") as qf, \
         open(questions_path, "w", encoding="utf-8") as out_f:

        for i, row in enumerate(csv.DictReader(qf)):
            doc_id = row["document_id"]

            if split_tag != "all" and row["set"] != split_tag:
                skipped_qns += 1
                continue

            if doc_id not in doc_chunk_map:
                skipped_qns += 1
                continue

            chunk_refs = [{"docid": cid} for cid in doc_chunk_map[doc_id]]

            out_f.write(json.dumps({
                "query_id": f"narrativeqa_{doc_id}_q{i}",
                "question": row["question"],
                "answers": [row["answer1"], row["answer2"]],
                "document_id": doc_id,
                "evidence_docs": chunk_refs,
                "gold_docs": chunk_refs,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"  Written : {written} questions")
    print(f"  Skipped : {skipped_qns}")
    print(f"Questions written → {questions_path}")
    print("\nDone.")
    print(f"  --corpus_path {corpus_path}")
    print(f"  --qns_path    {questions_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert NarrativeQA CSVs to chunked JSONL")
    p.add_argument("--narrativeqa_dir", type=str, default="narrativeqa-master")
    p.add_argument("--output_dir", type=str, default="narrative_qa")
    p.add_argument("--split", type=str, default="test", choices=["train", "test", "valid", "all"])
    p.add_argument("--chunk_size", type=int, default=6400,
                   help="Chunk size in words (default: 4000 ≈ ~5k tokens)")
    p.add_argument("--overlap", type=int, default=640,
                   help="Overlap between consecutive chunks in words (default: 400)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
