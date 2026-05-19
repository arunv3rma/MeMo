from langdetect import detect, LangDetectException, DetectorFactory
import json

DetectorFactory.seed = 42

file_dir = "/path/to/browsecomp_plus"
qns_path = f"{file_dir}/browsecomp_plus_questions.jsonl"

print(f"Using seed 42 for langdetect")

flagged_by_query_only = {}    # flagged by query text, NOT by evidence docs
flagged_by_both = {}          # flagged by both query text and evidence docs
flagged_by_docs_only = {}     # flagged by evidence docs only

with open(qns_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f):
        data = json.loads(line.strip())
        query_id = data.get('query_id')
        query_text = data.get('query', '')

        # Check 1: query text
        query_flagged = False
        query_lang = 'en'
        try:
            query_lang = detect(query_text)
            if query_lang != 'en':
                query_flagged = True
        except LangDetectException:
            query_lang = 'detection_failed'
            query_flagged = True

        # Check 2: evidence docs
        non_english_doc_count = 0
        for doc in data.get('evidence_docs', []):
            text = doc.get('text', '')
            try:
                doc_lang = detect(text) if text.strip() else 'en'
                if doc_lang != 'en':
                    non_english_doc_count += 1
            except LangDetectException:
                non_english_doc_count += 1

        docs_flagged = non_english_doc_count > 0

        if query_flagged and docs_flagged:
            flagged_by_both[query_id] = {'detected_lang': query_lang, 'non_english_doc_count': non_english_doc_count}
        elif query_flagged:
            flagged_by_query_only[query_id] = {'detected_lang': query_lang, 'query': query_text[:120]}
        elif docs_flagged:
            flagged_by_docs_only[query_id] = {'non_english_doc_count': non_english_doc_count}

total = line_num + 1

print(f"\n=== Query Language Detection Breakdown (total={total}) ===")

print(f"\nFlagged by query text ONLY ({len(flagged_by_query_only)}):")
for qid, info in flagged_by_query_only.items():
    print(f"  query_id={qid} | lang={info['detected_lang']} | query={info['query']}")

print(f"\nFlagged by evidence docs ONLY ({len(flagged_by_docs_only)}):")
for qid, info in flagged_by_docs_only.items():
    print(f"  query_id={qid} | non-english doc count={info['non_english_doc_count']}")

print(f"\nFlagged by BOTH query text and evidence docs ({len(flagged_by_both)}):")
for qid, info in flagged_by_both.items():
    print(f"  query_id={qid} | lang={info['detected_lang']} | non-english doc count={info['non_english_doc_count']}")

print(f"\n=== Summary ===")
print(f"  Flagged by query text only : {len(flagged_by_query_only)}")
print(f"  Flagged by evidence docs only: {len(flagged_by_docs_only)}")
print(f"  Flagged by both            : {len(flagged_by_both)}")
print(f"  Total unique flagged        : {len(flagged_by_query_only) + len(flagged_by_docs_only) + len(flagged_by_both)}")
