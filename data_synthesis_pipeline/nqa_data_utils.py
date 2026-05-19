import json
import gc

from nqa_subset_utils import SUBSET_10_DOC_IDS, SUBSET_5_1_DOC_IDS, SUBSET_5_2_DOC_IDS

SUBSET_MAP = {
    10: SUBSET_10_DOC_IDS,
    5_1: SUBSET_5_1_DOC_IDS,
    5_2: SUBSET_5_2_DOC_IDS,
}


def load_corpus_from_jsonl_nqa(file_path):
    """
    Loads unique documents from a NarrativeQA corpus JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        corpus_texts (list): List of document texts
        corpus_docids (list): List of document IDs (parallel to corpus_texts)
    """
    corpus_dict = {}

    print(f"Loading corpus from {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            doc = json.loads(line.strip())

            docid = doc.get('docid')
            text = doc.get('text', '')

            if docid and docid not in corpus_dict:
                corpus_dict[docid] = {
                    'text': text,
                    'url': doc.get('url', '')
                }

            if line_num % 1000 == 0:
                print(f"  Processed {line_num} lines, found {len(corpus_dict)} unique documents...")

    corpus_docids = list(corpus_dict.keys())
    corpus_texts = [corpus_dict[docid]['text'] for docid in corpus_docids]

    print(f"\n=== Corpus Loading Summary ===")
    print(f"Total lines processed: {line_num}")
    print(f"Unique documents found: {len(corpus_texts)}")

    if corpus_texts:
        print(f"\nSample - First document:")
        print(f"  DocID: {corpus_docids[0]}")
        print(f"  Text (first 200 chars): {corpus_texts[0][:200]}...")

    return corpus_texts, corpus_docids


def load_only_query_related_docs_nqa(corpus_path, qns_path, max_num_docs=None):
    """
    Filters the NarrativeQA corpus to only documents referenced by questions.

    Args:
        corpus_path: Path to the corpus JSONL file
        qns_path: Path to the questions JSONL file
        max_num_docs: If set, limit to the first N unique source documents
                      (by document_id). All chunk docids for those docs are included.

    Returns:
        corpus_texts (list): Texts of query-related documents
        corpus_docids (list): Doc IDs of query-related documents
    """
    def get_query_related_doc_ids(qns_path, max_num_docs):
        seen_source_docs = set()
        query_doc_ids = set()

        subset_doc_ids = SUBSET_MAP.get(max_num_docs) if max_num_docs is not None else None

        with open(qns_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())

                source_doc_id = data.get('document_id')
                if max_num_docs is not None:
                    if subset_doc_ids is not None:
                        if source_doc_id not in subset_doc_ids:
                            continue
                        seen_source_docs.add(source_doc_id)
                    else:
                        if source_doc_id not in seen_source_docs:
                            if len(seen_source_docs) >= max_num_docs:
                                continue
                            seen_source_docs.add(source_doc_id)

                evidence_docs = data.get('evidence_docs', [])
                gold_docs = data.get('gold_docs', [])

                for doc in evidence_docs + gold_docs:
                    doc_id = doc.get('docid')
                    if doc_id:
                        query_doc_ids.add(doc_id)

        print(f"\n=== Loading Queries Summary ===")
        print(f"Unique source docs loaded: {len(seen_source_docs)}")
        print(f"Num of unique chunk doc_ids: {len(query_doc_ids)}")

        return query_doc_ids

    query_doc_ids = get_query_related_doc_ids(qns_path, max_num_docs)

    corpus_dict = {}

    print(f"Loading corpus from {corpus_path}...")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            doc = json.loads(line.strip())

            docid = doc.get('docid')

            if docid not in query_doc_ids:
                continue

            text = doc.get('text', '')

            if docid and docid not in corpus_dict:
                corpus_dict[docid] = {
                    'text': text,
                    'url': doc.get('url', '')
                }

            if line_num % 1000 == 0:
                print(f"  Processed {line_num} lines, found {len(corpus_dict)} unique documents...")

    corpus_docids = list(corpus_dict.keys())
    corpus_texts = [corpus_dict[docid]['text'] for docid in corpus_docids]

    del corpus_dict
    gc.collect()

    print(f"\n=== Corpus Loading Summary ===")
    print(f"Total lines processed: {line_num}")
    print(f"Unique documents found: {len(corpus_texts)}")

    return corpus_texts, corpus_docids


def load_questions_with_evidence_docs_nqa(file_path, max_num_docs=None):
    """
    Loads NarrativeQA questions with their evidence documents.

    Args:
        file_path: Path to the questions JSONL file
        max_num_docs: If set, limit to the first N unique source documents
                      (by document_id). All questions for those docs are included.

    Returns:
        questions (list): List of question entries with evidence docs attached
    """
    questions = []
    seen_source_docs = set()

    subset_doc_ids = SUBSET_MAP.get(max_num_docs) if max_num_docs is not None else None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())

            source_doc_id = data.get('document_id')
            if max_num_docs is not None:
                if subset_doc_ids is not None:
                    if source_doc_id not in subset_doc_ids:
                        continue
                    seen_source_docs.add(source_doc_id)
                else:
                    if source_doc_id not in seen_source_docs:
                        if len(seen_source_docs) >= max_num_docs:
                            continue
                        seen_source_docs.add(source_doc_id)

            evidence_docs = data.get('evidence_docs', [])
            total_evidence_tokens = sum(len(doc.get('text', '')) / 4 for doc in evidence_docs)

            answers = data.get('answers', [])

            question_entry = {
                'question_no': data.get('query_id'),
                'question': data.get('question'),
                'groundtruth': answers,
                'gold_docs': data.get('gold_docs', []),
                'evidence_docs': evidence_docs,
                'total_evidence_tokens': int(total_evidence_tokens),
                'evidence_doc_count': len(evidence_docs),
            }

            questions.append(question_entry)

    print(f"\n=== Loading Summary ===")
    print(f"Unique source docs loaded: {len(seen_source_docs)}")
    print(f"Total questions loaded: {len(questions)}")

    return questions
