import json
import gc
from musique_query_negatives_utils import MUSIQUE_NEGATIVE_DOC_IDS_PER_QUERY, MUSIQUE_NEGATIVE_DOCS_MAP


def load_corpus_from_jsonl_musique(file_path):
    """
    Loads unique documents from a MuSiQue corpus JSONL file.
    Corpus format: {"docid": "...", "text": "...", "url": "..."}

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


def load_only_query_related_docs_musique(corpus_path, qns_path, max_num_questions=None):
    """
    Filters the MuSiQue corpus to only documents referenced by questions.

    MuSiQue questions JSONL format:
        {"query_id": "2hop__...", "question": "...", "answers": [...],
         "evidence_docs": [{"docid": "..."}], "gold_docs": [{"docid": "..."}], ...}

    No pre-defined subset lists are used — just takes the first max_num_questions
    questions in file order.

    Returns:
        corpus_texts (list): Texts of query-related documents
        corpus_docids (list): Doc IDs of query-related documents
    """
    def get_query_related_doc_ids(qns_path, max_num_questions):
        query_doc_ids = set()
        valid_count = 0

        with open(qns_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_num_questions is not None and valid_count >= max_num_questions:
                    break

                data = json.loads(line.strip())

                evidence_docs = data.get('evidence_docs', [])
                gold_docs = data.get('gold_docs', [])

                for doc in evidence_docs + gold_docs:
                    doc_id = doc.get('docid')
                    if doc_id:
                        query_doc_ids.add(doc_id)

                valid_count += 1

        print(f"\n=== Loading Queries Summary ===")
        print(f"Valid questions loaded: {valid_count}")
        print(f"Num of unique chunk doc_ids: {len(query_doc_ids)}")

        return query_doc_ids

    query_doc_ids = get_query_related_doc_ids(qns_path, max_num_questions)

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


def load_only_query_related_docs_with_negatives_musique(corpus_path, qns_path, max_num_questions=None, neg_n=1):
    """
    Filters the MuSiQue corpus to documents referenced by questions, plus
    pre-selected N negative docs per query (N = len(evidence_docs), selected
    once via generate_musique_negative_doc_ids.py with a fixed seed).
    """
    def get_query_related_doc_ids(qns_path, max_num_questions):
        query_doc_ids = set()
        valid_count = 0

        with open(qns_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_num_questions is not None and valid_count >= max_num_questions:
                    break

                data = json.loads(line.strip())

                evidence_docs = data.get('evidence_docs', [])
                gold_docs = data.get('gold_docs', [])

                for doc in evidence_docs + gold_docs:
                    doc_id = doc.get('docid')
                    if doc_id:
                        query_doc_ids.add(doc_id)

                pre_selected_neg_ids = MUSIQUE_NEGATIVE_DOCS_MAP[neg_n].get(data['query_id'], [])
                query_doc_ids.update(pre_selected_neg_ids)

                valid_count += 1

        print(f"\n=== Loading Queries Summary ===")
        print(f"Valid questions loaded: {valid_count}")
        print(f"Num of unique doc_ids: {len(query_doc_ids)}")

        return query_doc_ids

    query_doc_ids = get_query_related_doc_ids(qns_path, max_num_questions)

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


def load_questions_with_evidence_and_negative_docs_musique(file_path, max_num_questions=None, neg_n=1):
    """
    Loads MuSiQue questions with evidence docs and pre-selected negative docs.
    Negative doc IDs come from MUSIQUE_NEGATIVE_DOC_IDS_PER_QUERY and are
    returned as [{'docid': ...}] to match the BCP negative_docs structure.
    """
    questions = []
    valid_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_num_questions is not None and valid_count >= max_num_questions:
                break

            data = json.loads(line.strip())
            query_id = data.get('query_id')
            evidence_docs = data.get('evidence_docs', [])

            pre_selected_neg_ids = MUSIQUE_NEGATIVE_DOCS_MAP[neg_n].get(query_id, [])
            negative_docs = [{'docid': doc_id} for doc_id in pre_selected_neg_ids]

            question_entry = {
                'question_no': query_id,
                'question': data.get('question'),
                'groundtruth': data.get('answers', []),
                'gold_docs': data.get('gold_docs', []),
                'evidence_docs': evidence_docs,
                'negative_docs': negative_docs,
                'total_evidence_tokens': 0,
                'evidence_doc_count': len(evidence_docs),
            }

            questions.append(question_entry)
            valid_count += 1

    print(f"\n=== Loading Summary ===")
    print(f"Total questions loaded: {valid_count}")

    return questions


def load_questions_with_evidence_docs_musique(file_path, max_num_questions=None):
    """
    Loads MuSiQue questions with their evidence documents.

    MuSiQue field names differ from BrowseComp+:
      - "question"  (not "query")
      - "answers"   (list, not "answer")
      - evidence_docs entries are {"docid": "..."} only — no embedded text

    Returns:
        questions (list): List of question entries with evidence docs attached
    """
    questions = []
    valid_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_num_questions is not None and valid_count >= max_num_questions:
                break

            data = json.loads(line.strip())

            evidence_docs = data.get('evidence_docs', [])

            question_entry = {
                'question_no': data.get('query_id'),
                'question': data.get('question'),
                'groundtruth': data.get('answers', []),
                'gold_docs': data.get('gold_docs', []),
                'evidence_docs': evidence_docs,
                # evidence_docs have no embedded text in MuSiQue
                'total_evidence_tokens': 0,
                'evidence_doc_count': len(evidence_docs),
            }

            questions.append(question_entry)
            valid_count += 1

    print(f"\n=== Loading Summary ===")
    print(f"Total questions loaded: {valid_count}")

    return questions
