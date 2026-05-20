import json
import csv

BROWSECOMP_QUERY_IDS_TO_SKIP = ['802', '1068', '239', '199', '1248', '1076', '970', '394', '168', '861', '875', '202', '987', '406', '998', '311', '1207', '122', '471', '1211', '261', '805', '865', '986', '491', '372', '376', '424', '1234', '289', '403', '87', '1048', '1066', '788', '1206', '770', '131', '1046', '72', '486', '1021', '1058', '270', '639', '784', '390', '88', '1196', '769', '852', '1220', '1022', '23', '823', '494', '819', '876', '1236', '70', '395', '152', '520']
NON_EN_DOC_IDS_TO_SKIP = ['5390', '20259', '21886', '26863', '51408', '82736', '33577', '300', '93816', '88493', '64853', '25270', '15859', '28887', '99960', '65611', '92138', '11077', '43019', '19626', '95870', '57773', '13820', '96239', '44078', '79461', '13042', '69438', '27103', '38209', '37511', '13089', '46015', '10072', '88221', '27811', '95780', '93067', '91588', '62642', '96776', '93450', '65274', '52641', '91774', '88280', '99877', '11445', '57655', '36089', '25108', '33140', '52367', '60857', '56926', '50270', '16442', '1121', '1669', '72686', '84849', '11363', '4435', '31710', '57139', '2466', '8926', '73790', '80825', '16601', '18311', '88060', '64303', '15716', '28510', '42576', '37097', '53024', '25415', '32949', '7445', '13008', '88039', '45464', '22680', '77347', '1016', '26826', '88853', '93767', '14015', '23025', '45258', '33204', '4551', '277', '85607', '62794', '15639', '44317', '17400', '99740', '50781', '53929', '46031', '12183', '95549', '94477', '30615', '100159', '53777', '13524', '14444', '11576', '96100', '95111', '97086', '48563', '62695', '54249', '42890', '67225', '46965', '51818', '70007', '38301', '77559', '12019', '58490', '34605', '91654', '92091', '97992', '48019', '65691', '79445', '12743', '4720', '63460', '43436', '48837', '62685', '60168', '94187', '54897', '40897', '43180', '81299', '77481', '41336', '48342', '57218', '1124', '53729', '17904', '75257', '27213', '60684', '33239', '32130', '96316', '51609', '35484', '23206', '66159', '16009', '31437', '80445', '31397', '87233', '30078', '95429', '13099', '40967', '92423', '93940', '93294', '68046', '27836', '91479', '76255', '88205', '54252', '31177', '11693', '64159', '34828', '1395', '59098', '63593', '18626', '59338', '94191', '41500', '42749', '79037', '34975', '92517', '33332', '88682', '48297', '82646', '91372', '7370', '64521', '59470', '60452', '34385', '3069', '5491', '80169', '92750', '82796', '15582', '85910', '34752', '65824', '77681', '50079', '1863', '95322', '46822', '38742', '30545', '99673', '2497', '94853', '86536', '30378', '36943', '74366', '3464', '24220', '30438', '10680', '42657', '73996', '81841', '12257', '94615', '30415', '44585', '78103', '90523', '47650', '77982', '70010', '30085', '11335', '58640', '67940', '6873', '51820', '93070', '35812', '82527', '47500', '75548', '18639', '93660', '23753', '37080', '15952', '35398', '29856', '44768', '59349', '13617', '41480', '64221', '93346', '50808', '98645', '83701', '9438', '41759', '92201', '76983', '50482', '13878', '71947', '89196', '57599', '51193', '36133', '74736', '7823', '93768', '61369', '15801', '19204', '35354', '84312', '44044', '48607', '21696', '95710', '69958', '5166', '15518', '20464', '27123', '65121', '62629', '92358', '96707', '97267', '61101', '25140', '5596', '99037', '43209', '60432', '86196', '25252', '99138', '70580', '49795', '82668', '86896', '9337', '45615', '11457', '12999', '27323', '18614', '39367', '92534', '40264', '71223', '78706', '21839', '46674', '63815', '77025', '94285', '9641', '55793', '65051', '88698', '92098', '43214', '79478', '55328', '99005', '61342', '7143', '74546', '78234', '99462', '19317', '4698', '68818', '12516', '39571', '8709', '79341', '31010', '28668', '40336', '87787', '58460', '8240', '77689', '6743', '47269', '51013', '57542']

def load_corpus(file_path, get_chunks, return_as_list):
    """
    Loads corpus from a JSON file. 
    Assumes the JSON is a list of strings OR a list of dicts with a 'text' or 'content' field.

    # For 100 Speculative Decoding papers dataset:
    Specify get_chunks=True to return the raw file (which are already chunks). Otherwise, get_chunks=False will give you the whole papers.
    Specify return_as_list=True to return a list of papers as strings. Otherwise, return_as_list=False will return a dict of {title: text}.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not get_chunks:
        if "100papers" not in file_path:    # For 15 Speculative Decoding papers dataset
            return data

        if return_as_list:  # For 100 Speculative Decoding papers dataset
            raw_data = []  
            for paper in data:
                title = paper.get("filename", "N/A")
                if title.endswith(".pdf"):
                    title = title[:-4]
                paper_text = ""
                for chunk in paper["sections"]:
                    paper_text += chunk["title"] + " " + chunk["content"]
                raw_data.append("Paper Title: " +  title + "\n\nPaper Text: " + paper_text)
            print("Check first entry of raw_data (first 200 characters): ", raw_data[0][:200])
        else:
            raw_data = {}
            for paper in data:
                title = paper.get("filename", "N/A")
                if title.endswith(".pdf"):
                    title = title[:-4]
                paper_text = ""
                for chunk in paper["sections"]:
                    paper_text += chunk["title"] + " " + chunk["content"]
                raw_data[title] = paper_text
            print("Check first entry of raw_data (first 200 characters): ", list(raw_data.items())[0][1][:200])
        
        return raw_data
    
    corpus_texts = []
    # Handle the specific structure mentioned in previous context (list of papers with chunks)
    for paper in data:
        # Fallback if structure is simple list of strings
        if isinstance(paper, str):
            corpus_texts.append(paper)
        elif "chunks" in paper:
            for chunk in paper["chunks"]:
                corpus_texts.append("Paper Title: " + paper.get("title", "N/A") + "\n\n" + chunk)
        elif "sections" in paper:
            title = paper.get("filename", "N/A")
            if title.endswith(".pdf"):
                title = title[:-4]
            for chunk in paper["sections"]:
                text = chunk["title"] + " " + chunk["content"]
                corpus_texts.append("Paper Title: " + title + "\n\n" + text)
        else:
            # Fallback for generic dicts
            corpus_texts.append(str(paper))

    print("Check first entry of corpus_text (first 200 characters): ", corpus_texts[0][:200])  # Print first 200 characters
    return corpus_texts

def load_questions(file_path):
    """Loads questions from a CSV file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    result = []

    if 'cat_1' in file_path:    # For 15 Speculative Decoding papers dataset
        key_map_15papers = {'Question No.': 'question_no', 'Question Type': 'question_type', 'Question': 'question', 'Suggested Answer Key': 'groundtruth', 'Justification (if any)': 'groundtruth_justification'}
        for entry in data:
            new_data = {key_map_15papers.get(k, k): v for k, v in entry.items()}
            result.append(new_data)
    elif '100papers' in file_path:  # For 100 Speculative Decoding papers dataset
        key_map_100papers = {'formatted_question': 'question', 'correct_answer': 'groundtruth', "summarised_justification": "groundtruth_justification"}
        for i, entry in enumerate(data):
            new_data = {'question_no': i+1}
            new_data.update({key_map_100papers.get(k, k): v for k, v in entry.items()})
            result.append(new_data)
    
    print("Check first entry of questions: ", result[0])
    return result

# for in context learning
def load_questions_with_evidence_docs(file_path, max_valid_questions=None):
    """
    Loads questions with their evidence documents directly from JSONL file.
    No retrieval step - just uses all evidence_docs for each question.
    
    Args:
        file_path: Path to the JSONL file
        max_valid_questions: Number of valid (non-skipped) questions to load
        max_evidence_tokens: Maximum total token length for all evidence docs (default 120000)
    
    Returns:
        questions (list): List of question entries with evidence docs already attached
    """
    questions = []
    valid_question_count = 0
    invalid_question_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            # Stop if we have enough valid questions
            if max_valid_questions is not None and valid_question_count >= max_valid_questions:
                break
            
            data = json.loads(line.strip())
            
            if data['query_id'] in BROWSECOMP_QUERY_IDS_TO_SKIP:
                invalid_question_count += 1
                continue
            
            # Calculate total evidence docs tokens
            total_evidence_tokens = 0
            evidence_doc_count = 0
            
            # Collect evidence docs
            evidence_docs = data.get('evidence_docs', [])
            
            for doc in evidence_docs:
                text = doc.get('text', '')
                
                # Calculate tokens for this doc
                char_length = len(text)
                approx_tokens = char_length / 4
                total_evidence_tokens += approx_tokens
                evidence_doc_count += 1
            
            # Create question entry with evidence docs attached
            question_entry = {
                'question_no': data.get('query_id'),
                'question': data.get('query'),
                'groundtruth': data.get('answer'),
                'gold_docs': data.get('gold_docs', []),
                'evidence_docs': evidence_docs,
                'total_evidence_tokens': int(total_evidence_tokens),
                'evidence_doc_count': evidence_doc_count
            }
            
            questions.append(question_entry)
            valid_question_count += 1
    
    print(f"\n=== Loading Summary ===")
    print(f"Total lines read: {line_num}")
    print(f"Valid questions loaded: {valid_question_count}")
    print(f"Skipped questions: {invalid_question_count}")
    
    if questions:
        print(f"Check first entry of questions: {questions[0]}")
    
    return questions

def load_corpus_from_jsonl(file_path):
    """
    Loads only the corpus (unique documents) from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
    
    Returns:
        corpus_texts (list): List of document texts
        corpus_docids (list): List of document IDs (parallel to corpus_texts)
    """
    corpus_dict = {}  # Store unique documents: {docid: {'text': ..., 'url': ...}}
    
    print(f"Loading corpus from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            doc = json.loads(line.strip())
            
            # Extract documents from all available fields
            docid = doc.get('docid')
            if docid in NON_EN_DOC_IDS_TO_SKIP:
                continue
            
            text = doc.get('text', '')
            
            # Store unique documents
            if docid and docid not in corpus_dict:
                corpus_dict[docid] = {
                    'text': text,
                    'url': doc.get('url', '')
                }
            
            # Print progress every 1000 lines
            if line_num % 1000 == 0:
                print(f"  Processed {line_num} lines, found {len(corpus_dict)} unique documents...")
    
    # Create parallel lists for texts and docids
    corpus_docids = list(corpus_dict.keys())
    corpus_texts = [corpus_dict[docid]['text'] for docid in corpus_docids]
    
    # Print summary
    print(f"\n=== Corpus Loading Summary ===")
    print(f"Total lines processed: {line_num}")
    print(f"Unique documents found: {len(corpus_texts)}")
    
    if corpus_texts:
        print(f"\nSample - First document:")
        print(f"  DocID: {corpus_docids[0]}")
        print(f"  Text (first 200 chars): {corpus_texts[0][:200]}...")
    
    return corpus_texts, corpus_docids