from typing import Dict, List, Optional
import json
from pydantic import BaseModel
import requests


class LongHealthAnswerLocation(BaseModel):
    start: List[float]
    end: List[float]


# dict_keys(['No', 'question', 'answer_a', 'answer_b', 'answer_c', 'answer_d', 'answer_e', 'correct', 'answer_location'])
class LongHealthQuestion(BaseModel):
    question_id: str
    question: str
    correct: str

    answer_a: str
    answer_b: str
    answer_c: str
    answer_d: str
    answer_e: str

    # text_id -> answer_location in that text
    answer_location: Optional[Dict[str, LongHealthAnswerLocation]]

class LongHealthPatient(BaseModel):
    patient_id: str
    texts: Dict[str, str]
    name: str 
    birthday: str
    diagnosis: str
    questions: List[LongHealthQuestion]


    
DATASET_PATH = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"

def load_longhealth_dataset(patient_ids: Optional[List[str]] = None) -> List[LongHealthPatient]:
    response = requests.get(DATASET_PATH)
    response.raise_for_status()  # Raise an exception for HTTP errors
    data = json.loads(response.text)
    # rename no to idx
    for patient_id, row in data.items():
        for question in row["questions"]:
            question["question_id"] = patient_id + "_" + str(question["No"])

    patients = [
        LongHealthPatient(
            patient_id=patient_id,
            **row
        )
        for patient_id, row in data.items()
        if patient_ids is None or patient_id in patient_ids
    ]
    print("Num patients:", len(patients))
    return patients


DATASET_PATH = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"

def load_longhealth_dataset(patient_ids: Optional[List[str]] = None) -> List[LongHealthPatient]:
    response = requests.get(DATASET_PATH)
    response.raise_for_status()  # Raise an exception for HTTP errors
    data = json.loads(response.text)
    # rename no to idx
    for patient_id, row in data.items():
        for question in row["questions"]:
            question["question_id"] = patient_id + "_" + str(question["No"])

    patients = [
        LongHealthPatient(
            patient_id=patient_id,
            **row
        )
        for patient_id, row in data.items()
        if patient_ids is None or patient_id in patient_ids
    ]
    return patients
