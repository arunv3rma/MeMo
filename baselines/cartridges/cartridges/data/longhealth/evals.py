from typing import List, Optional, Tuple, Dict
import random

from pydrantic import ObjectConfig
from transformers import PreTrainedTokenizerFast

from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement
from cartridges.data.longhealth.utils import LongHealthQuestion, LongHealthPatient, load_longhealth_dataset
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING



class LongHealthMultipleChoiceGenerateDataset(GenerateEvalDataset):
    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True
        patient_ids: Optional[List[str]] = None
        max_questions: Optional[int] = None
        include_diagnosis: bool = True
        cot: bool = True


    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer
        
        self.patients = load_longhealth_dataset(config.patient_ids)
        
        def wrap_question(question: LongHealthQuestion, patient: LongHealthPatient):
            options = (
                f"{question.answer_a}\n"
                f"{question.answer_b}\n"
                f"{question.answer_c}\n"
                f"{question.answer_d}\n"
                f"{question.answer_e}"
            )
            if self.config.cot and self.tokenizer.name_or_path not in MODELS_WITH_THINKING:
                cot_prompt = "You should first think step by step. Then give your final answer exactly as it appears in the options. Your output should be in the following format: \n<thinking> {{YOUR_THOUGHT_PROCESS}} </thinking> "
            else:
                cot_prompt = "Please provide your answer exactly as it appears in the options with the following format:"
            
            if self.config.include_diagnosis:
                patient_info = f"ID {patient.patient_id}, Name: {patient.name}, Birthday: {patient.birthday}, Diagnosis: {patient.diagnosis}"
            else:
                patient_info = f"ID {patient.patient_id}, Name: {patient.name}, Birthday: {patient.birthday}"
                
            return (
                "Please answer the question below about the following patient: "
                f"{patient_info}"
                f"\n\n<question>\n{question.question}\n</question>"
                f"\n\n<options>\n{options}\n</options>\n{cot_prompt}"
                f"\n\n<answer>\n{{YOUR_ANSWER}}\n</answer>"
            )
         
        self.questions = [
            LongHealthQuestion(
                question_id=question.question_id,
                question=wrap_question(question, patient),
                correct=question.correct,
                answer_a=question.answer_a,
                answer_b=question.answer_b,
                answer_c=question.answer_c,
                answer_d=question.answer_d,
                answer_e=question.answer_e,
                answer_location=question.answer_location,
            )
            for patient in self.patients
            for question in patient.questions
        ]
        random.Random(seed).shuffle(self.questions)


        if self.config.max_questions is not None:
            self.questions = self.questions[:self.config.max_questions]
        self.question_id_to_idx = {
            question.question_id: idx for idx, question in enumerate(self.questions)
        }


        self.tokenizer = tokenizer


    def __getitem__(
        self, index: int
    ) -> GenerateEvalDatasetElement:
        # convo: ContextConvo = ContextConvo.model_validate(self.data[index])
        question: LongHealthQuestion = self.questions[index]

        kwargs = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = self.config.cot

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question.question}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,
        )

        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            prompt=question.question,
            answer=question.correct,
            convo_id=question.question_id,
            metadata={"idx": index}
        )

    def __len__(self):
        return len(self.questions)

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        from difflib import SequenceMatcher
        def find_best_match(reference, candidates):
            return max(candidates, key=lambda x: SequenceMatcher(None, reference, x).ratio())
        
        # Extract the answer between <answer> and </answer> tags
        import re
        question: LongHealthQuestion = self.questions[self.question_id_to_idx[convo_id]]

        options = [
            question.answer_a.strip().lower(),
            question.answer_b.strip().lower(),
            question.answer_c.strip().lower(),
            question.answer_d.strip().lower(),
            question.answer_e.strip().lower(),
        ]

        pred_match = re.search(r'<answer>(.*?)</answer>', pred, re.DOTALL)
        if pred_match:
            extracted_pred = pred_match.group(1).strip().lower()

            closest_match = find_best_match(extracted_pred, options)

            return closest_match == answer.strip().lower(), {"extracted_pred": extracted_pred}
        else:
            # If no tags found, random guess
            pred = question.answer_a
            return pred.strip().lower() == answer.strip().lower(), {"extracted_pred": None}
        