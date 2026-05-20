import json
from typing import List, Optional, Tuple, Dict
import random

from pydrantic import ObjectConfig
from transformers import PreTrainedTokenizerFast

from cartridges.data.ruler.niah import NIAHConfig, NIAHQuery, NIAHSample
from cartridges.data.ruler.variable_tracking import VariableTrackingConfig, VariableTrackingQuery, VariableTrackingSample
from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement
from cartridges.data.longhealth.utils import LongHealthQuestion, LongHealthPatient, load_longhealth_dataset
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING





class NIAHGenerateDataset(GenerateEvalDataset):
    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True
        niah_path: Optional[str] = None
        sample_idx: int = 0
        thinking: bool = True

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer


        with open(self.config.niah_path, "r") as f:
            self.data = json.load(f)
        
        # self.niah_config = NIAHConfig(**self.data["config"])

        sample = self.data["samples"][self.config.sample_idx]
        self.sample = NIAHSample(
            context=sample["context"],
            queries=[NIAHQuery(**query) for query in sample["queries"]]
        )
        self.queries = self.sample.queries
    

        self.tokenizer = tokenizer


    def __getitem__(
        self, index: int
    ) -> GenerateEvalDatasetElement:
        # convo: ContextConvo = ContextConvo.model_validate(self.data[index])
        queries: NIAHQuery = self.queries[index]

        kwargs = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = self.config.thinking
        elif self.config.thinking:
            cot_prompt = "Think before responding. Put your chain of thought between the <thinking> and </thinking> tags before providing your answer."
        else:
            cot_prompt = ""

        answer_prompt = queries.answer_prompt
        if len(queries.answers) > 1:
            answer_prompt = answer_prompt.replace("The special magic", f"The {len(queries.answers)} different special magic")
            duplicate_prompt = "Do not output the same value twice."
        else:
            duplicate_prompt = ""

        prompt = f"{queries.query}\n\n{cot_prompt}{answer_prompt}{duplicate_prompt}"

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,  
        )


        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            prompt=prompt,
            answer=queries.answers,
            convo_id=index,
            metadata={"idx": index}
        )

    def __len__(self):
        return len(self.queries)

    def score(
        self,
        pred: str,
        answer: List[str],
        convo_id: str
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        
        pred_answers = pred.split(":")[-1].strip("{}'\" .,\t\n")

        if len(answer) == 1:
            correct = str(answer[0]) == str(pred_answers)
        else:
            pred_answers = set([a.strip() for a in pred_answers.split(",")])
            answers = set(str(a) for a in answer)
            correct = pred_answers == answers

        return correct, {"pred_answers": str(pred_answers)}


class VariableTrackingGenerateDataset(GenerateEvalDataset):
    class Config(ObjectConfig):
        _pass_as_config = True
        variable_tracking_path: Optional[str] = None
        sample_idx: int = 0
        thinking: bool = True

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer

        with open(self.config.variable_tracking_path, "r") as f:
            self.data = json.load(f)
        
        sample = self.data["samples"][self.config.sample_idx]
        self.sample = VariableTrackingSample(
            context=sample["context"],
            queries=[VariableTrackingQuery(**query) for query in sample["queries"]]
        )
        self.queries = self.sample.queries

    def __getitem__(
        self, index: int
    ) -> GenerateEvalDatasetElement:
        query: VariableTrackingQuery = self.queries[index]

        kwargs = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = self.config.thinking
        elif self.config.thinking:
            cot_prompt = (
                "The variables are assigned in a chain of length 3."
                "Think through the chain step by step between <thinking> and </thinking> tags before providing your answer. "
                "For example, <thinking>12345 is equal to ABC, DEF is equal to ABC, GHI is equal to DEF</thinking>."
                "\n\n"
            )
        else:
            cot_prompt = ""

        # Combine context and query for variable tracking
        full_prompt = f"{query.query}\n\n{cot_prompt}{query.answer_prompt}"

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": full_prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,  
        )

        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            prompt=full_prompt,
            answer=query.answers,
            convo_id=index,
            metadata={"idx": index}
        )

    def __len__(self):
        return len(self.queries)

    def score(
        self,
        pred: str,
        answer: List[str],
        convo_id: str
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        
        import re
        
        # Extract predicted variables from <answer></answer> tags
        pred_variables = set()
        
        # Look for content within <answer></answer> tags
        for match in re.finditer(r'<answer>(.*?)</answer>', pred, re.DOTALL | re.IGNORECASE):
            answer_content = match.group(1).strip()
            # Split by lines and clean each variable name
            for line in answer_content.split('\n'):
                var = line.strip().lower()
                if var:  # Only keep alphabetic variable names
                    pred_variables.add(var)
        
        # Convert expected answers to set for comparison
        expected_variables = set(str(var).lower() for var in answer)
        
        # Calculate F1-score
        true_positives = len(pred_variables & expected_variables)
        false_positives = len(pred_variables - expected_variables)
        false_negatives = len(expected_variables - pred_variables)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1_score, {
            "pred_variables": sorted(list(pred_variables)),
            "expected_variables": sorted(list(expected_variables)),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
