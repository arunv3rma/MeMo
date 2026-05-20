from typing_extensions import Literal

from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

from cartridges.generate.structs import ContextConvo, Message
from cartridges.data.qasper.rewrite import RewrittenQasperQuestion
from cartridges.datasets import CartridgePerplexityDataset

PROMPT = """\
Please write a succinct answer to the following question.
You do not need to restate the paper name or answer in complete sentences.

<question>
{question}
</question>

Provide your answer in the following format (output nothing else):

<answer>
{{your answer here}}
</answer>"""

class QasperEvalDataset(CartridgePerplexityDataset):
    class Config(CartridgePerplexityDataset.Config):
        _pass_as_config = True
        
        dataset: str = "sabrieyuboglu/qasper-rewrite-gpt-4.1"
        topic: Literal["question"]


    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        
        self.dataset = load_dataset(config.dataset, split=config.topic)
        self.questions = [RewrittenQasperQuestion(**q) for q in self.dataset]

    def __getitem__(self, index: int) -> CartridgeDatasetElement:
        self.data = [
            ContextConvo(
                messages=[
                    Message(
                        role="user",
                        content=PROMPT.format(question=question.question),
                    ),
                    Message(
                        role="assistant",
                        content=f"<answer>\n{question.answer}\n</answer>",
                    )
                ],
                type="qasper",
                metadata={
                    "question_id": f"{idx}-{question.paper_id}",
                    "paper_id": question.paper_id,
                    "title": question.title,
                    "abstract": question.abstract,
                }
            )
            for idx, question in enumerate(self.questions)
        ]


        self.tokenizer = tokenizer