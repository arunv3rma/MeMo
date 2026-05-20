
from dataclasses import dataclass, asdict
import random
from datasets import load_dataset
import pydrantic

import asyncio
import os
import json
from typing import List, Dict, Any, Optional
import openai

from cartridges.data.qasper.resources import TOPIC_TO_IDS


# Set your OpenAI API key here or via environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

@dataclass
class RewrittenQasperQuestion:
    paper_id: str
    title: str
    abstract: str
    
    question: str
    answer: str

    old_answer: str
    old_question: str
    

ANSWER_PROMPT = """\
Can you please write a succinct answer to the following question based on the information provided?
You do not need to restate the paper name or answer in complete sentences. 

<question>
{question}
</question>

<answer-details>
{answer_details}
</answer-details>
"""

QUESTION_PROMPT = """\
Can you please rewrite the following question to be specific about which paper it is asking about?
The question should be answerable in a closed-book setting. It should require knowledge of the paper.
Do not output anything but the rewritten question.

<title>
{title}
</title>

<question>
{question}
</question>
"""

# --- OPENAI MIGRATION: Use openai>=1.0.0 API ---

async def async_openai_completion(prompt: str, model: str, max_tokens: int = 256) -> str:
    # Use the OpenAI async API for chat completion (openai>=1.0.0)
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

async def rewrite_questions(requests: List[Dict[str, Any]], model: str) -> List[str]:
    # Rewrite all questions asynchronously
    tasks = []
    for req in requests:
        prompt = QUESTION_PROMPT.format(
            title=req["title"],
            question=req["question"]
        )
        tasks.append(async_openai_completion(prompt, model=model))
    return await asyncio.gather(*tasks)

async def rewrite_answers(requests: List[Dict[str, Any]], new_questions: List[str], model: str) -> List[str]:
    # Rewrite all answers asynchronously, using the new questions
    tasks = []
    for req, new_q in zip(requests, new_questions):
        prompt = ANSWER_PROMPT.format(
            question=new_q,
            answer_details=req["answer_data"]
        )
        tasks.append(async_openai_completion(prompt, model=model))
    return await asyncio.gather(*tasks)


class RewriteQasperConfig(pydrantic.RunConfig):
    topic: str
    model: str = "gpt-4.1-2025-04-14"
    limit: Optional[int] = None

    def run(self):
        ids = TOPIC_TO_IDS[self.topic]
        dataset = load_dataset("allenai/qasper", split="train")
        df = dataset.to_pandas()
        df = df[df["id"].isin(ids)]
        papers = df.to_dict(orient="records")

        if self.limit is not None:
            papers = papers[:self.limit]

        requests = []
        for paper in papers:
            qas = paper["qas"]
            for question, answer in zip(qas["question"], qas["answers"]):
                answer = answer["answer"][0]
                if answer["unanswerable"]:
                    continue
                requests.append(
                    {
                        "paper_id": paper["id"],
                        "question": question,
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                        "answer_data": answer,
                        "old_answer": answer,
                    }
                )

        # Run async rewriting
        async def process():
            print(f"Rewriting {len(requests)} questions...")
            new_questions = await rewrite_questions(requests, self.model)
            print("Questions rewritten. Rewriting answers...")
            new_answers = await rewrite_answers(requests, new_questions, self.model)
            print("Answers rewritten. Saving...")

            rewritten: List[RewrittenQasperQuestion] = []
            for req, new_q, new_a in zip(requests, new_questions, new_answers):
                rewritten.append(
                    RewrittenQasperQuestion(
                        paper_id=req["paper_id"],
                        title=req["title"],
                        abstract=req["abstract"],
                        question=new_q,
                        answer=new_a,
                        
                        old_question=req["question"],
                        old_answer=req["old_answer"],
                    )
                )
            return rewritten
        rewritten = asyncio.run(process())
       
        from datasets import Dataset, DatasetDict
        for question in random.sample(rewritten, 5):
            print(f"Question: {question.question}")
            print(f"Old question: {question.old_question}")
            print(f"Answer: {question.answer}")
            print("-"*100)

        # Convert to dicts
        data = [asdict(r) for r in rewritten]
        ds = Dataset.from_list(data)

        # Optionally, create a DatasetDict if you want to upload multiple splits
        dataset_dict = DatasetDict({self.topic: ds})

        # Push to the hub
        # Change "your-username/qasper-rewritten" to your actual namespace/repo
        repo_id = f"sabrieyuboglu/qasper-rewrite-{self.model}"
        dataset_dict.push_to_hub(repo_id, private=False)
        print(f"Pushed to {repo_id}")
        

if __name__ == "__main__":
    config = RewriteQasperConfig(
        topic="question",
        model="gpt-4.1",
    )
    pydrantic.main([config])