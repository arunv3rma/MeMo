from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import random

from datasets import load_dataset

from cartridges.data.resources import Resource, sample_seed_prompts, SEED_TYPES


TOPIC_TO_IDS = {
    "question": [
        '1908.06606',
        '1704.05572',
        '1905.08949',
        '1808.09920',
        '1603.01417',
        '1808.03986',
        '1907.08501',
        '1603.07044',
        '1903.00172',
        '1912.01046',
        '1909.00542',
        '1811.08048',
        '2004.02393',
        '1703.06492',
        '1607.06275',
        '1703.04617'
    ]
}

SYSTEM_PROMPT_TEMPLATE = """\
Below is (part of) a scientific paper. Please read it and be prepared to answer questions. 
<paper>
{paper}
</paper>
"""

class QASPERResource(Resource):
    class Config(Resource.Config):
        topic: str = "question"
        seed_prompts: List[SEED_TYPES] = ["generic"]


    
    def __init__(self, config: Config):
        self.config = config

        dataset = load_dataset("allenai/qasper", split="train")
        df = dataset.to_pandas()

        paper_ids = TOPIC_TO_IDS[self.topic]
        df = df[df["id"].isin(paper_ids)]
        assert len(df) == len(paper_ids)

        papers = []
        for row in df.to_dict(orient="records"):
            sections = []
            for section_idx, (section_title, paragraphs) in enumerate(zip(
                row["full_text"]["section_name"], row["full_text"]["paragraphs"]
            )):
                sections.append(Section(
                    title=section_title,
                    section_number=section_idx,
                    paragraphs=paragraphs.tolist()
                ))
            paper = Paper(
                id=row["id"],
                title=row["title"],
                abstract=row["abstract"],
                sections=sections
            )
            papers.append(paper)
        self.papers = papers
    
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        paper: Paper = random.choice(self.papers)
        num_sections_per_paper = random.randint(1, len(paper.sections))
        sections = random.sample(paper.sections, num_sections_per_paper)
        sections_str = "\n".join([section.to_string for section in sections])

        section_divider = f"\n---Paper Title: {paper.title}---\n"
        context = PAPER_TEMPLATE.format(
            title=paper.title,
            abstract=paper.abstract,
            sections=section_divider.join([section.text for section in sections])
        )
        ctx = SYSTEM_PROMPT_TEMPLATE.format(
            paper=context
        )

        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts

    def to_string(self) -> str:
        out = f"Below is a panel of scientific papers."
        for paper in self.papers:
            out += "\n\n"
            out += f"<paper>\n{paper.to_string()}\n</paper>\n"
        return out
        

SECTION_TEMPLATE = """\
<section>
<section-title>{title}</section-title>
<section-number>{section_number}</section-number>
<paragraphs>
{paragraphs}
</paragraphs>
</section>
"""

@dataclass
class Section:
    title: str
    section_number: int
    paragraphs: List[str]

    @property
    def text(self) -> str:
        paragraph_divider = "\n\n"
        return SECTION_TEMPLATE.format(
            title=self.title,
            section_number=self.section_number,
            paragraphs=paragraph_divider.join(self.paragraphs)
        )


PAPER_TEMPLATE = """\
<title>{title}</title>
<abstract>{abstract}</abstract>
<sections>
{sections}
</sections>
"""
@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    sections: List[Section]

    @property
    def to_string(self) -> str:
        section_divider = f"\n---Paper Title: {self.title}---\n"
        return PAPER_TEMPLATE.format(
            title=self.title,
            abstract=self.abstract,
            sections=section_divider.join([section.text for section in self.sections])
        )
