from typing import Dict, List, Optional, Tuple, Any
import random

from cartridges.utils import get_logger
from cartridges.data.resources import Resource, sample_seed_prompts, SEED_TYPES
from cartridges.data.longhealth.utils import load_longhealth_dataset
logger = get_logger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
Below is a section of {name}'s medical record (ID: {patient_id}). 
They were born on {birthday} and have the following diagnosis: {diagnosis}.
The patients medical record consists of {num_notes} notes.
{notes}"""

NOTE_TEMPLATE = """\
<{note_id}>
{text}
</{note_id}>
"""

FULL_STRING_TEMPLATE = """\
<patient-record-{patient_id}>
Below is patient {name}'s medical record (ID: {patient_id}). 
They were born on {birthday} and have the following diagnosis: {diagnosis}.
The patients medical record consists of {num_notes} notes included below.
<notes>
{notes}
</notes>
</patient-record-{patient_id}>"""

class LongHealthResource(Resource):
    class Config(Resource.Config):
        patient_ids: Optional[List[str]] = None
        max_notes_per_prompt: int = 1
        min_notes_per_prompt: int = 1
        max_chars_per_note: Optional[int] = None
        seed_prompts: List[SEED_TYPES] = ["generic"]
        
    
    def __init__(self, config: Config):
        self.config = config
        self.patients = load_longhealth_dataset(self.config.patient_ids)
    
    def _chunk_note(self, note: str) -> List[str]:
        if self.config.max_chars_per_note is None or len(note) <= self.config.max_chars_per_note:
            return note

        start_idx = random.randint(0, len(note) - self.config.max_chars_per_note)
        end_idx = start_idx + self.config.max_chars_per_note
        note_chunk = note[start_idx:end_idx]

        return f"... {note_chunk} ..."

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        patient = random.choice(self.patients)
        num_notes = random.randint(self.config.min_notes_per_prompt, self.config.max_notes_per_prompt)
        note_ids = random.sample(list(patient.texts.keys()), min(num_notes, len(patient.texts)))
        texts = [self._chunk_note(patient.texts[note_id]) for note_id in note_ids]
        notes = "\n".join([NOTE_TEMPLATE.format(note_id=note_id, text=text) for note_id, text in zip(note_ids, texts)])
        
        ctx = SYSTEM_PROMPT_TEMPLATE.format(
            name=patient.name,
            patient_id=patient.patient_id,
            birthday=patient.birthday,
            diagnosis=patient.diagnosis,
            num_notes=num_notes,
            notes=notes,
        )
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts

    def to_string(self) -> str:
        out = f"Below is a panel of patient records."
        for patient in self.patients:
            notes = "\n".join([f"<{note_id}>\n{text}\n</{note_id}>" for note_id, text in patient.texts.items()])
            out += "\n\n"
            out += FULL_STRING_TEMPLATE.format(
                name=patient.name,
                patient_id=patient.patient_id,
                birthday=patient.birthday,
                diagnosis=patient.diagnosis,
                num_notes=len(patient.texts),
                notes=notes,
            )
        return out
        