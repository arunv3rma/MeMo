from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional, TypedDict

from cartridges.clients.base import FlatTopLogprobs
from cartridges.utils import get_logger


logger = get_logger(__name__)


class MessageDict(TypedDict):
    """This is simply a convenience type for typehints for a message dictionary
    compatible with OpenAI-apis and tokenizer.apply_chat_template.
    
    It differs from Message, which is a dataclass that also has fields for token_ids and 
    top_logprobs.
    """
    role: Literal["user", "assistant", "system"]
    content: str

@dataclass
class Conversation:
    messages: list[Conversation.Message]
    system_prompt: str
    metadata: dict
    type: Optional[str] = None


    @dataclass
    class Message:
        content: str
        role: Literal["user", "assistant", "system"]
        token_ids: Optional[List[int]]
        
        # Sparse dictionary of top logprobs for each token
        top_logprobs: Optional[FlatTopLogprobs] = None

        def to_message_dict(self) -> MessageDict:
            return {"content": self.content, "role": self.role}

    def _repr_html_(self) -> str:
        import markdown

        html = """
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <div class='context-convo p-4'>
        """
        for message in self.messages:
            if message.role == "user":
                role_class = "bg-blue-100 text-blue-800"
            else:
                role_class = "bg-green-100 text-green-800"
            role_display = f"<strong style='font-size: 1.5em;'>{message.role.capitalize()}</strong>"
            content_html = markdown.markdown(message.content)
            html += f"""
            <div class='p-2 my-2 rounded {role_class}'>
                {role_display} {content_html}
            </div>
            """
        html += "</div>"
        return html

    def to_html(self) -> str:
        return self._repr_html_()
    
    @staticmethod
    def from_dict(row: dict) -> Conversation:
        import numpy as np

        def _to_flat(tl):
            if tl is None:
                return None
            return FlatTopLogprobs(
                token_idx=np.asarray(tl["token_idx"]),
                token_id=np.asarray(tl["token_id"]),
                logprobs=np.asarray(tl["logprobs"]),
                shape=tuple(tl["shape"]),
            )

        return Conversation(
            messages=[
                Conversation.Message(
                    content=message["content"],
                    role=message["role"],
                    token_ids=message["token_ids"],
                    top_logprobs=_to_flat(message["top_logprobs"]),
                )
                for message in row["messages"]
            ],
            system_prompt=row["system_prompt"],
            metadata=row["metadata"],
            type=row["type"],
        )

def write_conversations(conversations: list[Conversation], path: str):
    path_str = str(path)
    if path_str.endswith(".parquet"):
        _conversations_to_parquet(conversations, path)
    elif path_str.endswith(".pkl"):
        _conversations_to_pkl(conversations, path)
    else:
        raise ValueError(f"Unsupported file extension: {path_str}")

def read_conversations(path: str) -> list[Conversation]:
    path_str = str(path)
    if path_str.endswith(".parquet"):
        return _conversations_from_parquet(path)
    elif path_str.endswith(".pkl"):
        return _conversations_from_pkl(path)
    else:
        raise ValueError(f"Unsupported file extension: {path_str}")

def _conversations_to_parquet(conversations: list[Conversation], path: str):
    import pyarrow.parquet as pq
    import pyarrow as pa
    from dataclasses import asdict

    rows = (asdict(row) for row in conversations)
    table = pa.Table.from_pylist(list(rows)) 
    pq.write_table(table, path, compression="snappy")

def _conversations_from_parquet(path: str) -> list[Conversation]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    conversations = []
    for batch in pf.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            conversations.append(Conversation.from_dict(row))
    return conversations

def _conversations_to_pkl(conversations: list[Conversation], path: str):
    """For backwards compatibility, we will eventually only support parquet as it is 
    roughly half the size of pkl."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(conversations, f)
    
def _conversations_from_pkl(path: str) -> list[Conversation]:
    """For backwards compatibility, we will eventually only support parquet as it is 
    roughly half the size of pkl."""
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "rows" in data:
        # backwards compatibility
        return data["rows"]
    else:
        return data

class TrainingExample(Conversation):
    # backwards compatibility
    pass