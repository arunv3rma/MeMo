from typing import Literal, TypedDict

Role = Literal["user", "assistant", "system"]

class Message(TypedDict):
    role: Role
    content: str

Conversation = list[Message]

def user_msg(content) -> Message:
    return Message(role='user', content=content)

def assistant_message(content) -> Message:
    return Message(role='assistant', content=content)

def system_msg(content) -> Message:
    return Message(role='system', content=content)
