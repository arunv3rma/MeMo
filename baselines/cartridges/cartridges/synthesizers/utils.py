from typing import List

from transformers import AutoTokenizer

from cartridges.structs import Message, Logprob, Section
from cartridges.clients.base import InputToken, Sample, SelectedToken, TopToken


def split_input_tokens_into_messages(
    tokens: List[InputToken], tokenizer: AutoTokenizer
) -> List[Message]:

    if tokenizer.name_or_path in [
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
    ]:
        # Llama 3.2 models use a special token to indicate the start of a message.
        # We need to extract the message from the input tokens.
        header_start = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        header_end = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        message_end = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    else:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path} is not supported")

    in_message, in_header = False, False

    role_token_ids, message_token_ids = [], []
    role_logprobs: List[List[TopToken]] = []
    message_logprobs: List[List[TopToken]] = []
    messages: List[Message] = []
    current_idx = 0
    while current_idx < len(tokens):
        token = tokens[current_idx]
        if token.id == header_start:
            in_header = True
        elif token.id == header_end:
            in_header, in_message = False, True

            # we want to include the logprobs for the first token in the message
            # there is a shift by one between the input token ids and the logprobs
            message_logprobs.append(token.top_logprobs)
        elif (token.id == message_end) and in_message:
            in_message, in_header = False, False
            message_token_ids.append(token.id)
            messages.append(
                Message(
                    role=tokenizer.decode(role_token_ids),
                    content=tokenizer.decode(message_token_ids),
                    token_ids=message_token_ids,
                    logprobs=[
                        [
                            Logprob(token_id=top_token.id, logprob=top_token.logprob)
                            for top_token in top_tokens
                        ]
                        for top_tokens in message_logprobs
                    ],
                    # top_logprobs=message_logprobs,
                )
            )
            role_token_ids, message_token_ids = [], []
            message_logprobs: List[List[TopToken]] = []
        elif in_header:
            role_token_ids.append(token.id)
        elif in_message:
            message_token_ids.append(token.id)
            message_logprobs.append(token.top_logprobs)
        current_idx += 1
    return messages


def convert_sample_into_message(
    sample: Sample,
) -> Message:

    return Message(
        role="assistant",
        content=sample.text,
        token_ids=[selected.id for selected in sample.tokens],
        logprobs=[
            [
                Logprob(token_id=token.id, logprob=token.logprob)
                for token in selected.top_logprobs
            ]
            for selected in sample.tokens
        ],
    )


BUFFER_TOKENS = 20


def unopionated_section_maker(
    lines: list[str],
    title: str,
    max_tokens_per_section: int,
    tokenizer,
) -> list[Section]:
    count_tokens = lambda s: len(tokenizer.encode(s))
    tokens_per_line = [count_tokens(line) for line in lines]

    section_content: list[str] = []
    current_section: list[str] = []
    current_section_tokens = 0

    def add_section():
        assert current_section

        section_content.append("\n".join(current_section))

        current_section.clear()
        nonlocal current_section_tokens
        current_section_tokens = 0

    for line, tokens in zip(lines, tokens_per_line):
        assert tokens + BUFFER_TOKENS < max_tokens_per_section

        if current_section_tokens + tokens + BUFFER_TOKENS > max_tokens_per_section:
            add_section()

        current_section.append(line)
        current_section_tokens += tokens

    if current_section:
        add_section()

    sections = [
        Section(
            desc=f"{title}, Section {i} / {len(section_content)}",
            content=content,
            tokens=count_tokens(content),
        )
        for i, content in enumerate(section_content, start=1)
    ]

    for section in sections:
        assert section.tokens < max_tokens_per_section

    return sections
