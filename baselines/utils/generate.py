import json
import re

QA_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the provided context. "
    "Output the answer in JSON format with strictly one string field: 'answer'. "
    "Do not wrap the JSON in markdown code blocks or backticks."
)

CARTRIDGE_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Output the answer in JSON format with strictly one string field: 'answer'. "
    "Do not wrap the JSON in markdown code blocks or backticks."
)


def _qa_messages(question, context_chunks):
    context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
    ]


def _parse_answer(output):
    if not isinstance(output, str):
        return "N/A"
    stripped = output.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
        stripped = stripped.strip()
    try:
        parsed = json.loads(stripped)
        lower = {k.lower(): v for k, v in parsed.items()}
        return str(lower.get("answer", "N/A"))
    except json.JSONDecodeError:
        m = re.search(r'"(?:answer|Answer)"\s*:\s*"([^"]*)"', stripped)
        if m:
            return m.group(1)
        m_alt = re.search(r'answer["\']?\s*[:=]\s*["\']([^"\']*)["\']', stripped, re.IGNORECASE)
        if m_alt:
            return m_alt.group(1)
        # As a last resort, treat the whole stripped output as the answer.
        return stripped if stripped else "N/A"


async def generate_answer_vllm_async(client, model_id, question, context_chunks,
                                     seed=1, temperature=0.7, max_tokens=512, timeout=240,
                                     **_unused):
    """Async QA call to an OpenAI-compatible chat endpoint. Returns answer string."""
    messages = _qa_messages(question, context_chunks)
    response = await client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    output = response.choices[0].message.content
    answer = _parse_answer(output)
    print(f"[Answer] {answer}")
    return answer


def generate_answer_vllm_sync(client, model_id, question, context_chunks,
                              seed=1, temperature=0.7, max_tokens=512, timeout=240,
                              **_unused):
    """Sync QA call to an OpenAI-compatible chat endpoint. Returns answer string."""
    messages = _qa_messages(question, context_chunks)
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    output = response.choices[0].message.content
    answer = _parse_answer(output)
    print(f"[Answer] {answer}")
    return answer
