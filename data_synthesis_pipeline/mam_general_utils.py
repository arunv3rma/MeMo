
def format_final_answer_for_LM(content_from_memory_bank, original_question):
    prompt = f"""You are a large language model with access to a memory bank of academic papers.

    Your task is to synthesize a final answer to the following question:
    "{original_question}"

    You have been provided with the following content from the memory bank, which represents answers to relevant sub-questions:
    {content_from_memory_bank}

    ### Instructions:
    1. Use the provided memory bank content as the primary evidence when forming your answer.
    2. When the memory bank content contains domain-specific keywords or technical terms that are essential to the answer, incorporate them to maintain technical accuracy.
    3. If the memory bank content is insufficient, you may supplement with general knowledge, but clearly separate this from the memory-based information.
    4. Always connect your final answer back to the original question.
    5. Provide a justification that explains how the memory bank content supports your answer.

    ### Output Format (strict JSON):
    {{
        "final_answer": "<your best possible synthesized answer to the original question>",
        "justification": "<explanation of how the memory bank content was used to form the answer>"
    }}
    """
    return prompt

