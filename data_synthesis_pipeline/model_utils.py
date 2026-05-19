from mam_general_utils import format_final_answer_for_LM


### PROMPTS


# for checking mem knowledge
def prepare_prompt_for_checking_mem_knowledge(relf_qn):
    prompt = f"""Answer this question: {relf_qn}

    IMPORTANT: Only provide an answer if you are highly confident. If you have any doubt, it's better to say "I don't know" than to guess incorrectly.

    Return Strict JSON:
    {{
        "answer": "<your answer or 'I don't know'>",
        "reasoning": "<your thought process>"
    }}
    """
    return prompt



def format_LM_final_answer(content_from_memory_bank, original_question):
    return format_final_answer_for_LM(content_from_memory_bank, original_question)



