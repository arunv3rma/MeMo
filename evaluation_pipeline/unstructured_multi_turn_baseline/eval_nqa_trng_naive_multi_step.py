"""
Naive multi-turn extension of eval_nqa_trng.py (single-turn).

Round 0: identical to single-turn — LM generates sub-questions, SM answers.
Loop 1..N: LM reviews full Q&A history, summarises what is known, identifies
           what is still missing, and generates the next round of sub-questions.
           All state tracking is implicit — delegated entirely to the LM's
           in-context reasoning over the raw history.
Final: same format_LM_final_answer synthesis as single-turn.
"""
import argparse
from openai import AsyncOpenAI
import json
import re
import os
import asyncio

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data_synthesis_pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from nqa_data_utils import load_questions_with_evidence_docs_nqa
from general_eval_prompt_utils import generate_prompt_from_LM_to_SM_general_v1, format_LM_final_answer

from model_utils import (
    prepare_prompt_for_checking_mem_knowledge,
)


def generate_naive_loop_prompt(question: str, conversation_history: list) -> str:
    """
    Prompt for the LM to review all prior Q&A, summarise what is known,
    identify what is still missing, and generate the next round of sub-questions.
    Returns JSON: { summary_of_known, still_unknown, requires_more_info, sub_questions, final_answer }
    """
    history_lines = []
    for entry in conversation_history:
        history_lines.append(f"--- Round {entry['round']} ---")
        for qa in entry['sm_responses']:
            history_lines.append(f"  Q: {qa['question']}")
            history_lines.append(f"  A: {qa['answer']}")
        history_lines.append("")
    history_str = "\n".join(history_lines).strip()

    return f"""You are helping answer a complex question by querying a memory bank across multiple rounds.

Original Question:
{question}

---

Retrieved Information So Far (all rounds):
{history_str}

---

Given the information retrieved above:

1. Summarise what you now know that is relevant to answering the original question.
2. Identify what information is still missing or unclear that would help answer the question.
3. Decide whether you need another round of retrieval, or whether you have enough to answer.

If you need more information, generate the next round of sub-questions to fill the gaps.

Return your response as a JSON object in the following format:
{{
    "summary_of_known": "a concise summary of what has been established from the retrieved information so far",
    "still_unknown": "what information is still missing that would help answer the original question",
    "requires_more_info": true or false,
    "sub_questions": ["question 1", "question 2", ...],
    "final_answer": "your best answer to the original question if requires_more_info is false, otherwise null"
}}

Rules:
- sub_questions must only be present and non-empty when requires_more_info is true
- If requires_more_info is false, provide your best final_answer based on all retrieved information
- Each sub-question must be fully self-contained — no pronouns or vague references
- Return only the JSON object, no other text"""


def extract_json(content: str) -> dict:
    """Extract JSON from LM response, handling markdown code fences."""
    if not content:
        raise ValueError("Empty response content")
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped.strip())


def load_questions(qns_path, max_num_docs=None):
    questions = load_questions_with_evidence_docs_nqa(qns_path, max_num_docs)
    print(f"Loaded questions count: {len(questions)}")
    return questions


async def generate_batched_responses_async(
    sm_client,
    sub_questions,
    prompt_formatter_func,
    sm_model_id,
    max_new_tokens=512,
    temperature=0.7,
    max_concurrent=10
):
    """Generate responses for sub-questions using vLLM asynchronously."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_sub_question(sub_q):
        async with semaphore:
            prompt = prompt_formatter_func(sub_q)
            try:
                response = await sm_client.chat.completions.create(
                    model=sm_model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                )
                return {'question': sub_q, 'answer': response.choices[0].message.content}
            except Exception as e:
                print(f"Error processing sub-question: {e}")
                return {'question': sub_q, 'answer': "I don't know"}

    tasks = [process_sub_question(sub_q) for sub_q in sub_questions]
    return await asyncio.gather(*tasks)


async def process_single_question(
    lm_client,
    sm_client,
    row,
    question_idx,
    max_new_tokens,
    thinking_budget,
    stream,
    lm_model_id=None,
    lm_temperature=1.1,
    sm_temperature=0.7,
    loop_temperature=1.0,
    final_temperature=0.3,
    max_turns=5,
):
    question = row['question']
    print(f"Processing Question No: {question_idx}")

    model_id = lm_model_id or (await lm_client.models.list()).data[0].id
    sm_model_id = (await sm_client.models.list()).data[0].id

    conversation_history = []
    all_sm_responses_flat = []

    # -------------------------------------------------------------------------
    # ROUND 0: identical to single-turn
    # -------------------------------------------------------------------------
    formatted_question = generate_prompt_from_LM_to_SM_general_v1(question)

    try:
        response = await lm_client.chat.completions.create(
            model=model_id,
            stream=stream,
            messages=[{"role": "user", "content": formatted_question}],
            max_tokens=max_new_tokens,
            temperature=lm_temperature,
            top_p=0.95,
            extra_body={"chat_template_kwargs": {"thinking_budget": thinking_budget}}
        )
        content = extract_json(response.choices[0].message.content)
        print(f"Question {question_idx} - Round 0 response: {content}")

        if not content.get('requires_memory_bank', False):
            print(f"Question {question_idx} - no memory bank needed, returning direct answer")
            return {
                "question_no": question_idx,
                "question": question,
                "conversation_history": [],
                "loop_history": [],
                "all_sm_responses": [],
                "all_sm_responses_filtered": [],
                "num_turns": 0,
                "groundtruth": row["groundtruth"],
                "model_final_response": content.get('final_answer', ''),
                "model_final_response_justification": "N/A",
            }

        sub_questions_r0 = content['sub_questions']
        print(f"Question {question_idx} - Round 0 sub-questions: {sub_questions_r0}")

        sm_responses_r0 = await generate_batched_responses_async(
            sm_client=sm_client,
            sub_questions=sub_questions_r0,
            prompt_formatter_func=prepare_prompt_for_checking_mem_knowledge,
            sm_model_id=sm_model_id,
            max_new_tokens=512,
            temperature=sm_temperature,
            max_concurrent=10
        )
        print(f"Question {question_idx} - Round 0 SM responses: {sm_responses_r0}")

        conversation_history.append({'round': 0, 'sm_responses': sm_responses_r0})
        all_sm_responses_flat.extend(sm_responses_r0)

    except Exception as e:
        print(f"Error in Round 0 for question {question_idx}: {e}")
        return None

    # -------------------------------------------------------------------------
    # LOOP: LM reviews history, summarises known, generates next questions
    # -------------------------------------------------------------------------
    loop_history = []

    for turn in range(max_turns):
        loop_prompt = generate_naive_loop_prompt(question, conversation_history)

        try:
            loop_response = await lm_client.chat.completions.create(
                model=model_id,
                stream=stream,
                messages=[{"role": "user", "content": loop_prompt}],
                max_tokens=max_new_tokens,
                temperature=loop_temperature,
                top_p=0.95,
                extra_body={"chat_template_kwargs": {"thinking_budget": thinking_budget}}
            )
            loop_content = extract_json(loop_response.choices[0].message.content)

            summary_of_known = loop_content.get('summary_of_known', '')
            still_unknown = loop_content.get('still_unknown', '')
            requires_more_info = loop_content.get('requires_more_info', False)

            print(f"Question {question_idx} - Turn {turn + 1}: requires_more_info={requires_more_info}")
            print(f"Question {question_idx} - Turn {turn + 1}: summary_of_known={summary_of_known}")
            print(f"Question {question_idx} - Turn {turn + 1}: still_unknown={still_unknown}")

            loop_round_meta = {
                'turn': turn + 1,
                'summary_of_known': summary_of_known,
                'still_unknown': still_unknown,
                'requires_more_info': requires_more_info,
            }

            if not requires_more_info:
                loop_history.append(loop_round_meta)
                print(f"Question {question_idx} - Turn {turn + 1}: LM has enough information")
                break

            if turn == max_turns - 1:
                print(f"Question {question_idx} - max_turns reached, forcing final answer")
                loop_round_meta['requires_more_info'] = False
                loop_history.append(loop_round_meta)
                break

            sub_questions = loop_content.get('sub_questions', [])
            if not sub_questions:
                print(f"Question {question_idx} - Turn {turn + 1}: no sub-questions generated, forcing final answer")
                loop_round_meta['requires_more_info'] = False
                loop_history.append(loop_round_meta)
                break

            print(f"Question {question_idx} - Turn {turn + 1}: asking {len(sub_questions)} sub-questions: {sub_questions}")

            sm_responses = await generate_batched_responses_async(
                sm_client=sm_client,
                sub_questions=sub_questions,
                prompt_formatter_func=prepare_prompt_for_checking_mem_knowledge,
                sm_model_id=sm_model_id,
                max_new_tokens=512,
                temperature=sm_temperature,
                max_concurrent=10
            )
            print(f"Question {question_idx} - Turn {turn + 1} SM responses: {sm_responses}")

            conversation_history.append({'round': turn + 1, 'sm_responses': sm_responses})
            all_sm_responses_flat.extend(sm_responses)

            loop_round_meta['sub_questions'] = sub_questions
            loop_round_meta['sm_responses'] = sm_responses
            loop_history.append(loop_round_meta)

        except Exception as e:
            print(f"Error in loop turn {turn + 1} for question {question_idx}: {e}")
            break

    # -------------------------------------------------------------------------
    # FINAL: synthesize answer from all collected SM responses
    # -------------------------------------------------------------------------
    try:
        all_sm_responses_filtered = [
            r for r in all_sm_responses_flat
            if "i don't know" not in r['answer'].lower()
        ]
        final_prompt = format_LM_final_answer(all_sm_responses_filtered, question)

        final_response = await lm_client.chat.completions.create(
            model=model_id,
            stream=stream,
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=max_new_tokens,
            temperature=final_temperature,
            top_p=0.95,
            extra_body={"chat_template_kwargs": {"thinking_budget": thinking_budget}}
        )
        final_content = extract_json(final_response.choices[0].message.content)
        final_answer = final_content['final_answer']
        final_justification = final_content['justification']
        print(f"Question {question_idx} - Final answer: {final_answer}")

    except Exception as e:
        print(f"Error in final answer generation for question {question_idx}: {e}")
        return None

    return {
        "question_no": question_idx,
        "question": question,
        "conversation_history": conversation_history,
        "loop_history": loop_history,
        "all_sm_responses": all_sm_responses_flat,
        "all_sm_responses_filtered": all_sm_responses_filtered,
        "num_turns": len(loop_history),
        "groundtruth": row["groundtruth"],
        "model_final_response": final_answer,
        "model_final_response_justification": final_justification,
    }


async def process_all_questions(
    lm_client,
    sm_client,
    eval_questions,
    max_new_tokens,
    thinking_budget,
    stream,
    lm_model_id=None,
    max_concurrent=5,
    lm_temperature=1.1,
    sm_temperature=0.7,
    loop_temperature=1.0,
    final_temperature=0.3,
    max_turns=5,
):
    """Process all questions with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(row, idx):
        async with semaphore:
            return await process_single_question(
                lm_client, sm_client, row, idx,
                max_new_tokens, thinking_budget, stream,
                lm_model_id=lm_model_id,
                lm_temperature=lm_temperature,
                sm_temperature=sm_temperature,
                loop_temperature=loop_temperature,
                final_temperature=final_temperature,
                max_turns=max_turns,
            )

    tasks = [process_with_semaphore(row, row['question_no']) for row in eval_questions]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--thinking_budget", type=int, default=-1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--lm_port", type=int, default=4322)
    parser.add_argument("--sm_port", type=int, default=4324)
    parser.add_argument("--nqa_qns_path", type=str, default="")
    parser.add_argument("--max_num_docs", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--max_concurrent", type=int, default=50)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--lm_temperature", type=float, default=1.1)
    parser.add_argument("--sm_temperature", type=float, default=0.7)
    parser.add_argument("--loop_temperature", type=float, default=1.0)
    parser.add_argument("--final_temperature", type=float, default=0.3)
    args = parser.parse_args()

    lm_client = AsyncOpenAI(base_url=f"http://localhost:{args.lm_port}/v1", api_key="dummy", timeout=60.0)
    sm_client = AsyncOpenAI(base_url=f"http://localhost:{args.sm_port}/v1", api_key="dummy", timeout=60.0)

    eval_questions = load_questions(args.nqa_qns_path, args.max_num_docs)

    output = await process_all_questions(
        lm_client, sm_client, eval_questions,
        args.max_new_tokens, args.thinking_budget, args.stream,
        max_concurrent=args.max_concurrent,
        lm_temperature=args.lm_temperature,
        sm_temperature=args.sm_temperature,
        loop_temperature=args.loop_temperature,
        final_temperature=args.final_temperature,
        max_turns=args.max_turns,
    )

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"Processed {len(output)} questions successfully")

    print("\n" + "=" * 80)
    print("Starting DeepEval evaluation...")
    print("=" * 80)

    from dotenv import load_dotenv
    load_dotenv()
    judge_client = AsyncOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY", "dummy")
    )
    judge_model_name = os.getenv("OPENROUTER_MODEL_NAME", "gpt-4")

    from deepeval_utils import run_evaluation_multi_answer
    data_pairs = [
        {
            'question': item['question'],
            'ans1': item['model_final_response'],
            'ans2': item['groundtruth']
        }
        for item in output
    ]

    eval_results = run_evaluation_multi_answer(
        client=judge_client,
        client_model_name=judge_model_name,
        data_pairs=data_pairs,
        run_id="eval-nqa-naive-multi-turn"
    )

    print("\n" + "=" * 80)
    print("DeepEval evaluation completed!")
    eval_output = []
    for res in eval_results:
        _, results = res
        for test_result in results:
            metric = test_result.metrics_data[0]
            eval_output.append({
                "score": metric.score,
                "input": test_result.input,
                "actual_output": test_result.actual_output,
                "expected_output": test_result.expected_output,
                "reason": metric.reason
            })
        break

    summary_output = {
        "summary": {
            "total_evaluated": len(eval_output),
            "total_correct": sum(1 for item in eval_output if item.get('score', 0.0) == 1.0),
            "overall_accuracy": sum(1 for item in eval_output if item.get('score', 0.0) == 1.0) / len(eval_output) if eval_output else 0.0
        },
        "logged_output": eval_output
    }

    deepeval_summary_path = args.output_path.replace(".json", "deepeval_summary.json")
    with open(deepeval_summary_path, 'w', encoding='utf-8') as h:
        json.dump(summary_output, h, indent=2)
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
