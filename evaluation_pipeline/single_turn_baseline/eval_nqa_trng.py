import argparse
from openai import AsyncOpenAI
import json
import os
import asyncio

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data_synthesis_pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from nqa_data_utils import load_questions_with_evidence_docs_nqa
from general_eval_prompt_utils import generate_prompt_from_LM_to_SM_general_v1, format_LM_final_answer
from model_utils import prepare_prompt_for_checking_mem_knowledge


def load_questions(qns_path, max_num_docs=None):
    questions = load_questions_with_evidence_docs_nqa(qns_path, max_num_docs)
    print(f"Loaded questions count: {len(questions)}")
    return questions


async def generate_batched_responses_async(sm_client, sub_questions, prompt_formatter_func, sm_model_id, max_new_tokens=512, temperature=0.7, max_concurrent=10):
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
                return {
                    'question': sub_q,
                    'answer': response.choices[0].message.content
                }
            except Exception as e:
                print(f"Error processing sub-question: {e}")
                return {
                    'question': sub_q,
                    'answer': "I don't know"
                }

    tasks = [process_sub_question(sub_q) for sub_q in sub_questions]
    return await asyncio.gather(*tasks)


async def process_single_question(lm_client, sm_client, row, question_idx, lm_model_id, sm_model_id, max_new_tokens, thinking_budget, stream):
    """Process a single NQA question asynchronously."""
    question = row['question']
    print(f"Processing Question No: {question_idx}")

    formatted_question = generate_prompt_from_LM_to_SM_general_v1(question)

    try:
        response = await lm_client.chat.completions.create(
            model=lm_model_id,
            stream=stream,
            messages=[{"role": "user", "content": formatted_question}],
            max_tokens=max_new_tokens,
            temperature=1.1,
            top_p=0.95,
            extra_body={
                "chat_template_kwargs": {
                    "thinking_budget": thinking_budget
                }
            }
        )

        content = response.choices[0].message.content
        print(f"Question No {question_idx} - Initial response received: {content}")

        try:
            content = json.loads(content)
            if content['requires_memory_bank']:
                sm_response = await generate_batched_responses_async(
                    sm_client=sm_client,
                    sub_questions=content['sub_questions'],
                    prompt_formatter_func=prepare_prompt_for_checking_mem_knowledge,
                    sm_model_id=sm_model_id,
                    max_new_tokens=512,
                    temperature=0.7,
                    max_concurrent=10
                )
                updated_sm_response = [
                    sm_res for sm_res in sm_response
                    if sm_res['answer'] != "I don't know" and "I don't know" not in sm_res["answer"]
                ]
                format_question_for_final_response = format_LM_final_answer(updated_sm_response, question)
                final_response = await lm_client.chat.completions.create(
                    model=lm_model_id,
                    stream=stream,
                    messages=[{"role": "user", "content": format_question_for_final_response}],
                    max_tokens=max_new_tokens,
                    temperature=0.3,
                    top_p=0.95,
                    extra_body={
                        "chat_template_kwargs": {
                            "thinking_budget": thinking_budget
                        }
                    }
                )
                final_response_content = final_response.choices[0].message.content
                print(f"Final response content for question {question_idx}: {final_response_content}")
                final_response_content = json.loads(final_response_content)
                final_answer = final_response_content['final_answer']
                final_justification = final_response_content['justification']
            else:
                print(f"No need for memory module for question no. {question_idx}")
                final_answer = content['final_answer']
                final_justification = "N/A"
                sm_response = []
                updated_sm_response = []

            return {
                "question_no": question_idx,
                "question": question,
                "sub_questions": content['sub_questions'] if content['requires_memory_bank'] else [],
                "sm_model_response": sm_response if content['requires_memory_bank'] else "N/A",
                "updated_sm_model_response": updated_sm_response if content['requires_memory_bank'] else "N/A",
                "groundtruth": row["groundtruth"],
                "model_initial_response": content['final_answer'],
                "model_final_response": final_answer,
                "model_final_response_justification": final_justification,
            }
        except Exception as e:
            print(f"Error parsing response for question no. {question_idx}: {e}")
            return None

    except Exception as e:
        print(f"Error processing question no. {question_idx}: {e}")
        return None


async def process_all_questions(lm_client, sm_client, eval_questions, lm_model_id, sm_model_id, max_new_tokens, thinking_budget, stream, max_concurrent=5):
    """Process all questions with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(row, idx):
        async with semaphore:
            return await process_single_question(
                lm_client, sm_client, row, idx,
                lm_model_id, sm_model_id,
                max_new_tokens, thinking_budget, stream
            )

    tasks = [process_with_semaphore(row, row['question_no']) for row in eval_questions]
    results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--thinking_budget", type=int, default=-1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--lm_port", type=int, default=4322, help="Port for large model vLLM server")
    parser.add_argument("--sm_port", type=int, default=4324, help="Port for small model vLLM server")
    parser.add_argument("--nqa_qns_path", type=str, default="")
    parser.add_argument("--max_num_docs", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--max_concurrent", type=int, default=50, help="Maximum number of concurrent requests")
    args = parser.parse_args()

    lm_client = AsyncOpenAI(base_url=f"http://localhost:{args.lm_port}/v1", api_key="dummy")
    sm_client = AsyncOpenAI(base_url=f"http://localhost:{args.sm_port}/v1", api_key="dummy")

    lm_model_id = (await lm_client.models.list()).data[0].id
    sm_model_id = (await sm_client.models.list()).data[0].id
    print(f"LM model: {lm_model_id}")
    print(f"SM model: {sm_model_id}")

    eval_questions = load_questions(args.nqa_qns_path, args.max_num_docs)

    output = await process_all_questions(
        lm_client, sm_client, eval_questions, lm_model_id, sm_model_id,
        args.max_new_tokens, args.thinking_budget, args.stream,
        max_concurrent=args.max_concurrent
    )

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"Processed {len(output)} questions successfully")

    print("\n" + "="*80)
    print("Starting DeepEval evaluation...")
    print("="*80)

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
        run_id="eval-nqa-single-turn"
    )

    eval_output = []
    for res in eval_results:
        _, results = res
        for test_result in results:
            metric = test_result.metrics_data[0]
            entry = {
                "score": metric.score,
                "input": test_result.input,
                "actual_output": test_result.actual_output,
                "expected_output": test_result.expected_output,
                "reason": metric.reason
            }
            eval_output.append(entry)
        break

    deepeval_output = {
        "summary": {
            "total_evaluated": len(eval_output),
            "total_correct": sum([1 if item.get('score', 0.0) == 1.0 else 0 for item in eval_output]),
            "overall_accuracy": sum([1 if item.get('score', 0.0) == 1.0 else 0 for item in eval_output]) / len(eval_output) if eval_output else 0.0
        },
        "logged_output": eval_output
    }

    deepeval_summary_path = args.output_path.replace(".json", "_deepeval_summary.json")
    with open(deepeval_summary_path, 'w', encoding='utf-8') as h:
        json.dump(deepeval_output, h, indent=2)

    print("\n" + "="*80)
    print("DeepEval evaluation completed!")
    print(f"Overall accuracy: {deepeval_output['summary']['overall_accuracy']:.3f}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
