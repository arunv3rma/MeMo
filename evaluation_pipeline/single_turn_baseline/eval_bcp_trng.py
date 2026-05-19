import argparse
from openai import OpenAI, AsyncOpenAI
import pandas as pd
import json
import ast
import re
import os
import asyncio
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data_synthesis_pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from bcp_data_utils import SUBSET_MAP, load_questions_with_evidence_docs
from general_eval_prompt_utils import generate_prompt_from_LM_to_SM_general_v1

from model_utils import (
    format_LM_final_answer,
    prepare_prompt_for_checking_mem_knowledge,
)
from deepeval_utils import run_evaluation

def load_questions(qns_path, max_valid_questions=None):
    
    all_qns = load_questions_with_evidence_docs(qns_path, max_valid_questions)
    if max_valid_questions is not None:
        subset_query_ids = SUBSET_MAP[max_valid_questions]
    
    filtered_qns = [qn for qn in all_qns if qn['question_no'] in subset_query_ids]
    print(f"Filtered questions count: {len(filtered_qns)}")
    return filtered_qns

async def generate_batched_responses_async(sm_client, sub_questions, prompt_formatter_func, max_new_tokens=512, temperature=0.7, max_concurrent=10):
    """Generate responses for sub-questions using vLLM asynchronously"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_sub_question(sub_q):
        async with semaphore:
            prompt = prompt_formatter_func(sub_q)
            try:
                response = await sm_client.chat.completions.create(
                    model=(await sm_client.models.list()).data[0].id,
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
    sm_responses = await asyncio.gather(*tasks)
    
    return sm_responses, []


async def process_single_question(lm_client, sm_client, row, question_idx, max_new_tokens, thinking_budget, stream):
    """Process a single question asynchronously"""
    question = row['question']
    print(f"Processing Question No: {question_idx}")

    formatted_question = generate_prompt_from_LM_to_SM_general_v1(question)
    
    try:
        response = await lm_client.chat.completions.create(
            model=(await lm_client.models.list()).data[0].id,
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
                sm_response, failed_questions = await generate_batched_responses_async(
                    sm_client=sm_client,
                    sub_questions=content['sub_questions'], 
                    prompt_formatter_func=prepare_prompt_for_checking_mem_knowledge, 
                    max_new_tokens=512,
                    temperature=0.7,
                    max_concurrent=10
                )
                # print('db1')
                # Filter out "I don't know" responses
                updated_sm_response = []
                for sm_res in sm_response:
                    if sm_res['answer'] == "I don't know" or "I don't know" in sm_res["answer"]:
                        continue
                    updated_sm_response.append(sm_res)
                # print('db2')
                format_question_for_final_response = format_LM_final_answer(updated_sm_response, question)
                final_response = await lm_client.chat.completions.create(
                    model=(await lm_client.models.list()).data[0].id,
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
                # print('db3')
                final_response_content = final_response.choices[0].message.content
                print(f"Final response content for question {question_idx}: {final_response_content}")
                final_response_content = json.loads(final_response_content)
                final_response = final_response_content['final_answer']
                final_response_justification = final_response_content['justification']
                final_response_content = final_response
                # print('db4')
            else:
                print(f"No need for memory module for question no. {question_idx}")
                final_response_content = content['final_answer']
                final_response_justification = "N/A"
            
            return {
                "question_no": question_idx,
                "question": question,
                "sub_questions": content['sub_questions'] if content['requires_memory_bank'] else [],
                "sm_model_respone": sm_response if content['requires_memory_bank'] else "N/A",
                "updated_sm_model_response" : updated_sm_response if content['requires_memory_bank'] else "N/A",
                "groundtruth": row["groundtruth"],
                "model_initial_repsonse": content['final_answer'],
                "model_final_response": final_response_content,
                "model_final_response_justification": final_response_justification,
            }
        except Exception as e:
            print(f"Error parsing response for question no. {question_idx}: {e}")
            print(type)
            return None
            
    except Exception as e:
        print(f"Error processing question no. {question_idx}: {e}")
        return None


async def process_all_questions(lm_client, sm_client, eval_questions, max_new_tokens, thinking_budget, stream, max_concurrent=5):
    """Process all questions with controlled concurrency"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(row, idx):
        async with semaphore:
            return await process_single_question(lm_client, sm_client, row, idx, max_new_tokens, thinking_budget, stream)
    
    tasks = [process_with_semaphore(row, row['question_no']) for row in eval_questions]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results (failed questions)
    return [r for r in results if r is not None]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--thinking_budget", type=int, default=-1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--lm_port", type=int, default=4322, help="Port for large model vLLM server")
    parser.add_argument("--sm_port", type=int, default=4324, help="Port for small model vLLM server")
    parser.add_argument("--bcp_qns_path", type=str, default="")
    parser.add_argument("--max_num_questions", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="/path/to/eval_output.json")
    parser.add_argument("--max_concurrent", type=int, default=50, help="Maximum number of concurrent requests")
    args = parser.parse_args()

    max_new_tokens = args.max_new_tokens
    thinking_budget = args.thinking_budget
    stream = args.stream

    # Create AsyncOpenAI clients for both models
    lm_client = AsyncOpenAI(base_url=f"http://localhost:{args.lm_port}/v1", api_key="dummy")
    sm_client = AsyncOpenAI(base_url=f"http://localhost:{args.sm_port}/v1", api_key="dummy")

    # Load questions
    eval_questions = load_questions(args.bcp_qns_path, args.max_num_questions)
    # eval_questions = eval_questions[:1]
    
    # Process all questions asynchronously
    output = await process_all_questions(
        lm_client, sm_client, eval_questions, 
        max_new_tokens, thinking_budget, stream,
        max_concurrent=args.max_concurrent
    )
    
    # Save results
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    
    print(f"Processed {len(output)} questions successfully")
    
    # Run evaluation using deepeval
    print("\n" + "="*80)
    print("Starting DeepEval evaluation...")
    print("="*80)
    
    # Set up judge client
    from dotenv import load_dotenv
    load_dotenv()
    judge_client = AsyncOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY", "dummy")
    )
    judge_model_name = os.getenv("OPENROUTER_MODEL_NAME", "gpt-4")
    
    # Prepare data pairs for evaluation
    data_pairs = []
    for item in output:
        data_pairs.append({
            'question': item['question'],
            'ans1': item['model_final_response'],
            'ans2': item['groundtruth']
        })
    
    # Run evaluation
    eval_results = run_evaluation(
        client=judge_client,
        client_model_name=judge_model_name,
        data_pairs=data_pairs,
        run_id="eval-single-turn"
    )
    
    print("\n" + "="*80)
    print("DeepEval evaluation completed!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())