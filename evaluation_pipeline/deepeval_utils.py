import asyncio
import json
from enum import Enum
from typing import List, Optional, Any
from openai import OpenAI, AsyncOpenAI

import os
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS"] = "100000"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

# --- DeepEval Imports ---
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM
from deepeval.evaluate import AsyncConfig
from deepeval.evaluate import CacheConfig

# --- Enum for Prompt Settings (Assumed from your context) ---
class PromptSetting(Enum):
    SemanticSimilarity = "SemanticSimilarity"
    # Add other settings if needed

# ==================================================================================
# 1. Your Original Hedging Logic (Refined for Integration)
# ==================================================================================
import re

def parse_content_string(content: str, prompt_setting: PromptSetting) -> str:
    """
    Parses the raw content to ensure it matches the expected format.
    Falls back to regex extraction if JSON parsing fails.
    """
    print(f"RAW MODEL OUTPUT:\n{content}\n---")
    if not content or not content.strip():
        print("⚠ Model returned empty content")
        return json.dumps({"score": 0, "reason": "Model returned empty response"})
    try:
        # Clean up markdown code blocks
        cleaned_content = content.strip()
        
        # Remove opening code block markers
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:].strip()
        elif cleaned_content.startswith("```"):
            cleaned_content = cleaned_content[3:].strip()
        
        # Remove closing code block markers
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3].strip()
        
        # Fix invalid backslash escapes (LaTeX like \mathcal, \pm, etc.)
        # Replace single backslashes with double backslashes, except for valid JSON escapes
        cleaned_content = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', cleaned_content)
        
        # Validate it's valid JSON
        parsed = json.loads(cleaned_content)
        
        # Verify it has the expected fields
        if 'score' in parsed:
            return cleaned_content
        else:
            raise ValueError("JSON missing 'score' field")

    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing failed: {e}")
        print(f"Attempting regex extraction as fallback...")
        
        # Fallback: Extract score and reason using regex
        try:
            # Look for score: 0 or score: 1 (with or without quotes)
            score_match = re.search(r'"?score"?\s*:\s*(\d+)', content, re.IGNORECASE)
            if score_match:
                score = int(score_match.group(1))
                # Validate it's 0 or 1
                if score not in [0, 1]:
                    print(f"⚠ Invalid score {score}, defaulting to 0")
                    score = 0
            else:
                print(f"⚠ Could not find score, defaulting to 0")
                score = 0
            
            # Look for reason (everything between "reason": and the next field or end)
            reason_match = re.search(r'"?reason"?\s*:\s*"([^"]*)"', content, re.IGNORECASE | re.DOTALL)
            if reason_match:
                reason = reason_match.group(1)
                # Clean up the reason (remove extra backslashes, newlines, etc.)
                reason = reason.replace('\\n', ' ').replace('\n', ' ').strip()
            else:
                # Try without quotes (sometimes models return unquoted strings)
                reason_match = re.search(r'"?reason"?\s*:\s*([^,}\n]+)', content, re.IGNORECASE)
                if reason_match:
                    reason = reason_match.group(1).strip()
                else:
                    reason = "Could not extract reason from output"
            
            # Construct valid JSON
            fallback_json = json.dumps({
                "score": score,
                "reason": reason[:500]  # Limit reason length to avoid issues
            })
            
            print(f"✓ Regex extraction successful: score={score}")
            return fallback_json
            
        except Exception as fallback_error:
            print(f"⚠ Regex extraction also failed: {fallback_error}")
            print(f"Full content: {content[:500]}...")
            
            # Last resort: return a valid JSON with score 0
            return json.dumps({
                "score": 0,
                "reason": "Failed to parse model output - both JSON and regex extraction failed"
            })


async def query_large_model_async(
    prompt_content, 
    use_openai, 
    client, 
    client_model_name, 
    preferred_error_message, 
    max_new_tokens, 
    prompt_setting, 
    stream, 
    thinking_budget, 
    temperature=0.0, # Lower temperature is usually better for Judges
    top_p=0.95
):
    try:
        if use_openai:
            # NO timeout wrapper here - let outer timeout handle it
            response = await client.chat.completions.create(
                model=client_model_name,
                messages=[{"role": "user", "content": prompt_content}],
                max_completion_tokens=max_new_tokens,
                # temperature=temperature,
                top_p=top_p
            )
        else:
            # Prepare kwargs for vLLM / Custom models
            create_kwargs = {
                "model": client_model_name,
                "stream": stream,
                "messages": [{"role": "user", "content": prompt_content}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                # for glm_air
                "extra_body" : {"chat_template_kwargs": {"enable_thinking": False}}
            }
            
            # NO timeout wrapper here - let outer timeout handle it
            response = await client.chat.completions.create(**create_kwargs)
        
        content = response.choices[0].message.content
        # Validate output format (ensure it's valid JSON)
        valid_content_str = parse_content_string(content, prompt_setting)
        return valid_content_str
    
    except asyncio.CancelledError:
        # print("Task was cancelled (another task won).") # Optional logging
        return None
    except ValueError as e:
        print(f"Parsing failed for one request: {e}")
        return None
    except Exception as e:
        print(f"{preferred_error_message}: {e}")
        return None

async def query_wrapper(event, result_queue, task_id, *args, **kwargs):
    """Wrapper that puts successful result into queue and triggers event."""
    result = await query_large_model_async(*args, **kwargs)
    
    if result is not None and not event.is_set():
        # print(f"✓ Task {task_id} succeeded and won the race!") # Optional logging
        event.set()
        await result_queue.put(result)
    # else:
        # print(f"✗ Task {task_id} failed or was too slow") # Optional logging

async def query_with_hedging(num_requests, request_id, *args, **kwargs):
    """Launches N requests and returns the first successful one."""
    timeout = 45  # ONLY timeout - increased to 60s to give requests more time
    event = asyncio.Event()
    result_queue = asyncio.Queue(1)
    tasks = []
    
    start_time = asyncio.get_event_loop().time()

    for hedge_num in range(num_requests):
        task_id = f"{request_id}-Hedge-{hedge_num+1}"
        task = asyncio.create_task(
            query_wrapper(event, result_queue, task_id, *args, **kwargs)
        )
        tasks.append((task, task_id))
    
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        successful_result = await result_queue.get()
        elapsed = asyncio.get_event_loop().time() - start_time
        # print(f"✓ {request_id} succeeded in {elapsed:.1f}s")  # Optional success logging
        return successful_result

    except asyncio.TimeoutError:
        elapsed = asyncio.get_event_loop().time() - start_time
        print(f"⚠ {request_id} timed out after {elapsed:.1f}s - no hedge succeeded")
        # Return a fallback JSON so DeepEval doesn't crash, or raise error
        return json.dumps({
            "score": 0, 
            "reason": "Timeout - No valid response received from Model within limit."
        })
    
    except Exception as e:
        elapsed = asyncio.get_event_loop().time() - start_time
        print(f"⚠ Unexpected error in {request_id} after {elapsed:.1f}s: {e}")
        raise
    
    finally:
        for task, t_id in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*[t for t, _ in tasks], return_exceptions=True)


# ==================================================================================
# 2. DeepEval Integration (Custom Judge Class)
# ==================================================================================

class HedgingVLLMJudge(DeepEvalBaseLLM):
    """
    Adapts your specific vLLM client + Hedging logic to be used by DeepEval.
    """
    def __init__(self, client, model_name, num_hedges=3):
        self.client = client
        self.model_name = model_name
        self.num_hedges = num_hedges

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        # DeepEval might call sync generate in some edge cases, 
        # but usually a_generate is used for async evaluation.
        return "Synchronous generation not implemented. Use async."

    async def a_generate(self, prompt: str) -> str:
        """
        DeepEval calls this to get the evaluation result.
        We route this call through your 'query_with_hedging' logic.
        """
        result = await query_with_hedging(
            num_requests=self.num_hedges,
            request_id="DeepEval-Eval",
            # Args passed to query_large_model_async:
            prompt_content=prompt,
            use_openai=False, # Set based on your actual client
            client=self.client,
            client_model_name=self.model_name,
            preferred_error_message="Error in Judge VLLM",
            max_new_tokens=512,
            prompt_setting=PromptSetting.SemanticSimilarity,
            stream=False,
            thinking_budget=-1 # or 0, depending on your model
        )
        return result

    def get_model_name(self):
        return self.model_name


# ==================================================================================
# 3. Main Execution Block
# ==================================================================================

# NOTE: This block assumes 'client' and 'client_model_name' are defined in your scope
# e.g., client = OpenAI(base_url="...", api_key="...")

def run_evaluation(client, client_model_name, data_pairs, run_id=None):
    
    # 1. Initialize Custom Judge
    vllm_judge = HedgingVLLMJudge(
        client=client, 
        model_name=client_model_name, 
        num_hedges=1  # 3 parallel hedges
    )

    # 2. Define the Metric
    # DeepEval will construct the prompt asking for JSON output.
    # Your 'parse_content_string' ensures that JSON is valid before returning it.
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_steps=[
            # Step 1: Extended/fuller forms of the expected answer are always acceptable
            "If the actual output provides a fuller or more complete form of the expected answer, treat this as correct. This includes cases where the actual output contains the expected answer as a recognisable subset, such as a full formal name encompassing a familiar short name, a location expanded with additional geographic context, a title extended with a subtitle or clarifying phrase, a bare value embedded within a descriptive sentence, or an alternate or translated form appended after a separator — provided the core answer remains identifiable and unambiguous.",

            # Step 2: Surface differences to ignore
            "Ignore differences in capitalisation, punctuation within titles (colons, periods, pipes, hyphens), surrounding quotation marks, definite articles, or minor grammatical words that are not part of the core answer.",

            # Step 3: Core containment check
            "Check that the core answer from the expected output is clearly present in or unambiguously implied by the actual output. If the expected output contains critical qualifiers or specifics such as a rank, category, year, or suffix, verify these are also present in the actual output; if missing, assign a score of 0.",
            
            # Step 4: Contradiction check
            "If the actual output directly contradicts the expected output, assign a score of 0. Extra context that does not contradict the expected answer is acceptable.",

            # Step 5: Scoring
            "Assign a score of 1 if the actual output correctly answers the question consistent with the expected output, even if worded more fully or with additional context. Assign a score of 0 only if there is a clear factual mismatch, a direct contradiction, or a missing essential qualifier."
            
            # Step 6: Specific cases
            "If the actual output is formatted differently, e.g. answered as a full sentence instead of just short phrases, it can also be considered similarly"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT,LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=vllm_judge,
        strict_mode=True
    )

    # 3. Create Test Cases
    test_cases = []
    for item in data_pairs:
        # Assuming item is dict like {'question': '...', 'ans1': '...', 'ans2': '...'}
        # If your data is just tuples of (ans1, ans2), adjust accordingly.
        test_case = LLMTestCase(
            input=item.get('question', "No input context provided"), 
            actual_output=str(item['ans1']),
            expected_output=str(item['ans2'])
        )
        test_cases.append(test_case)

    # 4. Run Evaluation
    # REDUCED max_concurrent to avoid overwhelming your server
    # With num_hedges=3 and max_concurrent=10, you'll have ~30 parallel requests max
    results = evaluate(
        test_cases=test_cases,
        metrics=[correctness_metric],
        async_config=AsyncConfig(
            max_concurrent=30,
        ),
        cache_config=CacheConfig(write_cache=False),
        identifier=run_id  
    )

    return results


def run_evaluation_multi_answer(client, client_model_name, data_pairs, run_id=None):
    """
    Like run_evaluation but accepts ans2 as a list of reference answers.
    Scores as correct if the model output matches any of the provided answers.
    """
    vllm_judge = HedgingVLLMJudge(
        client=client,
        model_name=client_model_name,
        num_hedges=1
    )

    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_steps=[
            "If the actual output provides a fuller or more complete form of any reference answer, treat this as correct. This includes cases where the actual output contains a reference answer as a recognisable subset, such as a full formal name encompassing a familiar short name, a location expanded with additional geographic context, a title extended with a subtitle or clarifying phrase, a bare value embedded within a descriptive sentence, or an alternate or translated form appended after a separator — provided the core answer remains identifiable and unambiguous.",

            "Ignore differences in capitalisation, punctuation within titles (colons, periods, pipes, hyphens), surrounding quotation marks, definite articles, or minor grammatical words that are not part of the core answer.",

            "The expected output contains multiple reference answers separated by newlines, each prefixed with 'Answer N:'. Treat the actual output as correct if it clearly matches ANY ONE of the provided reference answers.",

            "Check that the core answer from at least one reference answer is clearly present in or unambiguously implied by the actual output. If a reference answer contains critical qualifiers or specifics such as a rank, category, year, or suffix, verify these are also present in the actual output; if missing for all reference answers, assign a score of 0.",

            "If the actual output directly contradicts all reference answers, assign a score of 0. Extra context that does not contradict a reference answer is acceptable.",

            "Assign a score of 1 if the actual output correctly answers the question consistent with at least one reference answer, even if worded more fully or with additional context. Assign a score of 0 only if there is a clear factual mismatch or direct contradiction with all reference answers.",

            "If the actual output is formatted differently, e.g. answered as a full sentence instead of just short phrases, it can also be considered similarly."
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=vllm_judge,
        strict_mode=True
    )

    test_cases = []
    for item in data_pairs:
        answers = item['ans2']
        if isinstance(answers, list):
            expected = "\n".join(f"Answer {i+1}: {a}" for i, a in enumerate(answers))
        else:
            expected = str(answers)
        test_case = LLMTestCase(
            input=item.get('question', "No input context provided"),
            actual_output=str(item['ans1']),
            expected_output=expected
        )
        test_cases.append(test_case)

    results = evaluate(
        test_cases=test_cases,
        metrics=[correctness_metric],
        async_config=AsyncConfig(
            max_concurrent=30,
        ),
        cache_config=CacheConfig(write_cache=False),
        identifier=run_id
    )

    return results