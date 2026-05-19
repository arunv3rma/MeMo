"""
check_self_containment_post_combination.py

Checks QA pairs for self-containment and attempts to fix failing pairs.

Pipeline per QA pair:
  1. Check if self-contained (binary pass/fail)
  2. If fail → attempt correction with chunk_content + doc_extracted_date context
  3. Retry check after correction
  4. Repeat up to MAX_FIX_ATTEMPTS times
  5. Mark as permanently failed if still not self-contained after all attempts

Usage:
    python check_self_containment_post_combination.py \
        --ports 8000 8001 8002 \
        --input_file_path /path/to/qa_pairs.json \
        --output_file_path /path/to/output.json \
        --corpus_path /path/to/corpus.jsonl \
        [--resume_checkpoint /path/to/checkpoint.json] \
        [--max_new_tokens 4096] \
        [--max_concurrent_generation 50] \
        [--num_hedges 3] \
        [--checkpoint_iter_freq 20] \
        [--use_openai] \
        [--test_mode]
"""

import argparse
import asyncio
import json
import os
import re
import ast
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openai import AsyncOpenAI

from general_prompt_utils import (
    extract_doc_metadata,
    prepare_prompt_for_self_containment_check,
    prepare_prompt_for_self_containment_fix,
)
from bcp_data_utils import load_corpus_from_jsonl
from nqa_data_utils import load_corpus_from_jsonl_nqa


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FIX_ATTEMPTS = 5


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PromptSetting(Enum):
    SelfContainmentCheck   = "SelfContainmentCheck"
    SelfContainmentFix     = "SelfContainmentFix"


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

def parse_content_string(content: Any, prompt_setting: PromptSetting) -> Any:
    if not isinstance(content, str):
        return content

    cleaned = content.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    final = cleaned.strip()

    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', final)
    if json_match:
        final = json_match.group(1).strip()

    try:
        result = ast.literal_eval(final)
        if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
            return result
    except (ValueError, SyntaxError):
        pass

    try:
        return json.loads(final)
    except json.JSONDecodeError as e:
        print(f"Warning: parsing failed for {prompt_setting}: {e}")
        raise ValueError(f"Failed to parse content: {e}")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    """Tracks completed work and failures across all clients."""
    results:  list[dict] = field(default_factory=list)
    failures: dict       = field(default_factory=dict)
    _lock: asyncio.Lock  = field(default_factory=asyncio.Lock, repr=False, compare=False)

    def completed_keys(self) -> set[tuple[str, int]]:
        """Set of (doc_id, qa_index) pairs already processed."""
        return {
            (r["doc_id"], r["qa_index"])
            for r in self.results
            if "doc_id" in r and "qa_index" in r
        }

    async def add_result(self, entry: dict) -> None:
        async with self._lock:
            self.results.append(entry)

    async def add_failure(self, key: str, info: dict) -> None:
        async with self._lock:
            self.failures[key] = info


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_resume_checkpoint(path: str) -> tuple[list[dict], dict]:
    path = path.replace(".json", "_checkpoint.json")
    if not path or not os.path.exists(path):
        return [], {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results  = data.get("results", [])
        failures = data.get("failures", {})
        print(f"  Loaded resume checkpoint: {len(results)} results, {len(failures)} failures")
        return results, failures
    except Exception as e:
        print(f"  Warning: could not load checkpoint {path}: {e}")
        return [], {}


async def save_checkpoint(path: str, state: RunState) -> None:
    async with state._lock:
        data = {
            "total_entries":   len(state.results),
            "failures":        state.failures,
            "results":         state.results,
        }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.rename(tmp, path)


# ---------------------------------------------------------------------------
# Async LLM query
# ---------------------------------------------------------------------------

async def query_large_model_async(
    task_id: str,
    prompt_content: str,
    client,
    client_model_name: str,
    use_openai: bool,
    use_openrouter: bool,
    max_new_tokens: int,
    prompt_setting: PromptSetting,
    temperature: float = 0.0,
) -> Any:
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=client_model_name,
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            ),
            timeout=120,
        )
        content = response.choices[0].message.content
        return parse_content_string(content, prompt_setting)

    except asyncio.CancelledError:
        print(f"[{task_id}] Cancelled.")
        return None

    except ValueError as e:
        print(f"[{task_id}] Parsing failed: {e}")
        return ("PARSING_ERROR", str(e))

    except Exception as e:
        error_msg = str(e)
        print(f"[{task_id}] Error: {e}")
        return ("OTHER_ERROR", error_msg)


# ---------------------------------------------------------------------------
# Hedged requests
# ---------------------------------------------------------------------------

async def _hedge_wrapper(
    event: asyncio.Event,
    result_queue: asyncio.Queue,
    task_id: str,
    hedge_num: int,
    failure_event: asyncio.Event,
    **kwargs,
) -> None:
    if hedge_num > 0:
        delay = hedge_num * 60
        try:
            await asyncio.wait_for(failure_event.wait(), timeout=delay)
            print(f"[{task_id}] Launched early (previous hedge failed)")
            failure_event.clear()
        except asyncio.TimeoutError:
            pass

        if event.is_set():
            return

    result = await query_large_model_async(task_id=task_id, **kwargs)

    is_error = isinstance(result, tuple) and result[0] in ("PARSING_ERROR", "OTHER_ERROR")

    if is_error:
        failure_event.set()
        if not event.is_set():
            await result_queue.put(result)
        return

    if result is not None and not event.is_set():
        event.set()
        await result_queue.put(result)
    else:
        failure_event.set()


async def query_with_hedging(num_requests: int, request_id: str, **kwargs) -> Any:
    timeout = 60 + 60 * (num_requests - 1)

    event         = asyncio.Event()
    failure_event = asyncio.Event()
    result_queue  = asyncio.Queue(maxsize=num_requests)
    tasks         = []

    for hedge_num in range(num_requests):
        task_id = f"{request_id}-Hedge-{hedge_num}"
        task = asyncio.create_task(
            _hedge_wrapper(event, result_queue, task_id, hedge_num, failure_event, **kwargs)
        )
        tasks.append(task)

    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return await result_queue.get()
    except asyncio.TimeoutError:
        return ("TIMEOUT", "All hedges timed out")
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Core: check + fix loop for a single QA pair
# ---------------------------------------------------------------------------

async def check_and_fix_qa_pair(
    doc_id: str,
    qa_index: int,
    question: str,
    answer: str,
    chunk_content: str,
    doc_extracted_date: str,
    client,
    client_model_name: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    client_id: int,
    state: RunState,
) -> dict:
    start = time.time()
    request_base = f"Client{client_id}_Doc_{doc_id}_QA{qa_index}"

    async with semaphore:
        current_question = question
        current_answer   = answer
        attempts         = []

        for attempt in range(MAX_FIX_ATTEMPTS + 1):  # attempt 0 = initial check

            # ---- Step 1: Check self-containment ----------------------------
            print(f"[{request_base}] Attempt {attempt} — checking self-containment...")

            check_prompt = prepare_prompt_for_self_containment_check(
                current_question, current_answer
            )
            check_result = await query_with_hedging(
                num_requests=args.num_hedges,
                request_id=f"{request_base}_Check_Attempt{attempt}",
                prompt_content=check_prompt,
                client=client,
                client_model_name=client_model_name,
                use_openai=args.use_openai,
                use_openrouter=args.use_openrouter,
                max_new_tokens=256,
                prompt_setting=PromptSetting.SelfContainmentCheck,
                temperature=0.0,
            )

            # Handle check errors
            if isinstance(check_result, tuple):
                print(f"[{request_base}] Check failed with error: {check_result[0]}")
                attempts.append({
                    "attempt": attempt,
                    "stage": "check",
                    "error": check_result[0],
                    "question": current_question,
                    "answer": current_answer,
                })
                break

            is_self_contained = check_result.get("is_self_contained", False)
            print(f"[{request_base}] is_self_contained={is_self_contained}")

            attempts.append({
                "attempt": attempt,
                "stage": "check",
                "is_self_contained": is_self_contained,
                "question": current_question,
                "answer": current_answer,
            })

            # ---- Self-contained: done --------------------------------------
            if is_self_contained:
                entry = {
                    "doc_id":            doc_id,
                    "qa_index":          qa_index,
                    "original_question": question,
                    "original_answer":   answer,
                    "final_question":    current_question,
                    "final_answer":      current_answer,
                    "is_self_contained": True,
                    "attempts_needed":   attempt,
                    "attempts":          attempts,
                    "timing":            time.time() - start,
                }
                await state.add_result(entry)
                return entry

            # ---- Max attempts reached: give up -----------------------------
            if attempt == MAX_FIX_ATTEMPTS:
                print(f"[{request_base}] Exceeded {MAX_FIX_ATTEMPTS} fix attempts — marking as failed")
                entry = {
                    "doc_id":            doc_id,
                    "qa_index":          qa_index,
                    "original_question": question,
                    "original_answer":   answer,
                    "final_question":    current_question,
                    "final_answer":      current_answer,
                    "is_self_contained": False,
                    "attempts_needed":   attempt,
                    "attempts":          attempts,
                    "timing":            time.time() - start,
                    "error":             "MAX_FIX_ATTEMPTS_EXCEEDED",
                }
                await state.add_result(entry)
                await state.add_failure(
                    f"{doc_id}_qa{qa_index}",
                    {"reason": "MAX_FIX_ATTEMPTS_EXCEEDED", "last_question": current_question, "last_answer": current_answer}
                )
                return entry

            # ---- Step 2: Attempt fix ---------------------------------------
            print(f"[{request_base}] Fixing (attempt {attempt + 1}/{MAX_FIX_ATTEMPTS})...")

            fix_prompt = prepare_prompt_for_self_containment_fix(
                question=current_question,
                answer=current_answer,
                chunk_content=chunk_content,
                doc_extracted_date=doc_extracted_date,
            )
            fix_result = await query_with_hedging(
                num_requests=args.num_hedges,
                request_id=f"{request_base}_Fix_Attempt{attempt}",
                prompt_content=fix_prompt,
                client=client,
                client_model_name=client_model_name,
                use_openai=args.use_openai,
                use_openrouter=args.use_openrouter,
                max_new_tokens=1024,
                prompt_setting=PromptSetting.SelfContainmentFix,
                temperature=0.0,
            )

            # Handle fix errors
            if isinstance(fix_result, tuple):
                print(f"[{request_base}] Fix failed with error: {fix_result[0]}")
                attempts.append({
                    "attempt": attempt,
                    "stage": "fix",
                    "error": fix_result[0],
                })
                break

            # Update current question/answer for next check iteration
            current_question = fix_result.get("question", current_question)
            current_answer   = fix_result.get("answer", current_answer)

            attempts.append({
                "attempt": attempt,
                "stage": "fix",
                "question": current_question,
                "answer": current_answer,
            })

        # Fell out of loop due to error
        entry = {
            "doc_id":            doc_id,
            "qa_index":          qa_index,
            "original_question": question,
            "original_answer":   answer,
            "final_question":    current_question,
            "final_answer":      current_answer,
            "is_self_contained": False,
            "attempts_needed":   len(attempts),
            "attempts":          attempts,
            "timing":            time.time() - start,
            "error":             "PIPELINE_ERROR",
        }
        await state.add_result(entry)
        return entry


# ---------------------------------------------------------------------------
# Per-client worker
# ---------------------------------------------------------------------------

async def run_client(
    client_id: int,
    port: int,
    work_items: list[dict],
    args: argparse.Namespace,
    state: RunState,
    checkpoint_path: str,
) -> None:
    print(f"\n[Client {client_id}] Starting on port {port} — {len(work_items)} items")

    if args.use_openai:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(current_dir, os.pardir))
        from dotenv import load_dotenv
        load_dotenv()
        from third_party_api_client import OPENAI_CLIENT
        client = OPENAI_CLIENT
        model_name = os.getenv("OPENAI_MODEL_NAME")
    elif args.use_openrouter:
        from dotenv import load_dotenv
        load_dotenv()
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        model_name = args.model_name
        print(f"[Client {client_id}] Using OpenRouter model: {model_name}")
    else:
        client = AsyncOpenAI(
            base_url=f"http://{args.host}:{port}/v1",
            api_key="dummy",
        )
        try:
            models = await client.models.list()
            model_name = models.data[0].id
            print(f"[Client {client_id}] ✓ Server online: {model_name}")
        except Exception as e:
            print(f"[Client {client_id}] ✗ Server unreachable on port {port}: {e}")
            return

    semaphore = asyncio.Semaphore(args.max_concurrent_generation)

    async def _safe(item: dict) -> None:
        try:
            await check_and_fix_qa_pair(
                doc_id=item["doc_id"],
                qa_index=item["qa_index"],
                question=item["question"],
                answer=item["answer"],
                chunk_content=item["chunk_content"],
                doc_extracted_date=item["doc_extracted_date"],
                client=client,
                client_model_name=model_name,
                args=args,
                semaphore=semaphore,
                client_id=client_id,
                state=state,
            )
        except Exception as e:
            print(f"[Client {client_id}][Doc {item['doc_id']}][QA {item['qa_index']}] Unexpected: {e}")
            traceback.print_exc()

    tasks = [_safe(item) for item in work_items]
    total = len(tasks)
    completed = 0
    start_time = time.time()

    for coro in asyncio.as_completed(tasks):
        await coro
        completed += 1

        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta  = (total - completed) / rate if rate > 0 else 0
        print(
            f"[Client {client_id}] Progress: {completed}/{total} "
            f"({100*completed/total:.1f}%) | {rate:.2f}/s | ETA {eta/60:.1f}min"
        )

        if completed % args.checkpoint_iter_freq == 0:
            await save_checkpoint(checkpoint_path, state)
            print(f"[Client {client_id}] Checkpoint saved ({len(state.results)} total entries)")

    print(f"[Client {client_id}] All work done.")


# ---------------------------------------------------------------------------
# Work distribution
# ---------------------------------------------------------------------------

def build_work_items(
    input_data: list[dict],
    corpus_map: dict[str, str],
    num_clients: int,
    completed_keys: set[tuple[str, int]],
    test_mode: bool,
) -> list[list[dict]]:
    """
    Flatten all QA pairs from input data into individual work items,
    attach chunk_content and doc_extracted_date, then distribute across clients.
    """
    all_items = []

    for entry in input_data['qa_pairs_cache']:
        doc_id = entry.get("doc_id", "")
        chunk_content = corpus_map.get(doc_id, "")
        doc_extracted_date = extract_doc_metadata(chunk_content) if chunk_content else ""

        for qa_index, qa in enumerate(entry.get("qa_pairs", [])):
            if (doc_id, qa_index) in completed_keys:
                continue
            all_items.append({
                "doc_id":             doc_id,
                "qa_index":           qa_index,
                "question":           qa.get("question", ""),
                "answer":             qa.get("answer", ""),
                "chunk_content":      chunk_content,
                "doc_extracted_date": doc_extracted_date,
            })

    if test_mode:
        all_items = all_items[:num_clients * 3]

    # Round-robin distribute across clients
    per_client: list[list[dict]] = [[] for _ in range(num_clients)]
    for idx, item in enumerate(all_items):
        per_client[idx % num_clients].append(item)

    for cid, items in enumerate(per_client):
        print(f"  Client {cid}: {len(items)} work items")

    return per_client


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    # ---- Passthrough mode --------------------------------------------------
    if args.passthrough:
        print("[PASSTHROUGH] Skipping self-containment check — marking all QA pairs as self-contained...")
        with open(args.input_file_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        entries = input_data.get("qa_pairs_cache", []) if isinstance(input_data, dict) else input_data
        results = []
        for entry in entries:
            doc_id = entry.get("doc_id", "")
            for qa_index, qa in enumerate(entry.get("qa_pairs", [])):
                q = qa.get("question", "")
                a = qa.get("answer", "")
                results.append({
                    "doc_id":            doc_id,
                    "qa_index":          qa_index,
                    "original_question": q,
                    "original_answer":   a,
                    "final_question":    q,
                    "final_answer":      a,
                    "is_self_contained": True,
                    "attempts_needed":   0,
                    "attempts":          [],
                    "timing":            0.0,
                })
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file_path)), exist_ok=True)
        output = {"qa_pairs_cache": results}
        with open(args.output_file_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"[PASSTHROUGH] Wrote {len(results)} entries → {args.output_file_path}")
        return

    if args.use_openrouter and not args.model_name:
        raise ValueError("--model_name is required when using --use_openrouter")
    if not args.use_openrouter and not args.use_openai and not args.ports:
        raise ValueError("--ports is required when not using --use_openrouter or --use_openai")

    ports = args.ports
    num_clients = len(ports) if not args.use_openrouter else 1

    print(f"\n{'='*80}")
    print(f"  Clients : {num_clients}  |  Ports: {ports if not args.use_openrouter else 'N/A (OpenRouter)'}")
    print(f"  Output  : {args.output_file_path}")
    print(f"  Max fix attempts: {MAX_FIX_ATTEMPTS}")
    print("=" * 80 + "\n")

    # ---- Resume state ------------------------------------------------------
    if args.resume_checkpoint is not None:
        prior_results, prior_failures = load_resume_checkpoint(args.resume_checkpoint)
    else:
        prior_results = []
        prior_failures = {}
    
    # NEW: Filter out previously failed items so they get retried
    if getattr(args, "retry_failed", False):
        successful_results = []
        retry_count = 0
        for r in prior_results:
            if r.get("is_self_contained") is True:
                successful_results.append(r)
            else:
                retry_count += 1
                # Remove from failures dict so we start with a clean slate
                fail_key = f"{r.get('doc_id')}_qa{r.get('qa_index')}"
                prior_failures.pop(fail_key, None)
                
        prior_results = successful_results
        if retry_count > 0:
            print(f"  [Retry Mode] Filtered out {retry_count} previously failed items to retry.\n")

    state = RunState(results=prior_results, failures=prior_failures)
    completed_keys = state.completed_keys()
    print(f"  Resuming with {len(completed_keys)} completed (doc_id, qa_index) pairs\n")

    # ---- Load input QA pairs -----------------------------------------------
    with open(args.input_file_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    print(f"  Input: {len(input_data)} entries loaded")

    # ---- Load corpus for chunk_content lookup ------------------------------
    if args.dataset == "nqa":
        corpus_texts, corpus_docids = load_corpus_from_jsonl_nqa(args.corpus_path)
    else:
        corpus_texts, corpus_docids = load_corpus_from_jsonl(args.corpus_path)
    corpus_map = dict(zip(corpus_docids, corpus_texts))
    print(f"  Corpus: {len(corpus_map)} chunks loaded\n")

    # ---- Work distribution -------------------------------------------------
    print("Work distribution:")
    per_client_work = build_work_items(
        input_data=input_data,
        corpus_map=corpus_map,
        num_clients=num_clients,
        completed_keys=completed_keys,
        test_mode=args.test_mode,
    )

    total_work = sum(len(w) for w in per_client_work)
    if total_work == 0:
        print("\nNothing to do — all work already completed.")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file_path)), exist_ok=True)
        with open(args.output_file_path, "w", encoding="utf-8") as f:
            json.dump({"qa_pairs_cache": state.results}, f, indent=2)
        print(f"  Output saved → {args.output_file_path}")
        return

    print(f"\n  Total work items: {total_work}\n")

    # ---- Checkpoint path ---------------------------------------------------
    base = args.output_file_path.replace(".json", "")
    checkpoint_path = f"{base}_checkpoint.json"
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    # ---- Launch clients ----------------------------------------------------
    overall_start = time.time()

    client_tasks = [
        asyncio.create_task(
            run_client(
                client_id=cid,
                port=ports[cid] if not args.use_openrouter else None,
                work_items=per_client_work[cid],
                args=args,
                state=state,
                checkpoint_path=checkpoint_path,
            )
        )
        for cid in range(num_clients)
    ]

    await asyncio.gather(*client_tasks)

    # ---- Final save --------------------------------------------------------
    overall_elapsed = time.time() - overall_start
    total_passed  = sum(1 for r in state.results if r.get("is_self_contained"))
    total_failed  = sum(1 for r in state.results if not r.get("is_self_contained"))

    print(f"\n{'='*80}")
    print(f"  All clients finished in {overall_elapsed:.2f}s")
    print(f"  Total entries   : {len(state.results)}")
    print(f"  Passed          : {total_passed}")
    print(f"  Failed          : {total_failed}")
    print("=" * 80 + "\n")

    await save_checkpoint(checkpoint_path, state)
    
    output = {
        "qa_pairs_cache" : state.results
    }

    with open(args.output_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  Final output saved → {args.output_file_path}")
    print(f"  Checkpoint saved   → {checkpoint_path}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-containment check and fix pipeline")

    p.add_argument("--ports",        type=int, nargs="+", default=[],
                   help="Port numbers for each local vLLM client (not needed with --use_openrouter)")
    p.add_argument("--host",         type=str, default="localhost")
    p.add_argument("--use_openai",   action="store_true")
    p.add_argument("--use_openrouter", action="store_true",
                   help="Use OpenRouter API (reads OPENROUTER_API_KEY from .env)")
    p.add_argument("--model_name",   type=str, default=None,
                   help="Model name for OpenRouter (e.g. 'google/gemini-2.0-flash-001')")

    p.add_argument("--input_file_path",  type=str, required=True,
                   help="Path to QA pairs JSON generated by generate_consolidation_cache.py")
    p.add_argument("--output_file_path", type=str, required=True)
    p.add_argument("--corpus_path",      type=str, required=True,
                   help="Path to corpus JSONL for chunk_content lookup")
    p.add_argument("--dataset", type=str, default="bcp", choices=["bcp", "nqa"],
                   help="Dataset type: 'bcp' (BrowseComp+) or 'nqa' (NarrativeQA)")
    p.add_argument("--resume_checkpoint", type=str, default=None)

    p.add_argument("--max_new_tokens",            type=int,   default=4096)
    p.add_argument("--num_hedges",                type=int,   default=3)
    p.add_argument("--max_concurrent_generation", type=int,   default=50)
    p.add_argument("--checkpoint_iter_freq",      type=int,   default=20)

    p.add_argument("--test_mode", action="store_true",
                   help="Process only first 3 items per client")

    p.add_argument("--retry_failed", action="store_true",
                   help="If set, will re-queue and retry QA pairs from the checkpoint that previously failed.")

    p.add_argument("--passthrough", action="store_true",
                   help="Skip self-containment check and mark all QA pairs as self-contained (for ablation studies)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))