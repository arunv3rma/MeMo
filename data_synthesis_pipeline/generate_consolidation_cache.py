"""
generate_consolidation_cache.py

Takes the output of combine_direct_and_indirect.py and generates
consolidated QA pairs by combining related QA pairs per document using
prepare_prompt_for_consolidation.

- Groups all QA pairs per doc_id (across all epochs from the input)
- Distributes docs across N clients by round-robin (doc index % total_clients)
- Each doc produces one consolidation call
- Resume via a checkpoint file

Usage:
    python generate_consolidation_cache.py \\
        --ports 8000 8001 8002 \\
        --input_file_path /path/to/indirect_cache.json \\
        --output_file_path /path/to/consolidation_cache.json \\
        [--corpus_path /path/to/corpus.jsonl] \\
        [--resume_checkpoint /path/to/checkpoint.json] \\
        [--max_new_tokens 32768] \\
        [--thinking_budget -1] \\
        [--stream] \\
        [--use_openai] \\
        [--max_concurrent_generation 50] \\
        [--num_hedges 3] \\
        [--checkpoint_iter_freq 20] \\
        [--min_qa_pairs 2] \\
        [--temperature 1.1] \\
        [--top_p 0.95] \\
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from general_prompt_utils import prepare_prompt_for_consolidation, extract_doc_metadata
from bcp_data_utils import load_corpus_from_jsonl
from nqa_data_utils import load_corpus_from_jsonl_nqa


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ResultEntry = dict[str, Any]
FailureMap  = dict[str, dict]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PromptSetting(Enum):
    Consolidation = "Consolidation"


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
        print(f"Warning: parsing failed for {prompt_setting}")
        raise ValueError(f"Failed to parse content: {e}")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class RunState:
    """Tracks completed work and failures across all clients."""
    results:  list[ResultEntry] = field(default_factory=list)
    failures: FailureMap        = field(default_factory=dict)
    _lock: asyncio.Lock         = field(default_factory=asyncio.Lock, repr=False, compare=False)

    def completed_keys(self) -> set[str]:
        """Set of doc_ids already done (including permanent errors)."""
        PERMANENT_ERRORS = ["CONTEXT_LENGTH_ERROR"]
        return {
            r["doc_id"]
            for r in self.results
            if "doc_id" in r
            and (
                "error" not in r or
                r["error"].get("type") in PERMANENT_ERRORS
            )
        }

    async def add_result(self, entry: ResultEntry) -> None:
        async with self._lock:
            self.results.append(entry)

    async def add_failure(self, doc_id: str, info: dict) -> None:
        async with self._lock:
            self.failures[doc_id] = info


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_resume_checkpoint(path: str) -> tuple[list[ResultEntry], FailureMap]:
    if not path or not os.path.exists(path):
        return [], {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results  = data.get("qa_pairs_cache", []) if isinstance(data, dict) else data
        failures = data.get("failed_context_length_chunks", {}) if isinstance(data, dict) else {}
        print(f"  Loaded resume checkpoint: {len(results)} results, {len(failures)} failures")
        return results, failures
    except Exception as e:
        print(f"  Warning: could not load checkpoint {path}: {e}")
        return [], {}


async def save_checkpoint(path: str, state: RunState) -> None:
    async with state._lock:
        data = {
            "total_entries":               len(state.results),
            "total_qa_pairs":              sum(len(r.get("qa_pairs", [])) for r in state.results),
            "total_consolidated_pairs":    sum(len(r.get("consolidated_qa_pairs", [])) for r in state.results),
            "failed_context_length_chunks": state.failures,
            "qa_pairs_cache":              state.results,
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
    use_openai: bool,
    use_openrouter: bool,
    client,
    client_model_name: str,
    preferred_error_message: str,
    max_new_tokens: int,
    prompt_setting: PromptSetting,
    stream: bool,
    thinking_budget: int,
    temperature: float = 1.1,
    top_p: float = 0.95,
) -> Any:
    try:
        if use_openai or use_openrouter:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=client_model_name,
                    messages=[{"role": "user", "content": prompt_content}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ),
                timeout=300,
            )
        else:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=client_model_name,
                    stream=stream,
                    messages=[{"role": "user", "content": prompt_content}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body={"chat_template_kwargs": {"thinking_budget": thinking_budget}},
                ),
                timeout=300,
            )

        content = response.choices[0].message.content
        print(f"[{task_id}] Raw response received")
        return parse_content_string(content, prompt_setting)

    except asyncio.CancelledError:
        print(f"[{task_id}] Cancelled.")
        return None

    except ValueError as e:
        print(f"[{task_id}] Parsing failed: {e}")
        return ("PARSING_ERROR", str(e))

    except Exception as e:
        error_msg = str(e)
        print(f"[{task_id}] {preferred_error_message}: {e}")

        if "maximum context length" in error_msg or ("max_tokens" in error_msg and "too large" in error_msg):
            input_tokens = None
            m = re.search(r'(\d+)\s+input tokens', error_msg)
            if m:
                input_tokens = int(m.group(1))
            return ("CONTEXT_LENGTH_ERROR", error_msg, input_tokens)

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
        delay = hedge_num * 180
        try:
            await asyncio.wait_for(failure_event.wait(), timeout=delay)
            print(f"[{task_id}] Launched early (previous hedge failed)")
            failure_event.clear()
        except asyncio.TimeoutError:
            pass

        if event.is_set():
            print(f"[{task_id}] Cancelled — already succeeded")
            return

    result = await query_large_model_async(task_id=task_id, **kwargs)

    is_error = isinstance(result, tuple) and len(result) >= 2 and result[0] in (
        "PARSING_ERROR", "CONTEXT_LENGTH_ERROR", "OTHER_ERROR"
    )

    if is_error:
        print(f"[{task_id}] Failed with {result[0]}")
        if result[0] == "CONTEXT_LENGTH_ERROR":
            event.set()
            await result_queue.put(result)
            return
        failure_event.set()
        if not event.is_set():
            await result_queue.put(result)
        return

    if result is not None and not event.is_set():
        print(f"[{task_id}] Succeeded!")
        event.set()
        await result_queue.put(result)
    elif result is not None:
        print(f"[{task_id}] Succeeded but another hedge already won")
    else:
        failure_event.set()


async def query_with_hedging(num_requests: int, request_id: str, **kwargs) -> Any:
    timeout = 100 + 180 * (num_requests - 1)

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
        print(f"[{request_id}] Timed out after {timeout}s")
        last_error = None
        while not result_queue.empty():
            last_error = await result_queue.get()
        return last_error if (last_error and isinstance(last_error, tuple)) else ("TIMEOUT", "All hedges timed out")

    except Exception as e:
        print(f"[{request_id}] Unexpected error: {e}")
        raise

    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Single doc consolidation
# ---------------------------------------------------------------------------

async def generate_consolidation_for_doc(
    doc_id: str,
    qa_pairs_str: str,
    doc_date: str,
    source_qa_count: int,
    original_qa_pairs: list[dict],
    client,
    client_model_name: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    client_id: int,
    state: RunState,
) -> ResultEntry:
    start = time.time()

    async with semaphore:
        print(f"[Client {client_id}][Doc {doc_id}] Starting consolidation ({source_qa_count} source QA pairs)...")
        gen_start = time.time()

        prompt_content = prepare_prompt_for_consolidation(
            qa_pairs=qa_pairs_str,
            doc_extracted_date=doc_date,
        )

        result = await query_with_hedging(
            num_requests=args.num_hedges,
            request_id=f"Client{client_id}_Doc_{doc_id}_Consolidation",
            prompt_content=prompt_content,
            use_openai=args.use_openai,
            use_openrouter=args.use_openrouter,
            client=client,
            client_model_name=client_model_name,
            preferred_error_message=f"Error consolidating QA for doc {doc_id}",
            max_new_tokens=args.max_new_tokens,
            prompt_setting=PromptSetting.Consolidation,
            stream=args.stream,
            thinking_budget=args.thinking_budget,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        gen_elapsed = time.time() - gen_start

        # ---- Error handling -----------------------------------------------
        if isinstance(result, tuple) and len(result) >= 2:
            error_type, error_msg = result[0], result[1]
            print(f"[Client {client_id}][Doc {doc_id}] Failed: {error_type}")

            if error_type == "CONTEXT_LENGTH_ERROR":
                input_tokens = result[2] if len(result) > 2 else None
                await state.add_failure(doc_id, {"error_msg": error_msg, "input_tokens": input_tokens})

            entry = {
                "doc_id":                doc_id,
                "client_id":             client_id,
                "source_qa_count":       source_qa_count,
                "qa_pairs":              original_qa_pairs,
                "consolidated_qa_pairs": [],
                "error":                 {"type": error_type, "message": error_msg},
                "timing":                {"generation": gen_elapsed, "total": time.time() - start},
            }
            await state.add_result(entry)
            return entry

        if not result:
            print(f"[Client {client_id}][Doc {doc_id}] Empty response")
            entry = {
                "doc_id":                doc_id,
                "client_id":             client_id,
                "source_qa_count":       source_qa_count,
                "qa_pairs":              original_qa_pairs,
                "consolidated_qa_pairs": [],
                "error":                 {"type": "EMPTY_RESPONSE", "message": "No consolidated QA pairs generated"},
                "timing":                {"generation": gen_elapsed, "total": time.time() - start},
            }
            await state.add_result(entry)
            return entry

        consolidated = result.get("consolidated_qa_pairs", []) if isinstance(result, dict) else []

        total_elapsed = time.time() - start
        print(
            f"[Client {client_id}][Doc {doc_id}] "
            f"Done — {len(consolidated)} consolidated pairs in {total_elapsed:.2f}s"
        )

        entry = {
            "doc_id":                doc_id,
            "client_id":             client_id,
            "source_qa_count":       source_qa_count,
            "qa_pairs":              original_qa_pairs + consolidated,
            "consolidated_qa_pairs": consolidated,
            "timing":                {"generation": gen_elapsed, "total": total_elapsed},
        }
        await state.add_result(entry)
        return entry


# ---------------------------------------------------------------------------
# Per-client worker
# ---------------------------------------------------------------------------

async def run_client(
    client_id: int,
    port: int,
    work_items: list[tuple[str, str, str, int, list[dict]]],  # (doc_id, qa_pairs_str, doc_date, source_qa_count, original_qa_pairs)
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
        from third_party_api_client import OPENAI_CLIENT  # noqa
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

    async def _safe(doc_id: str, qa_pairs_str: str, doc_date: str, source_qa_count: int, original_qa_pairs: list[dict]) -> None:
        try:
            await generate_consolidation_for_doc(
                doc_id=doc_id,
                qa_pairs_str=qa_pairs_str,
                doc_date=doc_date,
                source_qa_count=source_qa_count,
                original_qa_pairs=original_qa_pairs,
                client=client,
                client_model_name=model_name,
                args=args,
                semaphore=semaphore,
                client_id=client_id,
                state=state,
            )
        except Exception as e:
            print(f"[Client {client_id}][Doc {doc_id}] Unexpected: {e}")
            traceback.print_exc()

    tasks = [_safe(doc_id, qa_str, doc_date, count, orig) for doc_id, qa_str, doc_date, count, orig in work_items]
    total = len(tasks)
    completed = 0
    start_time = time.time()
    checkpoint_counter = 0

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
            checkpoint_counter += 1
            await save_checkpoint(checkpoint_path, state)
            print(f"[Client {client_id}] Checkpoint {checkpoint_counter} saved ({len(state.results)} total entries)")

    print(f"[Client {client_id}] All work done.")


# ---------------------------------------------------------------------------
# Input loading and work distribution
# ---------------------------------------------------------------------------

def load_and_group_qa_pairs(input_path: str) -> dict[str, list[dict]]:
    """Load the indirect QA cache and group all QA pairs by doc_id across epochs."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("qa_pairs_cache", []) if isinstance(data, dict) else data

    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        doc_id = entry.get("doc_id")
        if not doc_id or "error" in entry:
            continue
        qa_pairs = entry.get("qa_pairs", [])
        if not qa_pairs:
            continue
        if doc_id not in grouped:
            grouped[doc_id] = []
        grouped[doc_id].extend(qa_pairs)

    return grouped


def format_qa_pairs_for_prompt(qa_pairs: list[dict]) -> str:
    """Convert a list of QA pair dicts into a numbered string for the consolidation prompt."""
    lines = []
    for i, qa in enumerate(qa_pairs, 1):
        q = qa.get("question", "")
        a = qa.get("answer", "")
        lines.append(f"{i}. Q: {q}\n   A: {a}")
    return "\n\n".join(lines)


def load_doc_dates(corpus_path: str, dataset: str = "bcp") -> dict[str, str]:
    """Load corpus and extract the date metadata for each doc_id."""
    try:
        if dataset == "nqa":
            texts, doc_ids = load_corpus_from_jsonl_nqa(corpus_path)
        else:
            texts, doc_ids = load_corpus_from_jsonl(corpus_path)
        return {doc_id: extract_doc_metadata(text) for text, doc_id in zip(texts, doc_ids)}
    except Exception as e:
        print(f"Warning: could not load corpus for dates: {e}")
        return {}


def build_work_items(
    grouped_qa: dict[str, list[dict]],
    doc_dates: dict[str, str],
    num_clients: int,
    completed_keys: set[str],
    min_qa_pairs: int,
    test_mode: bool,
) -> list[list[tuple[str, str, str, int, list[dict]]]]:
    """
    Distribute docs across clients by round-robin on sorted doc_id index.
    Returns a list of length num_clients, each a list of
    (doc_id, qa_pairs_str, doc_date, source_qa_count, original_qa_pairs).
    """
    per_client: list[list[tuple[str, str, str, int, list[dict]]]] = [[] for _ in range(num_clients)]

    for chunk_idx, doc_id in enumerate(sorted(grouped_qa.keys())):
        if doc_id in completed_keys:
            continue
        qa_pairs = grouped_qa[doc_id]
        if len(qa_pairs) < min_qa_pairs:
            print(f"  Skipping doc {doc_id}: only {len(qa_pairs)} QA pairs (min: {min_qa_pairs})")
            continue
        client_id = chunk_idx % num_clients
        qa_pairs_str = format_qa_pairs_for_prompt(qa_pairs)
        doc_date = doc_dates.get(doc_id, "No date provided in document")
        per_client[client_id].append((doc_id, qa_pairs_str, doc_date, len(qa_pairs), qa_pairs))

    if test_mode:
        per_client = [items[:3] for items in per_client]

    for cid, items in enumerate(per_client):
        print(f"  Client {cid}: {len(items)} work items")

    return per_client


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    # ---- Passthrough mode --------------------------------------------------
    if args.passthrough:
        print("[PASSTHROUGH] Skipping consolidation — passing QA pairs through unchanged...")
        grouped_qa = load_and_group_qa_pairs(args.input_file_path)
        results = [
            {
                "doc_id":                doc_id,
                "client_id":             0,
                "source_qa_count":       len(qa_pairs),
                "qa_pairs":              qa_pairs,
                "consolidated_qa_pairs": [],
                "timing":                {"generation": 0.0, "total": 0.0},
            }
            for doc_id, qa_pairs in grouped_qa.items()
        ]
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
    print(f"  Input   : {args.input_file_path}")
    print(f"  Output  : {args.output_file_path}")
    print("=" * 80 + "\n")

    # ---- Resume state ------------------------------------------------------
    prior_results, prior_failures = load_resume_checkpoint(args.resume_checkpoint)
    state = RunState(results=prior_results, failures=prior_failures)
    completed_keys = state.completed_keys()
    print(f"  Resuming with {len(completed_keys)} completed doc_ids\n")

    # ---- Load and group input QA pairs -------------------------------------
    print("Loading and grouping QA pairs from input file...")
    grouped_qa = load_and_group_qa_pairs(args.input_file_path)
    print(f"  Found {len(grouped_qa)} docs with QA pairs\n")

    # ---- Optionally load doc dates from corpus -----------------------------
    doc_dates: dict[str, str] = {}
    if args.corpus_path:
        print("Loading doc dates from corpus...")
        doc_dates = load_doc_dates(args.corpus_path, args.dataset)
        print(f"  Loaded dates for {len(doc_dates)} docs\n")

    # ---- Work distribution -------------------------------------------------
    print("Work distribution:")
    per_client_work = build_work_items(
        grouped_qa=grouped_qa,
        doc_dates=doc_dates,
        num_clients=num_clients,
        completed_keys=completed_keys,
        min_qa_pairs=args.min_qa_pairs,
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
    total_errors      = sum(1 for r in state.results if "error" in r)
    total_pairs       = sum(len(r.get("qa_pairs", [])) for r in state.results)
    total_consolidated = sum(len(r.get("consolidated_qa_pairs", [])) for r in state.results)

    print(f"\n{'='*80}")
    print(f"  All clients finished in {overall_elapsed:.2f}s")
    print(f"  Total entries            : {len(state.results)}")
    print(f"  Total qa_pairs           : {total_pairs}  (original + consolidated)")
    print(f"  Total consolidated only  : {total_consolidated}")
    print(f"  Total errors             : {total_errors}")
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
    p = argparse.ArgumentParser(description="Consolidation QA generation from indirect QA cache")

    # Clients
    p.add_argument("--ports", type=int, nargs="+", default=[],
                   help="Port numbers for each local vLLM client (not needed with --use_openrouter)")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--use_openai", action="store_true")
    p.add_argument("--use_openrouter", action="store_true",
                   help="Use OpenRouter API (reads OPENROUTER_API_KEY from .env)")
    p.add_argument("--model_name", type=str, default=None,
                   help="Model name for OpenRouter (e.g. 'google/gemini-2.0-flash-001')")

    # Generation
    p.add_argument("--max_new_tokens",   type=int,   default=32768)
    p.add_argument("--thinking_budget",  type=int,   default=-1)
    p.add_argument("--stream",           action="store_true")
    p.add_argument("--temperature",      type=float, default=1.1)
    p.add_argument("--top_p",            type=float, default=0.95)
    p.add_argument("--num_hedges",       type=int,   default=3)

    # Files
    p.add_argument("--input_file_path",  type=str, required=True,
                   help="Path to output JSON from combine_direct_and_indirect.py")
    p.add_argument("--output_file_path", type=str, required=True)
    p.add_argument("--corpus_path",      type=str, default=None,
                   help="Optional corpus JSONL path for extracting doc dates")
    p.add_argument("--dataset", type=str, default="bcp", choices=["bcp", "nqa"],
                   help="Dataset type: 'bcp' (BrowseComp+) or 'nqa' (NarrativeQA)")
    p.add_argument("--resume_checkpoint", type=str, default=None,
                   help="Path to a prior checkpoint to resume from")

    # Concurrency / checkpointing
    p.add_argument("--max_concurrent_generation", type=int, default=50)
    p.add_argument("--checkpoint_iter_freq",      type=int, default=20)

    # Filtering
    p.add_argument("--min_qa_pairs", type=int, default=2,
                   help="Minimum QA pairs a doc must have to attempt consolidation")

    # Testing
    p.add_argument("--test_mode", action="store_true",
                   help="Process only first 3 items per client")

    # Ablation
    p.add_argument("--passthrough", action="store_true",
                   help="Skip consolidation and pass QA pairs through unchanged (for ablation studies)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
