"""
with_neg_generate_crossdoc_entity_combination_cache.py

Takes the output of generate_surface_entity_cache.py and generates cross-document
entity combination QA pairs using evidence_docs groupings from the questions file.

For each query group (set of docs that are all evidence for the same question):
  - Identifies which docs have QA pairs in the surface entity cache
  - Skips groups with fewer than --min_docs_with_qa docs having QA pairs
  - For each doc as anchor: iterates through its QA pairs one at a time
  - Each anchor QA pair is compared against ALL other docs' QA pairs in batches
  - Generates two types of cross-doc QA pairs:

    TYPE A (converging_clues):
      Different facts from ≥2 docs all describe the SAME entity.
      Q: "Who [fact from doc X] and [fact from doc Y] [and fact from doc Z]?"
      A: "[Entity Name]. [1-2 confirming facts from the contributing docs]."

    TYPE B (parallel_property):
      Different docs each mention a DIFFERENT entity sharing the SAME fact/property.
      Q: "Which [entity type]s [shared property]?"
      A: "[Entity A] and [Entity B]. [1 confirming sentence per entity]."

Output format: qa_pairs_cache structure consumed by generate_paraphrase_cache.py.
Cross-doc entries use doc_id = min(participating_doc_ids) and carry extra fields:
  type: "crossdoc", query_id, source_doc_ids

With --include_source_qa_pairs, all original step-6 entries are passed through
unchanged, followed by the new cross-doc entries (one entry per query group,
deduplicated by question-text hash).

Usage:
    python with_neg_generate_crossdoc_entity_combination_cache.py \\
        --ports 4325 4327 \\
        --input_file_path /path/to/surface_entity_cache.json \\
        --output_file_path /path/to/crossdoc_entity_combination_cache.json \\
        --qns_path /path/to/browsecomp_plus_questions.jsonl \\
        --max_num_questions 100 \\
        [--include_source_qa_pairs] \\
        [--min_docs_with_qa 2] \\
        [--max_other_qa_per_batch 20] \\
        [--resume_checkpoint /path/to/checkpoint.json] \\
        [--max_new_tokens 32768] \\
        [--thinking_budget -1] \\
        [--stream] \\
        [--use_openai] \\
        [--use_openrouter] \\
        [--model_name google/gemini-2.0-flash-001] \\
        [--num_hedges 3] \\
        [--max_concurrent_generation 50] \\
        [--checkpoint_iter_freq 20] \\
        [--temperature 1.1] \\
        [--top_p 0.95] \\
        [--test_mode]
"""

import argparse
import asyncio
import hashlib
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

from bcp_data_utils import load_questions_with_evidence_and_negative_docs
from musique_data_utils import load_questions_with_evidence_and_negative_docs_musique
from general_prompt_utils import prepare_prompt_for_crossdoc_anchor_combination


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ResultEntry = dict[str, Any]
FailureMap  = dict[str, dict]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PromptSetting(Enum):
    CrossDocCombination = "CrossDocCombination"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


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
    """
    Tracks completed work at (query_id, anchor_doc_id, anchor_qa_idx, batch_idx)
    granularity for fine-grained checkpointing and resume.
    """
    results:  list[ResultEntry] = field(default_factory=list)
    failures: FailureMap        = field(default_factory=dict)
    _lock: asyncio.Lock         = field(default_factory=asyncio.Lock, repr=False, compare=False)

    def completed_keys(self) -> set[tuple[str, str, int, int]]:
        PERMANENT_ERRORS = ["CONTEXT_LENGTH_ERROR"]
        return {
            (r["query_id"], r["anchor_doc_id"], r["anchor_qa_idx"], r["batch_idx"])
            for r in self.results
            if all(k in r for k in ("query_id", "anchor_doc_id", "anchor_qa_idx", "batch_idx"))
            and (
                "error" not in r
                or r["error"].get("type") in PERMANENT_ERRORS
            )
        }

    async def add_result(self, entry: ResultEntry) -> None:
        async with self._lock:
            self.results.append(entry)

    async def add_failure(self, key: str, info: dict) -> None:
        async with self._lock:
            self.failures[key] = info


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_resume_checkpoint(path: str) -> tuple[list[ResultEntry], FailureMap]:
    if not path or not os.path.exists(path):
        return [], {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Checkpoint stores work-item-level results under "work_item_results"
        results  = data.get("work_item_results", [])
        failures = data.get("failed_context_length_chunks", {})
        print(f"  Loaded resume checkpoint: {len(results)} results, {len(failures)} failures")
        return results, failures
    except Exception as e:
        print(f"  Warning: could not load checkpoint {path}: {e}")
        return [], {}


async def save_checkpoint(path: str, state: RunState) -> None:
    """Atomically save work-item-level state for resume."""
    async with state._lock:
        data = {
            "total_work_item_results":      len(state.results),
            "failed_context_length_chunks": state.failures,
            "work_item_results":            state.results,
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
# Single work-item generation
# ---------------------------------------------------------------------------

async def generate_crossdoc_for_work_item(
    item: dict,
    client,
    client_model_name: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    client_id: int,
    state: RunState,
) -> ResultEntry:
    query_id             = item["query_id"]
    anchor_doc_id        = item["anchor_doc_id"]
    anchor_qa_idx        = item["anchor_qa_idx"]
    anchor_qa            = item["anchor_qa"]
    batch                = item["batch"]           # list of (doc_id, qa_pair)
    batch_idx            = item["batch_idx"]
    representative_doc_id = item["representative_doc_id"]
    source_doc_ids       = item["source_doc_ids"]

    start      = time.time()
    request_id = f"Client{client_id}_Q{query_id}_Anc{anchor_doc_id}_QA{anchor_qa_idx}_B{batch_idx}"

    async with semaphore:
        print(
            f"[Client {client_id}][Query {query_id}][Anchor {anchor_doc_id}]"
            f"[QA {anchor_qa_idx}][Batch {batch_idx}] Starting ({len(batch)} candidates)..."
        )
        gen_start = time.time()

        prompt_content = prepare_prompt_for_crossdoc_anchor_combination(
            anchor_qa=anchor_qa,
            anchor_doc_id=anchor_doc_id,
            other_qa_batch=batch,
        )

        result = await query_with_hedging(
            num_requests=args.num_hedges,
            request_id=request_id,
            prompt_content=prompt_content,
            use_openai=args.use_openai,
            use_openrouter=args.use_openrouter,
            client=client,
            client_model_name=client_model_name,
            preferred_error_message=(
                f"Error generating crossdoc QA for query {query_id} "
                f"anchor {anchor_doc_id} qa {anchor_qa_idx} batch {batch_idx}"
            ),
            max_new_tokens=args.max_new_tokens,
            prompt_setting=PromptSetting.CrossDocCombination,
            stream=args.stream,
            thinking_budget=args.thinking_budget,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        gen_elapsed = time.time() - gen_start

        # ---- Error handling -----------------------------------------------
        if isinstance(result, tuple) and len(result) >= 2:
            error_type, error_msg = result[0], result[1]
            print(f"[Client {client_id}][Query {query_id}][Anchor {anchor_doc_id}][QA {anchor_qa_idx}][Batch {batch_idx}] Failed: {error_type}")

            if error_type == "CONTEXT_LENGTH_ERROR":
                input_tokens = result[2] if len(result) > 2 else None
                await state.add_failure(request_id, {"error_msg": error_msg, "input_tokens": input_tokens})

            entry = {
                "query_id":              query_id,
                "anchor_doc_id":         anchor_doc_id,
                "anchor_qa_idx":         anchor_qa_idx,
                "batch_idx":             batch_idx,
                "representative_doc_id": representative_doc_id,
                "source_doc_ids":        source_doc_ids,
                "qa_pairs":              [],
                "error":                 {"type": error_type, "message": error_msg},
                "timing":                {"generation": gen_elapsed, "total": time.time() - start},
            }
            await state.add_result(entry)
            return entry

        if not result:
            entry = {
                "query_id":              query_id,
                "anchor_doc_id":         anchor_doc_id,
                "anchor_qa_idx":         anchor_qa_idx,
                "batch_idx":             batch_idx,
                "representative_doc_id": representative_doc_id,
                "source_doc_ids":        source_doc_ids,
                "qa_pairs":              [],
                "error":                 {"type": "EMPTY_RESPONSE", "message": "No cross-doc QA pairs generated"},
                "timing":                {"generation": gen_elapsed, "total": time.time() - start},
            }
            await state.add_result(entry)
            return entry

        # Extract QA pairs — strip type/source_doc_ids metadata for downstream
        raw_pairs = result.get("crossdoc_qa_pairs", []) if isinstance(result, dict) else []
        qa_pairs = [
            {"question": p.get("question", ""), "answer": p.get("answer", "")}
            for p in raw_pairs
            if isinstance(p, dict) and p.get("question") and p.get("answer")
        ]

        total_elapsed = time.time() - start
        print(
            f"[Client {client_id}][Query {query_id}][Anchor {anchor_doc_id}]"
            f"[QA {anchor_qa_idx}][Batch {batch_idx}] Done — {len(qa_pairs)} pairs in {total_elapsed:.2f}s"
        )

        entry = {
            "query_id":              query_id,
            "anchor_doc_id":         anchor_doc_id,
            "anchor_qa_idx":         anchor_qa_idx,
            "batch_idx":             batch_idx,
            "representative_doc_id": representative_doc_id,
            "source_doc_ids":        source_doc_ids,
            "qa_pairs":              qa_pairs,
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

    async def _safe(item: dict) -> None:
        try:
            await generate_crossdoc_for_work_item(
                item=item,
                client=client,
                client_model_name=model_name,
                args=args,
                semaphore=semaphore,
                client_id=client_id,
                state=state,
            )
        except Exception as e:
            print(
                f"[Client {client_id}][Query {item.get('query_id')}]"
                f"[Anchor {item.get('anchor_doc_id')}][QA {item.get('anchor_qa_idx')}] Unexpected: {e}"
            )
            traceback.print_exc()

    tasks = [_safe(item) for item in work_items]
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
# Data loading
# ---------------------------------------------------------------------------

def load_surface_entity_cache(
    input_path: str,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Load the surface entity cache.

    Returns:
        original_entries: full list of entries (for pass-through with --include_source_qa_pairs)
        qa_map: dict[doc_id → list[{question, answer}]] — only docs with valid QA pairs
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_entries = data.get("qa_pairs_cache", []) if isinstance(data, dict) else data

    qa_map: dict[str, list[dict]] = {}
    for entry in original_entries:
        doc_id = entry.get("doc_id")
        if not doc_id or "error" in entry:
            continue
        qa_pairs = [
            {"question": qa.get("question", ""), "answer": qa.get("answer", "")}
            for qa in entry.get("qa_pairs", [])
            if isinstance(qa, dict) and qa.get("question") and qa.get("answer")
        ]
        if qa_pairs:
            qa_map[doc_id] = qa_pairs

    print(f"  Loaded surface entity cache: {len(original_entries)} entries, {len(qa_map)} docs with QA pairs")
    return original_entries, qa_map


# ---------------------------------------------------------------------------
# Work distribution
# ---------------------------------------------------------------------------

def _chunks(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def build_work_items(
    questions: list[dict],
    qa_map: dict[str, list[dict]],
    min_docs_with_qa: int,
    max_other_qa_per_batch: int,
    completed_keys: set[tuple[str, str, int, int]],
    num_clients: int,
    test_mode: bool,
) -> list[list[dict]]:
    """
    For each query group, iterate through every doc as anchor and compare each of
    its QA pairs against all other docs' QA pairs (batched).

    Work item key: (query_id, anchor_doc_id, anchor_qa_idx, batch_idx)
    Items already in completed_keys are skipped.

    Returns per_client work lists (round-robin distributed).
    """
    all_items: list[dict] = []

    for question in questions:
        query_id = str(question["question_no"])
        evidence_doc_ids = [
            doc.get("docid", "")
            for doc in question.get("evidence_docs", [])
            if doc.get("docid")
        ]
        
        # adding negative docs
        negative_doc_ids = [
            doc.get("docid", "")
            for doc in question.get("negative_docs", [])
            if doc.get("docid")
        ]
        
        evidence_doc_ids.extend(negative_doc_ids)
        # end of adding negative docs in

        participating = sorted([doc_id for doc_id in evidence_doc_ids if doc_id in qa_map])

        if len(participating) < min_docs_with_qa:
            continue

        representative_doc_id = participating[0]   # lexicographically smallest

        for anchor_doc_id in participating:
            qa_A = qa_map[anchor_doc_id]

            # All other docs' QA pairs, flattened with doc labels, in deterministic order
            other_qa_flat: list[tuple[str, dict]] = [
                (doc_id, qa)
                for doc_id in participating
                if doc_id != anchor_doc_id
                for qa in qa_map[doc_id]
            ]

            if not other_qa_flat:
                continue

            for anchor_idx, anchor_qa in enumerate(qa_A):
                batches = list(_chunks(other_qa_flat, max_other_qa_per_batch))
                for batch_idx, batch in enumerate(batches):
                    key = (query_id, anchor_doc_id, anchor_idx, batch_idx)
                    if key in completed_keys:
                        continue
                    all_items.append({
                        "query_id":              query_id,
                        "anchor_doc_id":         anchor_doc_id,
                        "anchor_qa_idx":         anchor_idx,
                        "anchor_qa":             anchor_qa,
                        "batch":                 batch,
                        "batch_idx":             batch_idx,
                        "representative_doc_id": representative_doc_id,
                        "source_doc_ids":        participating,
                    })

    if test_mode:
        all_items = all_items[: num_clients * 3]

    # Round-robin distribute across clients
    per_client: list[list[dict]] = [[] for _ in range(num_clients)]
    for idx, item in enumerate(all_items):
        per_client[idx % num_clients].append(item)

    for cid, items in enumerate(per_client):
        print(f"  Client {cid}: {len(items)} work items")

    return per_client


# ---------------------------------------------------------------------------
# Output aggregation
# ---------------------------------------------------------------------------

def aggregate_results_by_query(
    results: list[ResultEntry],
) -> dict[str, dict]:
    """
    Collect all per-work-item QA pairs by query_id.
    Deduplicate within each query group by SHA1 hash of the lowercased question string.

    Returns dict[query_id → {representative_doc_id, source_doc_ids, qa_pairs}]
    """
    by_query: dict[str, dict] = {}

    for r in results:
        query_id = r.get("query_id")
        if not query_id:
            continue

        if query_id not in by_query:
            by_query[query_id] = {
                "representative_doc_id": r["representative_doc_id"],
                "source_doc_ids":        r["source_doc_ids"],
                "qa_pairs":              [],
                "_seen_hashes":          set(),
            }

        for qa in r.get("qa_pairs", []):
            question = qa.get("question", "").strip()
            if not question:
                continue
            h = hashlib.sha1(question.lower().encode("utf-8")).hexdigest()
            if h not in by_query[query_id]["_seen_hashes"]:
                by_query[query_id]["_seen_hashes"].add(h)
                by_query[query_id]["qa_pairs"].append({
                    "question": question,
                    "answer":   qa.get("answer", ""),
                })

    # Remove internal dedup tracker before returning
    for v in by_query.values():
        v.pop("_seen_hashes", None)

    return by_query


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    if args.use_openrouter and not args.model_name:
        raise ValueError("--model_name is required when using --use_openrouter")
    if not args.use_openrouter and not args.use_openai and not args.ports:
        raise ValueError("--ports is required when not using --use_openrouter or --use_openai")

    ports = args.ports
    num_clients = len(ports) if not args.use_openrouter else 1

    print(f"\n{'='*80}")
    print(f"  Clients           : {num_clients}  |  Ports: {ports if not args.use_openrouter else 'N/A (OpenRouter)'}")
    print(f"  Input             : {args.input_file_path}")
    print(f"  Questions         : {args.qns_path}")
    print(f"  Output            : {args.output_file_path}")
    print(f"  min_docs_with_qa  : {args.min_docs_with_qa}")
    print(f"  max_other_qa/batch: {args.max_other_qa_per_batch}")
    print("=" * 80 + "\n")

    # ---- Resume state ------------------------------------------------------
    prior_results, prior_failures = load_resume_checkpoint(args.resume_checkpoint)
    state = RunState(results=prior_results, failures=prior_failures)
    completed_keys = state.completed_keys()
    print(f"  Resuming with {len(completed_keys)} completed work items\n")

    # ---- Load surface entity cache -----------------------------------------
    print("Loading surface entity cache...")
    original_entries, qa_map = load_surface_entity_cache(args.input_file_path)

    # ---- Load questions + build evidence groups ----------------------------
    print("Loading questions with evidence doc groups...")
    if args.dataset == "musique":
        questions = load_questions_with_evidence_and_negative_docs_musique(args.qns_path, args.max_num_questions)
    else:
        questions = load_questions_with_evidence_and_negative_docs(args.qns_path, args.max_num_questions)
    print(f"  Questions loaded: {len(questions)}\n")

    # ---- Work distribution -------------------------------------------------
    print("Building work items...")
    per_client_work = build_work_items(
        questions=questions,
        qa_map=qa_map,
        min_docs_with_qa=args.min_docs_with_qa,
        max_other_qa_per_batch=args.max_other_qa_per_batch,
        completed_keys=completed_keys,
        num_clients=num_clients,
        test_mode=args.test_mode,
    )

    total_work = sum(len(w) for w in per_client_work)
    if total_work == 0:
        print("\nNothing to do — all work already completed.")
    else:
        print(f"\n  Total work items: {total_work}\n")

        # ---- Checkpoint path -----------------------------------------------
        base = args.output_file_path.replace(".json", "")
        checkpoint_path = f"{base}_checkpoint.json"
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

        # ---- Launch clients ------------------------------------------------
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

        overall_elapsed = time.time() - overall_start
        total_errors = sum(1 for r in state.results if "error" in r)
        total_pairs  = sum(len(r.get("qa_pairs", [])) for r in state.results)

        print(f"\n{'='*80}")
        print(f"  All clients finished in {overall_elapsed:.2f}s")
        print(f"  Total work-item results : {len(state.results)}")
        print(f"  Total raw QA pairs      : {total_pairs}")
        print(f"  Total errors            : {total_errors}")
        print("=" * 80 + "\n")

        await save_checkpoint(checkpoint_path, state)
        print(f"  Checkpoint saved → {checkpoint_path}")

    # ---- Aggregate cross-doc results per query_id --------------------------
    print("\nAggregating and deduplicating cross-doc results by query_id...")
    aggregated = aggregate_results_by_query(state.results)

    crossdoc_entries_written = 0
    crossdoc_qa_total = 0
    crossdoc_entries: list[dict] = []

    for query_id, data in sorted(aggregated.items()):
        if not data["qa_pairs"]:
            continue
        crossdoc_entries.append({
            "doc_id":          data["representative_doc_id"],
            "type":            "crossdoc",
            "query_id":        query_id,
            "source_doc_ids":  data["source_doc_ids"],
            "qa_pairs":        data["qa_pairs"],
        })
        crossdoc_entries_written += 1
        crossdoc_qa_total += len(data["qa_pairs"])

    print(f"  Cross-doc query groups with QA pairs : {crossdoc_entries_written}")
    print(f"  Total cross-doc QA pairs (deduped)   : {crossdoc_qa_total}")

    # ---- Build final output ------------------------------------------------
    output_entries: list[dict] = []

    if args.include_source_qa_pairs:
        output_entries.extend(original_entries)
        print(f"  Passed through {len(original_entries)} original surface-entity entries")

    output_entries.extend(crossdoc_entries)

    output = {"qa_pairs_cache": output_entries}

    with open(args.output_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Final output saved → {args.output_file_path}")
    print(f"  Total entries in output: {len(output_entries)}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-document entity combination QA generation from surface entity cache"
    )

    # Clients
    p.add_argument("--ports", type=int, nargs="+", default=[],
                   help="Port numbers for each local vLLM client (not needed with --use_openrouter)")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--use_openai", action="store_true")
    p.add_argument("--use_openrouter", action="store_true",
                   help="Use OpenRouter API (reads OPENROUTER_API_KEY from .env)")
    p.add_argument("--model_name", type=str, default=None,
                   help="Model name for OpenRouter (e.g. 'google/gemini-2.0-flash-001')")

    # Files
    p.add_argument("--input_file_path",   type=str, required=True,
                   help="Path to surface entity cache JSON (output of generate_surface_entity_cache.py)")
    p.add_argument("--output_file_path",  type=str, required=True)
    p.add_argument("--qns_path",          type=str, required=True,
                   help="Path to questions JSONL (browsecomp_plus_questions.jsonl)")
    p.add_argument("--max_num_questions", type=int, default=None,
                   help="Max number of valid questions to load (same subset as rest of pipeline)")
    p.add_argument("--resume_checkpoint", type=str, default=None,
                   help="Path to a prior checkpoint to resume from")

    # Generation
    p.add_argument("--max_new_tokens",   type=int,   default=32768)
    p.add_argument("--thinking_budget",  type=int,   default=-1)
    p.add_argument("--stream",           action="store_true")
    p.add_argument("--temperature",      type=float, default=1.1)
    p.add_argument("--top_p",            type=float, default=0.95)
    p.add_argument("--num_hedges",       type=int,   default=3)

    # Concurrency / checkpointing
    p.add_argument("--max_concurrent_generation", type=int, default=50)
    p.add_argument("--checkpoint_iter_freq",      type=int, default=20)

    # Cross-doc specific
    p.add_argument("--min_docs_with_qa",      type=int, default=2,
                   help="Min number of docs in an evidence group that must have QA pairs")
    p.add_argument("--max_other_qa_per_batch", type=int, default=20,
                   help="Max number of other-docs QA pairs per LLM call batch")
    p.add_argument("--include_source_qa_pairs", action="store_true",
                   help="Pass through original surface-entity entries unchanged into the output")

    p.add_argument("--dataset", type=str, default="bcp", choices=["bcp", "musique"],
                   help="Dataset type: 'bcp' (BrowseComp+) or 'musique' (MuSiQue)")

    # Testing
    p.add_argument("--test_mode", action="store_true",
                   help="Process only first (num_clients * 3) work items total")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
