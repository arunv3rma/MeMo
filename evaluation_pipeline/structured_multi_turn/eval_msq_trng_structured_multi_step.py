import argparse
from openai import AsyncOpenAI
import json
import re
import os
import asyncio
import sys


def _compute_candidate_idk_streaks(conversation_history: list) -> dict[str, int]:
    idk_counts: dict[str, int] = {}
    for entry in conversation_history:
        if entry.get('phase') != 'entity_pinning':
            continue
        sub_questions_raw = entry.get('sub_questions_raw', [])
        sm_responses = entry.get('sm_responses', [])
        answer_map = {r['question']: r['answer'] for r in sm_responses}
        for sq in sub_questions_raw:
            target = sq.get('targets_candidate')
            if not target:
                continue
            answer = answer_map.get(sq.get('question', ''), '')
            if "i don't know" in answer.lower():
                idk_counts[target] = idk_counts.get(target, 0) + 1
    return idk_counts


def _pick_best_candidate(candidate_entities: list) -> tuple[str | None, bool]:
    if not candidate_entities:
        return None, False
    ranked = sorted(candidate_entities, key=lambda c: c.get('rank', 999))
    top = ranked[0]
    rank1 = [c for c in candidate_entities if c.get('rank', 999) == 1]
    ambiguous = len(rank1) > 1
    return top.get('name'), ambiguous


def extract_json(content: str) -> dict:
    """Extract JSON from LM response, handling markdown code fences."""
    if not content:
        raise ValueError("Empty response content")
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped.strip())


# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data_synthesis_pipeline'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from musique_data_utils import load_questions_with_evidence_docs_musique
from general_eval_prompt_utils import (
    generate_prompt_from_LM_to_SM_general_v1,
    generate_grounding_subquestions,
    generate_entity_pinning_prompt,
    generate_answer_seeking_prompt,
    format_LM_final_answer,
)

from model_utils import prepare_prompt_for_checking_mem_knowledge
from deepeval_utils import run_evaluation


def load_questions(qns_path, max_num_questions=None):
    questions = load_questions_with_evidence_docs_musique(qns_path, max_num_questions)
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
                return {
                    'question': sub_q,
                    'answer': response.choices[0].message.content,
                }
            except Exception as e:
                print(f"Error processing sub-question: {e}")
                return {
                    'question': sub_q,
                    'answer': "I don't know",
                }

    tasks = [process_sub_question(sub_q) for sub_q in sub_questions]
    sm_responses = await asyncio.gather(*tasks)
    return sm_responses


def filter_sm_responses(sm_responses: list) -> list:
    """Filter out responses where SM says it doesn't know."""
    return [
        r for r in sm_responses
        if "i don't know" not in r['answer'].lower()
    ]


async def query_with_majority_vote(
    sm_client,
    sub_questions_raw: list,
    prompt_formatter_func,
    sm_model_id: str,
    temperature: float,
    max_concurrent: int = 10,
) -> tuple[list, list]:
    """Query SM with each canonical sub-question once (no paraphrase expansion).

    Returns:
        consolidated_responses: one {question, answer, paraphrase_confidence} per canonical sub-question
        raw_responses: same list (identical to consolidated when paraphrasing is off)
    """
    canonical_questions = [item['question'] for item in sub_questions_raw]

    all_raw = await generate_batched_responses_async(
        sm_client=sm_client,
        sub_questions=canonical_questions,
        prompt_formatter_func=prompt_formatter_func,
        sm_model_id=sm_model_id,
        max_new_tokens=512,
        temperature=temperature,
        max_concurrent=max_concurrent,
    )

    consolidated = [
        {
            'question': item['question'],
            'answer': raw['answer'],
            'paraphrase_confidence': 'high',
        }
        for item, raw in zip(sub_questions_raw, all_raw)
    ]

    return consolidated, all_raw


async def process_single_question(
    lm_client,
    sm_client,
    row,
    question_idx,
    max_new_tokens,
    thinking_budget,
    stream,
    lm_model_id=None,
    sm_model_id=None,
    lm_grounding_temperature=1.1,
    sm_grounding_temperature=0.7,
    lm_entity_temperature=1.0,
    sm_entity_temperature=0.7,
    lm_answer_temperature=1.0,
    sm_answer_temperature=0.7,
    lm_final_temperature=0.3,
    max_entity_turns=5,
    max_answer_turns=5,
    dead_end_threshold=2,
    sm_max_concurrent=100,
    skip_entity_pinning=False,
):
    question = row['question']

    model_id = lm_model_id
    sm_model_id = sm_model_id

    conversation_history = []
    all_sm_responses_flat = []
    accumulated_known_facts: list = []

    # -------------------------------------------------------------------------
    # TURN 0: Initial grounding — LM breaks down clues, SM answers
    # -------------------------------------------------------------------------
    grounding_prompt = generate_grounding_subquestions(question)

    try:
        grounding_response = await lm_client.chat.completions.create(
            model=model_id,
            stream=stream,
            messages=[{"role": "user", "content": grounding_prompt}],
            max_tokens=max_new_tokens,
            temperature=lm_grounding_temperature,
            top_p=0.95,
            extra_body={"chat_template_kwargs": {"thinking_budget": thinking_budget}}
        )

        grounding_content = extract_json(grounding_response.choices[0].message.content) if lm_model_id else json.loads(grounding_response.choices[0].message.content)
        grounding_questions_raw = grounding_content['grounding_questions']

        if grounding_questions_raw and isinstance(grounding_questions_raw[0], str):
            grounding_questions_raw = [
                {"question": q, "paraphrases": [q], "question_type": "single",
                 "clue_indices": [i], "clues_involved": 1}
                for i, q in enumerate(grounding_questions_raw)
            ]

        grounding_subquestions = [item['question'] for item in grounding_questions_raw]
        print(f"Question {question_idx} - Round 0 (grounding) sub-questions: {grounding_subquestions}")

        grounding_sm_responses, grounding_sm_responses_raw = await query_with_majority_vote(
            sm_client=sm_client,
            sub_questions_raw=grounding_questions_raw,
            prompt_formatter_func=prepare_prompt_for_checking_mem_knowledge,
            sm_model_id=sm_model_id,
            temperature=sm_grounding_temperature,
            max_concurrent=sm_max_concurrent,
        )
        print(f"Question {question_idx} - Round 0 SM responses: {grounding_sm_responses}")

        conversation_history.append({
            'round': 0,
            'phase': 'grounding',
            'sm_responses': grounding_sm_responses
        })
        all_sm_responses_flat.extend(grounding_sm_responses)

    except Exception as e:
        print(f"Error in Turn 0 for question {question_idx}: {e}")
        return {
            "question_no": question_idx,
            "question": question,
            "groundtruth": row["groundtruth"],
            "model_final_response": "",
            "model_final_response_justification": "",
            "error": f"Turn 0 failed: {e}",
        }

    # -------------------------------------------------------------------------
    # PHASE 1: Entity Pinning (skipped when skip_entity_pinning=True)
    # -------------------------------------------------------------------------
    candidate_entities = []
    confirmed_entity = None
    entity_was_confirmed = False
    entity_phase_history = []

    if skip_entity_pinning:
        print(f"Question {question_idx} - Entity pinning skipped.")

    for entity_turn in range(0 if skip_entity_pinning else max_entity_turns):
        entity_turns_remaining = max_entity_turns - entity_turn - 1

        entity_prompt = generate_entity_pinning_prompt(
            question=question,
            conversation_history=conversation_history,
            turns_remaining=entity_turns_remaining,
            dead_end_threshold=dead_end_threshold,
            accumulated_known_facts=accumulated_known_facts,
            candidate_entities=candidate_entities,
            is_first_turn=(entity_turn == 0),
            candidate_idk_streaks=_compute_candidate_idk_streaks(conversation_history),
        )

        try:
            entity_response = await lm_client.chat.completions.create(
                model=model_id,
                stream=stream,
                messages=[{"role": "user", "content": entity_prompt}],
                max_tokens=max_new_tokens,
                temperature=lm_entity_temperature,
                top_p=0.95,
                extra_body={"chat_template_kwargs": {"thinking_budget": thinking_budget}}
            )

            raw_entity_text = entity_response.choices[0].message.content
            try:
                entity_content = extract_json(raw_entity_text) if lm_model_id else json.loads(raw_entity_text)
            except Exception as parse_err:
                print(f"Question {question_idx} - Entity turn {entity_turn + 1}: JSON parse failed ({parse_err}). Storing raw response and retrying.")
                conversation_history.append({
                    'round': len(conversation_history),
                    'phase': 'entity_pinning',
                    'sm_responses': [],
                    'raw_lm_response': raw_entity_text,
                })
                continue

            accumulated_known_facts = entity_content.get('known_facts', accumulated_known_facts)
            candidate_entities = entity_content.get('candidate_entities', candidate_entities)
            entity_confirmed_flag = entity_content.get('entity_confirmed', False)
            confirmed_entity_name = entity_content.get('confirmed_entity', None)
            primary_candidate = entity_content.get('primary_candidate', None)
            ambiguous = entity_content.get('ambiguous', False)
            decision = entity_content.get('decision', 'exhaust')
            sub_questions_raw = entity_content.get('sub_questions', [])

            print(f"Question {question_idx} - Entity turn {entity_turn + 1}: confirmed={entity_confirmed_flag}, primary={primary_candidate}, ambiguous={ambiguous}, decision={decision}")

            entity_round_meta = {
                'turn': entity_turn + 1,
                'phase': 'entity_pinning',
                'assessment': entity_content.get('assessment', ''),
                'new_sm_response_evaluation': entity_content.get('new_sm_response_evaluation', []),
                'known_facts': accumulated_known_facts,
                'candidate_entities': candidate_entities,
                'primary_candidate': primary_candidate,
                'ambiguous': ambiguous,
                'entity_confirmed': entity_confirmed_flag,
                'confirmed_entity': confirmed_entity_name,
                'decision': decision,
                'reasoning': entity_content.get('reasoning', ''),
            }

            if entity_confirmed_flag and confirmed_entity_name:
                confirmed_entity = confirmed_entity_name
                entity_was_confirmed = True
                entity_phase_history.append(entity_round_meta)
                print(f"Question {question_idx} - Entity confirmed: {confirmed_entity}")
                break

            if decision == 'exhaust' or not sub_questions_raw:
                confirmed_entity, ambiguous_at_exit = _pick_best_candidate(candidate_entities)
                entity_was_confirmed = False
                entity_round_meta['decision'] = 'exhaust'
                entity_phase_history.append(entity_round_meta)
                print(f"Question {question_idx} - Entity pinning exhausted. Best candidate: {confirmed_entity}, ambiguous={ambiguous_at_exit}")
                break

            print(f"Question {question_idx} - Entity turn {entity_turn + 1}: asking {len(sub_questions_raw)} sub-questions")

            entity_sm_responses, entity_sm_responses_raw = await query_with_majority_vote(
                sm_client=sm_client,
                sub_questions_raw=sub_questions_raw,
                prompt_formatter_func=prepare_prompt_for_checking_mem_knowledge,
                sm_model_id=sm_model_id,
                temperature=sm_entity_temperature,
                max_concurrent=sm_max_concurrent,
            )
            print(f"Question {question_idx} - Entity turn {entity_turn + 1} SM responses (consolidated): {entity_sm_responses}")

            conversation_history.append({
                'round': len(conversation_history),
                'phase': 'entity_pinning',
                'sm_responses': entity_sm_responses,
                'sub_questions_raw': sub_questions_raw,
            })
            all_sm_responses_flat.extend(entity_sm_responses)

            entity_round_meta['sub_questions_raw'] = sub_questions_raw
            entity_round_meta['sm_responses'] = entity_sm_responses
            entity_round_meta['sm_responses_raw'] = entity_sm_responses_raw
            entity_phase_history.append(entity_round_meta)

        except Exception as e:
            print(f"Error in entity-pinning turn {entity_turn + 1} for question {question_idx}: {e}")
            break

    else:
        confirmed_entity, _ = _pick_best_candidate(candidate_entities)
        entity_was_confirmed = False
        print(f"Question {question_idx} - max_entity_turns exhausted. Best candidate: {confirmed_entity}")

    # Edge case: no candidates at all (only bail out early if entity pinning actually ran)
    if confirmed_entity is None and not skip_entity_pinning:
        print(f"Question {question_idx} - No candidates identified. Skipping Phase 2.")
        all_sm_responses_filtered = filter_sm_responses(all_sm_responses_flat)
        final_prompt = format_LM_final_answer(
            all_sm_responses_filtered,
            question,
            accumulated_known_facts=accumulated_known_facts,
            accumulated_known_entities=[],
        )
        try:
            final_response = await lm_client.chat.completions.create(
                model=model_id,
                stream=stream,
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=max_new_tokens,
                temperature=lm_final_temperature,
                top_p=0.95,
                extra_body={"chat_template_kwargs": {"thinking_budget": thinking_budget}}
            )
            final_content = extract_json(final_response.choices[0].message.content) if lm_model_id else json.loads(final_response.choices[0].message.content)
            final_answer = final_content['final_answer']
            final_justification = final_content['justification']
        except Exception as e:
            print(f"Error in final answer (no-entity path) for question {question_idx}: {e}")
            return {
                "question_no": question_idx,
                "question": question,
                "model_final_response": "",
                "model_final_response_justification": "",
                "error": f"Final answer generation failed (no-entity path): {e}",
                "groundtruth": row["groundtruth"],
            }

        return {
            "question_no": question_idx,
            "question": question,
            "grounding_subquestions": grounding_subquestions,
            "grounding_sm_responses": grounding_sm_responses,
            "entity_phase_history": entity_phase_history,
            "confirmed_entity": None,
            "entity_was_confirmed": False,
            "entity_identified": False,
            "answer_phase_history": [],
            "all_sm_responses": all_sm_responses_flat,
            "all_sm_responses_filtered": all_sm_responses_filtered,
            "final_known_facts": accumulated_known_facts,
            "final_remaining_unknowns": [],
            "groundtruth": row["groundtruth"],
            "model_final_response": final_answer,
            "model_final_response_justification": final_justification,
        }

    # -------------------------------------------------------------------------
    # PHASE 2: Answer Seeking
    # -------------------------------------------------------------------------
    answer_phase_history = []

    for answer_turn in range(max_answer_turns):
        answer_turns_remaining = max_answer_turns - answer_turn - 1

        answer_prompt = generate_answer_seeking_prompt(
            question=question,
            conversation_history=conversation_history,
            turns_remaining=answer_turns_remaining,
            dead_end_threshold=dead_end_threshold,
            accumulated_known_facts=accumulated_known_facts,
            confirmed_entity=confirmed_entity,
            entity_was_confirmed=entity_was_confirmed,
            candidate_entities=candidate_entities,
        )

        try:
            answer_response = await lm_client.chat.completions.create(
                model=model_id,
                stream=stream,
                messages=[{"role": "user", "content": answer_prompt}],
                max_tokens=max_new_tokens,
                temperature=lm_answer_temperature,
                top_p=0.95,
                extra_body={"chat_template_kwargs": {"thinking_budget": thinking_budget}}
            )

            raw_answer_text = answer_response.choices[0].message.content
            try:
                answer_content = extract_json(raw_answer_text) if lm_model_id else json.loads(raw_answer_text)
            except Exception as parse_err:
                print(f"Question {question_idx} - Answer turn {answer_turn + 1}: JSON parse failed ({parse_err}). Storing raw response and retrying.")
                conversation_history.append({
                    'round': len(conversation_history),
                    'phase': 'answer_seeking',
                    'sm_responses': [],
                    'raw_lm_response': raw_answer_text,
                })
                continue

            accumulated_known_facts = answer_content.get('known_facts', accumulated_known_facts)
            decision = answer_content.get('decision', 'answer')
            entity_pivot = answer_content.get('entity_pivot', None)
            sub_questions_raw = answer_content.get('sub_questions', [])

            if entity_pivot and entity_pivot != confirmed_entity:
                print(f"Question {question_idx} - Answer turn {answer_turn + 1}: pivoting entity from '{confirmed_entity}' to '{entity_pivot}'")
                confirmed_entity = entity_pivot
                entity_was_confirmed = False

            print(f"Question {question_idx} - Answer turn {answer_turn + 1}: decision={decision}, entity={confirmed_entity}")

            answer_round_meta = {
                'turn': answer_turn + 1,
                'phase': 'answer_seeking',
                'assessment': answer_content.get('assessment', ''),
                'new_sm_response_evaluation': answer_content.get('new_sm_response_evaluation', []),
                'known_facts': accumulated_known_facts,
                'remaining_unknowns': answer_content.get('remaining_unknowns', []),
                'confirmed_entity': confirmed_entity,
                'entity_pivot': entity_pivot,
                'decision': decision,
                'reasoning': answer_content.get('reasoning', ''),
            }

            if decision == 'answer':
                answer_phase_history.append(answer_round_meta)
                print(f"Question {question_idx} - Answer turn {answer_turn + 1}: LM has enough information")
                break

            if not sub_questions_raw:
                answer_round_meta['decision'] = 'answer'
                answer_round_meta['reasoning'] += " (no new sub-questions generated, forcing answer)"
                answer_phase_history.append(answer_round_meta)
                print(f"Question {question_idx} - Answer turn {answer_turn + 1}: no sub-questions, forcing answer")
                break

            print(f"Question {question_idx} - Answer turn {answer_turn + 1}: asking {len(sub_questions_raw)} sub-questions")

            answer_sm_responses, answer_sm_responses_raw = await query_with_majority_vote(
                sm_client=sm_client,
                sub_questions_raw=sub_questions_raw,
                prompt_formatter_func=prepare_prompt_for_checking_mem_knowledge,
                sm_model_id=sm_model_id,
                temperature=sm_answer_temperature,
                max_concurrent=sm_max_concurrent,
            )
            print(f"Question {question_idx} - Answer turn {answer_turn + 1} SM responses (consolidated): {answer_sm_responses}")

            conversation_history.append({
                'round': len(conversation_history),
                'phase': 'answer_seeking',
                'sm_responses': answer_sm_responses
            })
            all_sm_responses_flat.extend(answer_sm_responses)

            answer_round_meta['sub_questions_raw'] = sub_questions_raw
            answer_round_meta['sm_responses'] = answer_sm_responses
            answer_round_meta['sm_responses_raw'] = answer_sm_responses_raw
            answer_phase_history.append(answer_round_meta)

        except Exception as e:
            print(f"Error in answer-seeking turn {answer_turn + 1} for question {question_idx}: {e}")
            break

    # -------------------------------------------------------------------------
    # FINAL: LM synthesizes answer from all collected SM responses
    # -------------------------------------------------------------------------
    try:
        all_sm_responses_filtered = filter_sm_responses(all_sm_responses_flat)
        final_prompt = format_LM_final_answer(
            all_sm_responses_filtered,
            question,
            accumulated_known_facts=accumulated_known_facts,
            accumulated_known_entities=[confirmed_entity] if confirmed_entity else [],
        )

        final_response = await lm_client.chat.completions.create(
            model=model_id,
            stream=stream,
            messages=[{"role": "user", "content": final_prompt}],
            max_tokens=max_new_tokens,
            temperature=lm_final_temperature,
            top_p=0.95,
            extra_body={"chat_template_kwargs": {"thinking_budget": thinking_budget}}
        )

        final_content = extract_json(final_response.choices[0].message.content) if lm_model_id else json.loads(final_response.choices[0].message.content)
        final_answer = final_content['final_answer']
        final_justification = final_content['justification']
        print(f"Question {question_idx} - Final answer: {final_answer}")

    except Exception as e:
        print(f"Error in final answer generation for question {question_idx}: {e}")
        return {
            "question_no": question_idx,
            "question": question,
            "grounding_subquestions": grounding_subquestions,
            "grounding_sm_responses": grounding_sm_responses,
            "entity_phase_history": entity_phase_history,
            "confirmed_entity": confirmed_entity,
            "entity_was_confirmed": entity_was_confirmed,
            "answer_phase_history": answer_phase_history,
            "all_sm_responses": all_sm_responses_flat,
            "groundtruth": row["groundtruth"],
            "model_final_response": "",
            "model_final_response_justification": "",
            "error": f"Final answer generation failed: {e}",
        }

    final_remaining_unknowns = (
        answer_phase_history[-1].get('remaining_unknowns', [])
        if answer_phase_history else []
    )

    return {
        "question_no": question_idx,
        "question": question,
        "grounding_subquestions": grounding_subquestions,
        "grounding_sm_responses": grounding_sm_responses,
        "entity_phase_history": entity_phase_history,
        "confirmed_entity": confirmed_entity,
        "entity_was_confirmed": entity_was_confirmed,
        "entity_identified": True,
        "num_entity_turns": len(entity_phase_history),
        "answer_phase_history": answer_phase_history,
        "num_answer_turns": len(answer_phase_history),
        "all_sm_responses": all_sm_responses_flat,
        "all_sm_responses_filtered": all_sm_responses_filtered,
        "final_known_facts": accumulated_known_facts,
        "final_remaining_unknowns": final_remaining_unknowns,
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
    sm_model_id=None,
    max_concurrent=5,
    sm_max_concurrent=100,
    lm_grounding_temperature=1.1,
    sm_grounding_temperature=0.7,
    lm_entity_temperature=1.0,
    sm_entity_temperature=0.7,
    lm_answer_temperature=1.0,
    sm_answer_temperature=0.7,
    lm_final_temperature=0.3,
    max_entity_turns=5,
    max_answer_turns=5,
    dead_end_threshold=2,
    skip_entity_pinning=False,
    max_retries=3,
):
    """Process all questions with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(row, idx):
        async with semaphore:
            for attempt in range(1, max_retries + 1):
                result = await process_single_question(
                    lm_client, sm_client, row, idx,
                    max_new_tokens, thinking_budget, stream,
                    lm_model_id=lm_model_id,
                    sm_model_id=sm_model_id,
                    lm_grounding_temperature=lm_grounding_temperature,
                    sm_grounding_temperature=sm_grounding_temperature,
                    lm_entity_temperature=lm_entity_temperature,
                    sm_entity_temperature=sm_entity_temperature,
                    lm_answer_temperature=lm_answer_temperature,
                    sm_answer_temperature=sm_answer_temperature,
                    lm_final_temperature=lm_final_temperature,
                    max_entity_turns=max_entity_turns,
                    max_answer_turns=max_answer_turns,
                    dead_end_threshold=dead_end_threshold,
                    sm_max_concurrent=sm_max_concurrent,
                    skip_entity_pinning=skip_entity_pinning,
                )
                if result is not None and 'error' not in result:
                    return result
                print(f"Question {idx} - attempt {attempt}/{max_retries} failed: {result['error']}. {'Retrying...' if attempt < max_retries else 'Giving up.'}")
            return result

    tasks = [process_with_semaphore(row, row['question_no']) for row in eval_questions]
    return await asyncio.gather(*tasks)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--thinking_budget", type=int, default=-1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--lm_port", type=int, default=4322)
    parser.add_argument("--lm_model_name", type=str, default=None, help="If set, routes LM calls to OpenRouter using this model name")
    parser.add_argument("--sm_port", type=int, default=4324)
    parser.add_argument("--msq_qns_path", type=str, default="")
    parser.add_argument("--max_num_questions", type=int, default=None, help="Limit to the first N questions")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--max_concurrent", type=int, default=100)
    parser.add_argument("--sm_max_concurrent", type=int, default=100)
    parser.add_argument("--max_entity_turns", type=int, default=5)
    parser.add_argument("--max_answer_turns", type=int, default=5)
    parser.add_argument("--dead_end_threshold", type=int, default=2)
    parser.add_argument("--skip_entity_pinning", action="store_true", help="Skip Phase 1 entity pinning and go directly to answer seeking")
    parser.add_argument("--lm_grounding_temperature", type=float, default=1.1)
    parser.add_argument("--sm_grounding_temperature", type=float, default=0.7)
    parser.add_argument("--lm_entity_temperature", type=float, default=1.0)
    parser.add_argument("--sm_entity_temperature", type=float, default=0.7)
    parser.add_argument("--lm_answer_temperature", type=float, default=1.0)
    parser.add_argument("--sm_answer_temperature", type=float, default=0.7)
    parser.add_argument("--lm_final_temperature", type=float, default=0.3)
    parser.add_argument("--eval_only", action="store_true", help="Skip process_all_questions and load output_path directly for DeepEval")
    parser.add_argument("--max_retries", type=int, default=3, help="Number of retry attempts for failed questions")
    args = parser.parse_args()

    if args.max_entity_turns < 1 or args.max_answer_turns < 1:
        raise ValueError("--max_entity_turns and --max_answer_turns must both be >= 1")

    if not args.eval_only:
        import httpx
        lm_http_limits = 50 if args.lm_model_name else 1000
        _lm_http = httpx.AsyncClient(limits=httpx.Limits(max_connections=lm_http_limits, max_keepalive_connections=200))
        _sm_http = httpx.AsyncClient(limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200))

        if args.lm_model_name:
            from dotenv import load_dotenv
            load_dotenv()
            lm_client = AsyncOpenAI(
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                api_key=os.getenv("OPENROUTER_API_KEY", "dummy"),
                timeout=120.0,
                http_client=_lm_http,
            )
            lm_model_id = args.lm_model_name
        else:
            lm_client = AsyncOpenAI(base_url=f"http://localhost:{args.lm_port}/v1", api_key="dummy", timeout=120.0, http_client=_lm_http)
            lm_model_id = None
        sm_client = AsyncOpenAI(base_url=f"http://localhost:{args.sm_port}/v1", api_key="dummy", timeout=120.0, http_client=_sm_http)

        if lm_model_id is None:
            lm_model_id = (await lm_client.models.list()).data[0].id
        sm_model_id = (await sm_client.models.list()).data[0].id
        print(f"LM model: {lm_model_id}")
        print(f"SM model: {sm_model_id}")

    if args.eval_only:
        print(f"--eval_only: loading existing output from {args.output_path}")
        with open(args.output_path, "r", encoding="utf-8") as f:
            output = json.load(f)
        print(f"Loaded {len(output)} results")
    else:
        eval_questions = load_questions(args.msq_qns_path, args.max_num_questions)

        output = await process_all_questions(
            lm_client, sm_client, eval_questions,
            args.max_new_tokens, args.thinking_budget, args.stream,
            lm_model_id=lm_model_id,
            sm_model_id=sm_model_id,
            max_concurrent=args.max_concurrent,
            sm_max_concurrent=args.sm_max_concurrent,
            lm_grounding_temperature=args.lm_grounding_temperature,
            sm_grounding_temperature=args.sm_grounding_temperature,
            lm_entity_temperature=args.lm_entity_temperature,
            sm_entity_temperature=args.sm_entity_temperature,
            lm_answer_temperature=args.lm_answer_temperature,
            sm_answer_temperature=args.sm_answer_temperature,
            lm_final_temperature=args.lm_final_temperature,
            max_entity_turns=args.max_entity_turns,
            max_answer_turns=args.max_answer_turns,
            dead_end_threshold=args.dead_end_threshold,
            skip_entity_pinning=args.skip_entity_pinning,
            max_retries=args.max_retries,
        )

        print(f"Processed {len(output)} questions successfully")

        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)

    # -------------------------------------------------------------------------
    # DeepEval scoring
    # -------------------------------------------------------------------------
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

    question_to_no = {
        item['question']: item.get('question_no', '')
        for item in output
        if item is not None
    }

    data_pairs = [
        {
            'question': item['question'],
            'ans1': item.get('model_final_response', 'No answer'),
            'ans2': item['groundtruth']
        }
        for item in output
        if item is not None
    ]

    eval_results = run_evaluation_multi_answer(
        client=judge_client,
        client_model_name=judge_model_name,
        data_pairs=data_pairs,
        run_id="eval-msq-multi-turn-v2"
    )

    print("\n" + "="*80)
    print("DeepEval evaluation completed!")
    eval_output = []
    for res in eval_results:
        _, results = res
        for test_result in results:
            try:
                metric = test_result.metrics_data[0]
                question_no = question_to_no.get(test_result.input, '')
                entry = {
                    "question_no": question_no,
                    "score": metric.score,
                    "input": test_result.input,
                    "actual_output": test_result.actual_output,
                    "expected_output": test_result.expected_output,
                    "reason": metric.reason
                }
                eval_output.append(entry)
            except Exception as e:
                print(f"Warning: failed to extract metric for a test result: {e}")

        break

    from collections import defaultdict

    def _get_hop_type(question_no: str) -> str:
        match = re.match(r'(\d+hop)', str(question_no))
        return match.group(1) if match else 'unknown'

    hop_correct: dict = defaultdict(int)
    hop_total: dict = defaultdict(int)
    for entry in eval_output:
        hop = _get_hop_type(entry.get('question_no', ''))
        hop_total[hop] += 1
        if entry.get('score', 0.0) == 1.0:
            hop_correct[hop] += 1

    hop_accuracy = {
        hop: {
            "correct": hop_correct[hop],
            "total": hop_total[hop],
            "accuracy": hop_correct[hop] / hop_total[hop] if hop_total[hop] > 0 else 0.0,
        }
        for hop in sorted(hop_total.keys())
    }

    total_evaluated = len(eval_output)
    total_correct = sum(1 if item.get('score', 0.0) == 1.0 else 0 for item in eval_output)
    deepeval_output = {
        "summary": {
            "total_evaluated": total_evaluated,
            "total_correct": total_correct,
            "overall_accuracy": total_correct / total_evaluated if total_evaluated > 0 else 0.0,
            "hop_accuracy": hop_accuracy,
        },
        "logged_output": eval_output
    }

    deepeval_summary_path = args.output_path.replace(".json", "deepeval_summary.json")
    with open(deepeval_summary_path, 'w', encoding='utf-8') as h:
        json.dump(deepeval_output, h, indent=2)
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
