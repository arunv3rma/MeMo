def _render_facts_by_entity(facts: list) -> str:
    """Render entity-tagged facts grouped by entity for LM consumption.
    Accepts list of {fact, entity} dicts (new format) or plain strings (legacy).
    """
    grouped = {}
    for f in facts:
        if isinstance(f, dict):
            entity = f.get('entity') or 'unknown'
            fact = f.get('fact', '')
        else:
            entity = 'unknown'
            fact = str(f)
        grouped.setdefault(entity, []).append(fact)
    lines = []
    for entity, entity_facts in grouped.items():
        lines.append(f"  [{entity}]:")
        for fact in entity_facts:
            lines.append(f"    - {fact}")
    return "\n".join(lines)


def _compute_dead_ends(conversation_history: list, dead_end_threshold: int) -> list:
    """
    Returns question strings that have received "I don't know" at least
    dead_end_threshold times. Used by the entity-pinning and answer-seeking prompts.
    """
    fail_counts: dict[str, int] = {}
    for entry in conversation_history:
        for qa in entry.get('sm_responses', []):
            if "i don't know" in qa['answer'].lower():
                fail_counts[qa['question']] = fail_counts.get(qa['question'], 0) + 1
    return [q for q, count in fail_counts.items() if count >= dead_end_threshold]


## v2, added the requirement of Complete and answerable for subquestions
def generate_prompt_from_LM_to_SM_general_v1(original_question):
    prompt = f"""You are a large language model with access to a memory bank containing long-form content (academic papers, transcripts, reports, documentation, etc.).

    Your task is to answer the following original question:
    "{original_question}"

    Follow these steps when generating your output:

    1. Determine whether the memory bank is needed:
    - If the question can be answered fully using general knowledge, set "requires_memory_bank" to false.
    - If the question requires specific information from the stored content, set "requires_memory_bank" to true.

    2. If "requires_memory_bank" is true:
    - First, identify what information is needed to fully answer the original question
    - Then, break that down into as many sub-questions as required to retrieve that information from the memory bank

    **Requirements for sub-questions:**
    - **Self-contained**: Each question must be independently understandable without prior context. Use complete terms instead of pronouns (it, they, this, that) or blank placeholders. Include all necessary information within the question itself.
    - **Comprehensive**: Err on the side of asking more questions rather than fewer. The goal is to surface as much relevant information from the memory bank as possible to fully answer the original question.
    - **Need-driven**: Pose questions that directly target what is required to answer the original question — as many as needed to cover it thoroughly.
    - **Complete and answerable**: Avoid fill-in-the-blank formats, placeholder symbols, or incomplete phrasing. Every question should be fully formed and answerable by someone encountering the topic for the first time.

    **Bad practices (avoid these):**
    - Using pronouns: "What are its benefits?" → "What are the benefits of [specific approach]?"
    - Too vague: "What was said about it?" → "What points were made about [specific topic]?"
    - Compound: "What is X and why does it matter?" → Split into two questions

    3. Always provide a final answer:
    - If no memory bank is required, answer directly from general knowledge.
    - If the memory bank is required, synthesize the final answer using the sub-question answers.

    Return your response in the following strict JSON format:

    {{
        "requires_memory_bank": <true/false>,
        "sub_questions": [
            <question1>,
            <question2>,
            <question3>
        ],
        "final_answer": "<your final answer to the original question>"
    }}

    Do not return anything else.
    """
    return prompt


def generate_grounding_subquestions(question: str) -> str:
    return f"""You are given a question containing clues that point to a specific entity or event. Break down these clues into open-ended, candidate-seeking questions for entity identification.

    Question: {question}

    STEP 1 — CLUE EXTRACTION:
    List every distinct, atomic clue in the question. Each clue must express exactly one constraint. Rules:
    - Split clues containing two independent facts into separate clues
    - Never skip clues, especially specific or unusual ones
    - Never fabricate or infer details not explicitly stated in the question
    - Never assume timing, ordinals, or causality unless explicitly stated
    - Do not include the final question being asked as a clue
    - Bundling errors to avoid — do not combine:
    - Two events at different points in time
    - An action with its consequence
    - A personal attribute with an unrelated action or event
    - If unsure, ask: "Could these facts appear in different sentences in a biography?" If yes, split them

    STEP 1B — CLUE TYPE CLASSIFICATION:
    For each clue from Step 1, classify it as one of:
    - (A) entity-narrowing: a fact used to identify who or what the question is about. These clues constrain the search space and narrow it to specific candidates.
    - (B) answer-carrying: a fact that, if retrieved directly, would yield the final answer to the original question. These are clues that ARE the answer (e.g., "what grade was X in", "what award did X win", "what was X's role").

    Most clues will be type (A). A clue is type (B) only if asking it directly would expose the exact final answer the original question seeks — not just another identifying attribute.

    STEP 2 — CLUE ASSESSMENT:
    Rate each clue's narrowing power independently:
    - HIGH: Specific or unusual clues (unique naming conventions, rare personal events,
    precise numerical details). Never omit these
    - LOW: Generic clues describing large populations (common professions, widely held
    qualifications, popular hobbies). May be deprioritized if they add little
    discriminative value

    Assign canonical question counts:
    - HIGH clues: 3 canonical questions (from meaningfully different angles — person-first, event-first, action-first)
    - LOW clues: 1 canonical question

    Each canonical question will also receive exactly 3 paraphrase variations in the Step 5 output, used for independent SM querying and majority voting.

    STEP 3 — QUESTION GENERATION:
    Generate the assigned number of canonical questions per clue. For each canonical question, also generate exactly 3 paraphrase variations (different phrasings of the same core question, preserving all constraints). Apply different rules based on clue type:

    For type (A) entity-narrowing clues:
    Canonical questions:
    - Must approach the clue from meaningfully different angles (e.g. person-first, event-first, action-first)
    - Must not be superficially different (no synonym swaps or word reordering)

    Construction:
    - Use concrete values only — every name, location, date, and detail from the clue must appear verbatim
    - Describe unknown entities using their known attributes only — never use labels like "Entity A" or "Person X"
    - Never use pronouns or vague references
    - Frame as candidate-seeking, not direct lookups: "Who are people that...?" not "Who is X?"
    - State all constraints explicitly so invalid answers are immediately identifiable
    - Keep questions short and factual
    - Anchor to the domain implied by the original question where relevant
    - Never ask a question whose answer would directly reveal the final entity
    - Each question must be self-contained within its own clue only. Never reference
    constraints from other clues — every constraint in the question must trace back
    to the single clue being tested

    For type (B) answer-carrying clues:
    - Generate direct retrieval questions (not candidate-seeking) — these are allowed to directly ask for the final answer because the answer is itself an atomic retrievable fact, not a synthesis.
    - Embed as much entity-identifying context from other clues as possible so the SM can locate the right entity.
    - Example: instead of "What grade was the child in?", ask "What grade was the oldest child of [entity description with all known attributes] in as of [date]?"
    - These questions should be phrased as "What [attribute] did/was [entity description]...?" to elicit a specific factual answer.

    Paraphrase variations:
    - For each canonical question, provide exactly 3 paraphrase variations
    - Paraphrases are different phrasings of the same core question — each will be sent to the SM independently for majority voting
    - Paraphrases must preserve all constraints from the canonical question

    Example of valid canonical + paraphrases for "founded a company in Singapore in 1998":
    - canonical: "Which entrepreneurs founded a company in Singapore in 1998?"
    - paraphrase 1: "What companies were established in Singapore in 1998 and who were their founders?"
    - paraphrase 2: "Who are individuals that started a business in Singapore during 1998?"
    - paraphrase 3: "Name entrepreneurs who launched a company in Singapore in 1998."

    STEP 3B — COMBINATION QUESTIONS:
    After generating single-clue canonical questions, generate combination questions that pair HIGH-narrowing clues from Step 2 across different constraint types (e.g. time + role, time + action, location + role).

    Rules:
    - Every combination must include at least one HIGH-narrowing clue
    - Only pair clues from different constraint types — do not combine two time constraints or two role constraints
    - Cap each combination at 2–3 clues maximum — beyond that SM answers become unreliable
    - Generate 1 canonical combination question per eligible HIGH clue pair, each with exactly 3 paraphrase variations
    - Combination questions should seek candidates satisfying all combined constraints simultaneously
    - Label each combination question with its source clue indices (zero-based, from the ordered list in Step 1)

    STEP 4 — VERIFICATION:
    Check each canonical question and all its paraphrases against the following. Rewrite or remove as needed:
    1. Contains placeholders, pronouns, vague references, or anonymized labels? → Rewrite
    2. Missing constraints from the clue? → Add them
    3. Tests more than one clue? Watch for "and"/"then", two time periods, or mixed state+action → Split (applies to single-clue questions only; combination questions intentionally span multiple clues)
    4. Variations not meaningfully distinct? → Rewrite
    5. Contains inferred details not verbatim in the original question? → Remove. Watch for assumed timing, ordinals, and causality
    6. Borrows constraints from another clue to create variation? → Remove borrowed constraints (single-clue questions only)
    7. Circular or self-answering? → Rewrite to genuinely seek candidates
    8. Does this question reference constraints belonging to a different clue? Each
    clue's questions must be entirely self-contained — constraints from one clue
    must never appear in the questions of another clue. To check: identify every
    constraint in the question and trace each one back to the single clue being
    tested. If any constraint belongs to a different clue, remove it immediately.
    A clue's questions must stand alone without borrowing context from any other
    clue in the list

    STEP 5 — OUTPUT:
    Return only this JSON object with no intermediate reasoning or notes:
    {{
        "grounding_questions": [
            {{
                "question": "canonical question text",
                "paraphrases": ["paraphrase 1", "paraphrase 2", "paraphrase 3"],
                "question_type": "single",
                "clue_indices": [0],
                "clues_involved": 1
            }},
            {{
                "question": "combination canonical question",
                "paraphrases": ["combo paraphrase 1", "combo paraphrase 2", "combo paraphrase 3"],
                "question_type": "combination",
                "clue_indices": [0, 2],
                "clues_involved": 2
            }}
        ]
    }}"""


def generate_entity_pinning_prompt(
    question: str,
    conversation_history: list,
    turns_remaining: int,
    dead_end_threshold: int = 2,
    accumulated_known_facts: list = None,
    candidate_entities: list = None,
    is_first_turn: bool = False,
    candidate_idk_streaks: dict = None,
) -> str:
    """
    Phase 1 prompt: focused exclusively on identifying which entity or event
    the original question is about. Does NOT ask the final question.
    Should return JSON with candidate_entities, entity_confirmed, sub_questions.
    """
    accumulated_known_facts = accumulated_known_facts or []
    candidate_entities = candidate_entities or []

    history_lines = []
    for i, entry in enumerate(conversation_history):
        is_latest = (i == len(conversation_history) - 1)
        label = f"--- Round {entry['round']} ({entry.get('phase', 'unknown')}) (most recent) ---" if is_latest else f"--- Round {entry['round']} ({entry.get('phase', 'unknown')}) ---"
        history_lines.append(label)
        raw_lm = entry.get('raw_lm_response')
        if raw_lm:
            history_lines.append(f"  [NOTE: The LM response for this round could not be parsed as JSON. Raw LM output is shown below for context:]")
            history_lines.append(f"  {raw_lm.strip()}")
        for qa in entry.get('sm_responses', []):
            answered = "I DON'T KNOW" if "i don't know" in qa['answer'].lower() else qa['answer']
            history_lines.append(f"  Q: {qa['question']}")
            history_lines.append(f"  A: {answered}")
        history_lines.append("")
    history_str = "\n".join(history_lines).strip()

    if accumulated_known_facts:
        facts_lines = _render_facts_by_entity(accumulated_known_facts)
        known_facts_section = f"\n---\n\nKNOWN FACTS (grouped by entity — check that facts attributed to the same entity are consistent with one another):\n{facts_lines}\n"
    else:
        known_facts_section = ""

    if candidate_entities:
        cand_lines = "\n".join(
            f"  - [{c.get('rank', '?')}] {c.get('name', '?')} — {c.get('confidence_note', '')}"
            for c in sorted(candidate_entities, key=lambda c: c.get('rank') if c.get('rank') is not None else 999)
        )
        candidates_section = f"""
---

CURRENT CANDIDATE ENTITIES (ranked by prior assessment — update these based on new evidence):
{cand_lines}
"""
    elif is_first_turn:
        candidates_section = """
---

CURRENT CANDIDATE ENTITIES: None yet.

IMPORTANT — This is the first entity-pinning turn. Before building the candidate list, scan ALL answers in the conversation history (especially the Round 0 grounding responses) for:
1. Named individuals, places, organisations, works (books, films, songs, companies), or events mentioned in any non-IDK answer — these are explicit candidate signals.
2. Entities implied by the names of organisations, projects, awards, or works mentioned in any answer. For example, if an answer references a named prize, an organisation bearing a person's name, or a project associated with a specific location, the person or place embedded in that name may itself be the entity being sought.
List all such items as provisional candidates and rank them by how many clues from the original question they appear to satisfy.
"""
    else:
        candidates_section = ""

    dead_ends = _compute_dead_ends(conversation_history, dead_end_threshold)
    if dead_ends:
        dead_end_lines = "\n".join(f"  - {q}" for q in dead_ends)
        dead_end_section = f"""
---

DEAD ENDS — the following questions have received "I DON'T KNOW" {dead_end_threshold} or more times. Do NOT ask these again or generate close reformulations of them:

{dead_end_lines}
"""
    else:
        dead_end_section = ""

    candidate_idk_streaks = candidate_idk_streaks or {}
    if candidate_idk_streaks:
        streak_lines = "\n".join(
            f"  - {name}: {count} targeted IDK(s)"
            for name, count in sorted(candidate_idk_streaks.items(), key=lambda x: -x[1])
        )
        candidate_streak_section = f"""
---

CANDIDATE IDK STREAKS (count of questions specifically targeting each candidate that returned "I DON'T KNOW" — use these when applying decay rules in Step 2):
{streak_lines}
"""
    else:
        candidate_streak_section = ""

    turns_warning = (
        f"\nIMPORTANT: You have {turns_remaining} turn(s) remaining in the entity-pinning phase. "
        f"If turns_remaining is 0, you MUST set decision to \"exhaust\" regardless of confidence."
        if turns_remaining <= 1
        else f"\nYou have {turns_remaining} turn(s) remaining in the entity-pinning phase."
    )

    return f"""You are helping answer a complex question by first identifying exactly which entity or event the question is about.

Your ONLY goal in this phase is to determine the identity of the entity. Do NOT attempt to answer the original question itself — that happens in a later phase once the entity is confirmed.{turns_warning}
{known_facts_section}{candidates_section}{dead_end_section}{candidate_streak_section}
---

Original Question:
{question}

---

Conversation History (all rounds so far):
{history_str}

---

Follow these steps:

Step 1 — Evaluate the most recent SM responses for entity evidence.
For each response in the most recent round:
- Does it name or strongly imply a specific entity that matches one or more clues in the original question?
- Does it confirm or contradict any clue constraint (time period, location, role, quantity)?
- Does it support, weaken, or eliminate any candidate from the current list?
Assign each response: "confirms candidate" (directly supports a known candidate), "new candidate" (surfaces a previously unknown candidate), "eliminates candidate" (rules one out), or "no signal" (irrelevant or I DON'T KNOW).

Step 2 — Update the candidate list.
Starting from the CURRENT CANDIDATE ENTITIES above, incorporate evidence from the most recent round:
- Promote candidates whose clue coverage increased
- Demote or eliminate candidates that fail a hard constraint
- Add any newly surfaced candidates
- Assign each candidate an integer rank (1 = most likely) and a brief confidence_note explaining the ranking
- If two candidates are genuinely indistinguishable at this point, they may share rank 1 — but only do this if there is real ambiguity, not as a default
- Cross-clue temporal consistency: before treating a mismatch as disqualifying, check whether it is simply an expected change over time. The original question may describe the same entity at multiple points in time under different attributes (e.g. different ages, roles, locations, or titles). If a candidate matches one clue, compute what their attributes would be at the time of each other clue and check whether those derived attributes are consistent — do NOT disqualify a candidate solely because their attribute at clue A differs from their attribute at clue B if the difference is explainable by the passage of time or a natural progression. For example: a candidate confirmed as age 14 at one clue is expected to be age 16 roughly two years later, and age 13 roughly one year before — treat this as supporting evidence for those other clues, not as a contradiction.
- Apply IDK-streak decay using the CANDIDATE IDK STREAKS section above: if a candidate has 2 or more targeted IDKs with no confirming signal across all rounds, demote it by 2 rank positions. If it has 3 or more targeted IDKs with zero confirmation, eliminate it from the candidate list entirely and add its name to the `eliminated_candidates` field. Zero confirmation means no SM response has ever directly named or implied this candidate in relation to a clue from the original question.

Step 3 — Decide: confirmed, ambiguous, or continue?
- Set entity_confirmed = true if one candidate clearly satisfies more clues than all others and no hard constraint disqualifies it. When true, set confirmed_entity to that candidate's exact name and set primary_candidate likewise.
- Set primary_candidate to the rank-1 candidate's name even when entity_confirmed = false, so the next phase has a best guess. If two candidates share rank 1, set primary_candidate = null and ambiguous = true.
- Set decision = "exhaust" if you have no further useful entity-identification questions to ask (all angles dead-ended or fully explored), even if entity_confirmed = false.
- Set decision = "continue" if there are still productive entity-identification angles to try.

Step 4 — If continuing, generate new entity-identification sub-questions.
- Questions must target entity identification only — do NOT ask the final question from the original question
- Do NOT ask about dead ends listed above
- Use known facts and clue constraints to make questions targeted
- For candidates still in contention, generate verification questions that test their remaining unconfirmed clues
- All questions must be fully self-contained with no pronouns or vague references
- Questions must always be open and candidate-seeking — phrased to invite a list of possible answers, not to confirm a specific one. Even at the simplest level, ask "What is the name of the person who [fact]?" or "Who are people that [fact]?" — never "Did [candidate] do [fact]?"
- Assign each question a complexity level based on how many independent facts from the original question it contains:
    - "single-fact": one attribute or clue constrains the answer
    - "two-fact": two independent attributes combined
    - "multi-fact": three or more attributes combined
- For any prior question that received "I DON'T KNOW" but is NOT yet a dead end, generate a simpler reformulation by dropping one fact and reducing the complexity level by one step (multi-fact → two-fact → single-fact). Keep the most distinctive remaining fact. Do NOT drop below single-fact — if a single-fact question has already failed, treat the topic as a dead end rather than simplifying further.
- For each sub-question, provide exactly 3 paraphrase variations in the "paraphrases" field. These are different phrasings of the same question — each will be sent to the SM independently and the majority answer across the 3 will be used, reducing the risk that a single unusual phrasing biases the result.

---

Return your response as a JSON object in the following format:
{{
    "assessment": "brief summary of which candidates are in contention and what evidence shifted since last round",
    "new_sm_response_evaluation": [
        {{
            "question": "the sub-question that was asked",
            "answer_summary": "brief summary",
            "signal": "confirms candidate | new candidate | eliminates candidate | no signal",
            "affects_candidate": "candidate name, or null"
        }}
    ],
    "known_facts": [
        {{"fact": "the established fact", "entity": "which entity this fact pertains to — use the exact candidate name, or a descriptive label if unnamed (e.g. 'hat shop owner', 'unknown person')"}},
        ...
    ],
    "candidate_entities": [
        {{
            "name": "candidate entity name or best description if unnamed",
            "rank": 1,
            "supporting_clues": ["which clues from the original question this candidate satisfies"],
            "disqualifying_clues": ["which clues this candidate fails, or empty list"],
            "confidence_note": "brief reason for this ranking"
        }}
    ],
    "primary_candidate": "name of rank-1 candidate, or null if genuinely ambiguous",
    "eliminated_candidates": ["candidates removed due to IDK streaks, or empty list"],
    "ambiguous": true,
    "entity_confirmed": true,
    "confirmed_entity": "exact name if entity_confirmed is true, otherwise null",
    "reasoning": "why entity is confirmed, still ambiguous, or exhausted",
    "decision": "continue | exhaust",
    "sub_questions": [
        {{
            "question": "fully self-contained entity-identification sub-question",
            "paraphrases": ["variation 1 of the same question", "variation 2", "variation 3"],
            "purpose": "which clue or candidate hypothesis this tests",
            "targets_candidate": "candidate name being verified, or null for open search",
            "complexity": "single-fact | two-fact | multi-fact",
            "is_simplification_of": "prior question this reformulates, or null",
            "simplified_from_complexity": "the complexity level of the question being simplified, or null",
            "is_pivot": true
        }}
    ]
}}

Rules:
- sub_questions must only be present when decision is "continue" AND entity_confirmed is false. If entity_confirmed is true or decision is "exhaust", omit sub_questions or set it to an empty list.
- You must only use information explicitly stated in the conversation history above. Do not draw on external knowledge or training data.
- A fact is more reliable when the SM's answer directly and unambiguously concerns the same entity described in the original question. A general fact that merely matches one clue is still useful signal — use it to raise a candidate's rank and guide further questions, while being cautious about treating it as ground truth until corroborated by a further entity-specific response. Do not discard such signal; weight it proportionally to how specific it is.
- Never generate a sub-question that closely paraphrases a dead end — pivots must approach the information from a meaningfully different direction.
- known_facts and candidate_entities in your response must be the complete updated lists — they replace the prior lists entirely.
- Each sub-question must have exactly 3 entries in "paraphrases" — rephrase the same core question three different ways while preserving all constraints.
- Return only the JSON object, no other text."""


def generate_answer_seeking_prompt(
    question: str,
    conversation_history: list,
    turns_remaining: int,
    dead_end_threshold: int = 2,
    accumulated_known_facts: list = None,
    confirmed_entity: str = None,
    entity_was_confirmed: bool = False,
    candidate_entities: list = None,
) -> str:
    """
    Phase 2 prompt: focused exclusively on answering the original question,
    given the confirmed (or best-candidate) entity from Phase 1.
    Every sub-question must embed the entity name directly.
    Should return JSON with decision, sub_questions, entity_pivot.
    """
    accumulated_known_facts = accumulated_known_facts or []
    candidate_entities = candidate_entities or []

    history_lines = []
    for i, entry in enumerate(conversation_history):
        is_latest = (i == len(conversation_history) - 1)
        label = f"--- Round {entry['round']} ({entry.get('phase', 'unknown')}) (most recent) ---" if is_latest else f"--- Round {entry['round']} ({entry.get('phase', 'unknown')}) ---"
        history_lines.append(label)
        raw_lm = entry.get('raw_lm_response')
        if raw_lm:
            history_lines.append(f"  [NOTE: The LM response for this round could not be parsed as JSON. Raw LM output is shown below for context:]")
            history_lines.append(f"  {raw_lm.strip()}")
        for qa in entry.get('sm_responses', []):
            answered = "I DON'T KNOW" if "i don't know" in qa['answer'].lower() else qa['answer']
            history_lines.append(f"  Q: {qa['question']}")
            history_lines.append(f"  A: {answered}")
        history_lines.append("")
    history_str = "\n".join(history_lines).strip()

    if entity_was_confirmed:
        entity_preamble = f"""The entity this question is about has been confirmed as: {confirmed_entity}

Every sub-question you generate MUST embed the name "{confirmed_entity}" directly. Never refer to it as "the entity", "they", "it", or any vague reference."""
    else:
        entity_preamble = f"""The entity-pinning phase did not reach full confidence. The best candidate identified is: {confirmed_entity}

Treat this as a working hypothesis. Every sub-question you generate MUST embed the name "{confirmed_entity}" directly. If answer-seeking questions for this candidate consistently return "I DON'T KNOW", consider pivoting to an alternative candidate via the entity_pivot field."""

    if accumulated_known_facts:
        facts_lines = _render_facts_by_entity(accumulated_known_facts)
        known_facts_section = f"\n---\n\nKNOWN FACTS (grouped by entity — verify that facts attributed to the same entity are mutually consistent before asking further questions):\n{facts_lines}\n"
    else:
        known_facts_section = ""

    if candidate_entities:
        cand_lines = "\n".join(
            f"  - [{c.get('rank', '?')}] {c.get('name', '?')} — {c.get('confidence_note', '')}"
            for c in sorted(candidate_entities, key=lambda c: c.get('rank') if c.get('rank') is not None else 999)
        )
        candidates_section = f"""
---

ALTERNATIVE CANDIDATES (available for pivot if current entity fails):
{cand_lines}
"""
    else:
        candidates_section = ""

    dead_ends = _compute_dead_ends(conversation_history, dead_end_threshold)
    if dead_ends:
        dead_end_lines = "\n".join(f"  - {q}" for q in dead_ends)
        dead_end_section = f"""
---

DEAD ENDS — the following questions have received "I DON'T KNOW" {dead_end_threshold} or more times. Do NOT ask these again:

{dead_end_lines}
"""
    else:
        dead_end_section = ""

    turns_warning = (
        f"\nIMPORTANT: You have {turns_remaining} turn(s) remaining. "
        f"If turns_remaining is 0, you MUST set decision to \"answer\" regardless of confidence."
        if turns_remaining <= 1
        else f"\nYou have {turns_remaining} turn(s) remaining."
    )

    return f"""You are helping answer a complex question. The entity identification phase is complete. Your ONLY goal now is to retrieve the information needed to answer the original question about the identified entity.{turns_warning}

{entity_preamble}
{known_facts_section}{candidates_section}{dead_end_section}
---

Original Question:
{question}

---

Conversation History (all rounds, most recent marked):
{history_str}

---

Follow these steps:

Step 0 — Classify the confirmed entity's role in the original question.
Before evaluating any SM responses, determine the logical role of "{confirmed_entity}" in the original question:
- **subject**: "{confirmed_entity}" is the entity performing the action or possessing the attribute being asked about. The final answer is a direct attribute of "{confirmed_entity}".
- **object**: "{confirmed_entity}" is the entity upon which an action was performed or about which a fact is incidentally true. The final answer is the name or attribute of the entity that acted upon or is related to "{confirmed_entity}" in the way the question specifies.
- **intermediate**: "{confirmed_entity}" is a stepping stone — the question requires first identifying "{confirmed_entity}", then finding a related person, work, or event whose attribute is the actual answer.

Also determine the `answer_subject`:
- If role is "subject": `answer_subject` = "{confirmed_entity}"
- If role is "object" or "intermediate": `answer_subject` = the name or description of the entity whose attribute is actually being asked about (e.g., "{confirmed_entity}'s sibling", "the person who [action] {confirmed_entity}")

All sub-questions must target the `answer_subject`. If the answer_subject is a related person or work, questions must anchor to "{confirmed_entity}" as a constraint but seek the related entity (e.g., "What is the name of {confirmed_entity}'s sibling who was a doctor and poet?" rather than "What is {confirmed_entity}'s middle name?").

Step 1 — Evaluate the most recent SM responses against established facts.
For each response in the most recent round:
- Does it directly answer or partially answer the original question about {confirmed_entity}?
- Is it consistent with the ESTABLISHED FACTS? Flag contradictions.
- Does it add new information beyond what is already established?
Assign each response: "high relevance" (directly answers part of the original question about the confirmed entity), "medium relevance" (adds useful context), "low relevance" (redundant or tangential), "contradiction" (conflicts with an established fact), or "non-responsive" (addresses the topic of the question but does not actually answer what was asked — e.g., question asks for a specific grade level and answer gives a general performance description, or question asks for a year and answer gives an era). A non-responsive answer must NOT be added to known_facts and must NOT be treated as satisfying a remaining unknown. Instead, treat it as a remaining unknown with `previously_asked: true` and generate a simplified reformulation targeting the specific fact type that was not returned.

Step 2 — Update the known facts.
Incorporate any new high or medium relevance facts. Exclude contradictions.

Step 3 — Identify what is still needed to answer the original question.
List every piece of information still missing. For each, note whether it has been asked before and whether it is a dead end.
Additionally: if the confirmed entity has been asked about repeatedly and consistently returns "I DON'T KNOW", consider whether a pivot to an alternative candidate is warranted.

Step 4 — Decide whether to continue or answer.
- CRITICAL: Before generating any new sub-questions, first ask: can the original question already be answered from the known facts and collected SM responses? If yes, you MUST choose "answer" immediately — do not generate additional sub-questions for information you already possess.
- Choose "answer" if the known facts are sufficient to answer the original question with reasonable confidence.
- Choose "answer" if all remaining unknowns are dead ends with no viable alternative angle.
- Choose "continue" ONLY if there is specific information that (a) is still missing, (b) has not already been retrieved, and (c) is required to answer the original question — not just generally interesting context.
- You MUST choose "answer" if turns_remaining is 0.

Step 5 — If continuing, generate answer-seeking sub-questions.
- EVERY sub-question MUST embed "{confirmed_entity}" (or the entity_pivot target) by name — never use pronouns or vague references
- Cross-incident entity linking: check whether {confirmed_entity} might also appear under a different description in another part of the original question (different time period, role, location, or name variation). If so, generate sub-questions anchored to {confirmed_entity} for those other clues directly
- Do NOT ask about dead ends
- Do NOT ask entity-identification questions — the entity is already known
- All sub-questions must be fully self-contained
- Questions must always be open and candidate-seeking — phrased to invite a descriptive answer, not to confirm a yes/no. Even when the entity is known, ask "What [attribute] did {confirmed_entity} [action]?" not "Did {confirmed_entity} [action]?", so the answer space remains open and matches how the memory bank QA pairs are indexed.
- Assign each question a complexity level based on how many independent constraints it contains beyond the entity name itself:
    - "single-fact": asks about one attribute of {confirmed_entity} with no additional constraining facts
    - "two-fact": asks about one attribute with one additional constraining fact (e.g. a time period or location)
    - "multi-fact": asks about one attribute with two or more additional constraining facts
- For any prior question that received "I DON'T KNOW" but is NOT yet a dead end, generate a simpler reformulation by removing one constraining fact and reducing the complexity level by one step (multi-fact → two-fact → single-fact). Do NOT drop below single-fact — if a single-fact answer-seeking question has already failed, treat the topic as a dead end rather than simplifying further.
- For each sub-question, provide exactly 3 paraphrase variations in the "paraphrases" field. These are different phrasings of the same question — each will be sent to the SM independently and the majority answer across the 3 will be used, reducing the risk that a single unusual phrasing biases the result.

---

Return your response as a JSON object in the following format:
{{
    "assessment": "brief summary of what has been established about {confirmed_entity} and what is still blocking the answer",
    "entity_role": "subject | object | intermediate",
    "answer_subject": "the entity whose attribute is the actual answer — either {confirmed_entity} or a description of the related entity",
    "new_sm_response_evaluation": [
        {{
            "question": "the sub-question that was asked",
            "answer_summary": "brief summary",
            "relevance": "high relevance | medium relevance | low relevance | contradiction | non-responsive",
            "contradiction_detail": "if contradiction, explain; if non-responsive, explain what specific fact type was missing; otherwise null"
        }}
    ],
    "known_facts": [
        {{"fact": "the fact text", "entity": "which entity this fact pertains to — use the exact confirmed entity name or a descriptive label"}},
        ...
    ],
    "remaining_unknowns": [
        {{
            "unknown": "what is still needed",
            "previously_asked": true,
            "is_dead_end": true,
            "alternative_angle": "different angle to try, or null"
        }}
    ],
    "entity_pivot": null,
    "decision": "continue | answer",
    "reasoning": "why continuing or answering, addressing any dead ends or pivot",
    "sub_questions": [
        {{
            "question": "fully self-contained question with {confirmed_entity} name embedded directly",
            "paraphrases": ["variation 1 of the same question", "variation 2", "variation 3"],
            "purpose": "what this is trying to establish",
            "uses_known_fact": "the known fact constraining this question, or null",
            "complexity": "single-fact | two-fact | multi-fact",
            "is_simplification_of": "prior question this reformulates, or null",
            "simplified_from_complexity": "the complexity level of the question being simplified, or null",
            "is_pivot": true
        }}
    ]
}}

Rules:
- sub_questions must only be present when decision is "continue". If decision is "answer", omit sub_questions or set it to an empty list.
- entity_pivot must be the exact name of an alternative candidate from the ALTERNATIVE CANDIDATES list above, or null. Only set it when answer-seeking for the current entity has genuinely stalled.
- You must only use information explicitly stated in the conversation history above. Do not draw on external knowledge or training data.
- Every sub-question must contain "{confirmed_entity}" (or the entity_pivot name) verbatim — this is a hard requirement.
- known_facts in your response must be the complete updated list — it replaces the prior list entirely.
- Each sub-question must have exactly 3 entries in "paraphrases" — rephrase the same core question three different ways while preserving all constraints.
- Return only the JSON object, no other text."""


def format_LM_final_answer(
    all_sm_responses: list,
    original_question: str,
    accumulated_known_facts: list = None,
    accumulated_known_entities: list = None,
) -> str:
    """
    Prompt for the LM to synthesize a final answer from the full conversation history.
    """
    accumulated_known_facts = accumulated_known_facts or []
    accumulated_known_entities = accumulated_known_entities or []

    qa_lines = []
    for qa in all_sm_responses:
        qa_lines.append(f"Q: {qa['question']}\nA: {qa['answer']}")
    qa_str = "\n\n".join(qa_lines)

    if accumulated_known_facts:
        facts_str = _render_facts_by_entity(accumulated_known_facts)
        known_facts_block = f"""
### Established Facts (grouped by entity — anchor your constraint reasoning to these; facts under the same entity label must be mutually consistent):
{facts_str}
"""
    else:
        known_facts_block = ""

    if accumulated_known_entities:
        entities_str = "\n".join(f"  - {e}" for e in accumulated_known_entities)
        known_entities_block = f"""
### Identified Entities (confirmed named entities — responses about these should be weighted more highly):
{entities_str}
"""
    else:
        known_entities_block = ""

    return f"""You are a large language model with access to a memory bank.

    Your task is to synthesize a final answer to the following question:
    "{original_question}"
{known_facts_block}{known_entities_block}
    You have been provided with the following question-answer pairs retrieved from the memory bank across multiple rounds of retrieval:
    {qa_str}

    ### Instructions:
    1. Use the Established Facts and Identified Entities above as your highest-confidence evidence. Anchor your answer to these first.
    2. Identify every candidate entity or answer that appears in the retrieved Q&A pairs. For each candidate, explicitly check how many of the distinct constraints in the original question it satisfies. A constraint is a specific clue: a time period, a location, a role, a quantity, a relationship, etc. Do this reasoning step-by-step before selecting an answer.
    3. Select the candidate that satisfies the greatest number of constraints from the original question — not the one that appears most frequently in the retrieved answers. Frequency of mention is not a reliable signal; constraint coverage is.
    4. When evaluating the retrieved question-answer pairs, give more weight to responses that directly concern an Identified Entity or that are consistent with the Established Facts.
    5. If a retrieved answer contradicts an Established Fact, prefer the Established Fact and note the contradiction in your justification.
    6. When the memory bank content contains domain-specific keywords or technical terms that are essential to the answer, incorporate them to maintain technical accuracy.
    7. If the memory bank content is insufficient, you may supplement with general knowledge, but clearly separate this from the memory-based information.
    8. Always connect your final answer back to the original question.
    9. Provide a justification that explains your constraint-by-constraint reasoning, which candidate best satisfies the original question, and any contradictions encountered.

    ### Output Format (strict JSON):
    {{
        "final_answer": "<your best possible synthesized answer to the original question>",
        "justification": "<step-by-step constraint reasoning across candidates, then the final selection with explanation>"
    }}
    """
