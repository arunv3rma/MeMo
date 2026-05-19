import re


def extract_doc_metadata(text):
    # Find all key-value pairs
    pattern = r"(\w+):\s*(.+)"
    matches = re.findall(pattern, text)

    metadata = {key: value.strip() for key, value in matches}
    # print(metadata)
    title = metadata.get('title', 'No title provided in document')
    author = metadata.get('author', 'No author provided in document')
    date = metadata.get('date', 'No date provided in document')

    # metadata = f"""TITLE:{title}\nAUTHOR:{author}\nDATE:{date}"""
    date = metadata.get('date', 'No date provided in document')
    return date
    # return metadata



def prepare_prompt_for_indirect_fact_extraction(chunk_content, doc_extracted_date):
    prompt = f"""
    You are tasked with extracting indirect, self-contained QA pairs from a given document. Indirect facts are facts that require combining, resolving, or deriving information from multiple parts of the document — they are not explicitly stated in a single sentence.

    ## INPUT FORMAT
    <document>
    {chunk_content}
    </document>

    <document_metadata>
    Document date: {doc_extracted_date}
    </document_metadata>

    Use the document date from metadata for temporal grounding throughout your QA pairs.

    ---

    ## STEP 1: DOCUMENT ANALYSIS
    - Read the entire document carefully
    - Note the document date — use it as your reference point for all temporal grounding
    - Identify ALL entities mentioned — major or minor, named or unnamed (people, organizations, places, dates, numbers, products, roles, events, concepts specific to the document)
    - Identify pronoun references and note what entity they refer to
    - Identify possessive relationships (e.g. "her company", "his report", "the firm's CEO")
    - Identify action chains where one person's action leads to another's outcome
    - Identify relative time expressions that need to be resolved to absolute dates
    - Identify facts that are only knowable by combining two or more separate statements

    ---

    ## STEP 2: IDENTIFY INDIRECT FACTS
    Indirect facts fall into the following categories. Work through each category systematically.

    ### 2a. Calculated Attributes
    Derive facts by combining two or more pieces of information:
    - Ages at specific events: birth year + event year → age at that event
    - Durations: start date + end date → length of time
    - Start dates: completion date + duration → when something began
    - Any other numerical derivation from two stated values

    ### 2b. Possessives & Ownership — Extract Bidirectionally
    When the document uses possessives, extract the relationship from BOTH directions:
    - "Gustave Eiffel's engineering firm" →
    - "What did Gustave Eiffel own?" → his engineering firm
    - "Who owned the engineering firm that built the Eiffel Tower?" → Gustave Eiffel
    - "Maria Chen's report" →
    - "What did Maria Chen author?" → the report
    - "Who authored the report?" → Maria Chen

    Always create both the forward and backward version of every possessive relationship.

    ### 2c. Pronoun Resolution
    Identify every pronoun in the document (he, she, it, they, the company, the organization, the team, the firm) and resolve it to the actual entity name. Extract the resolved fact as a QA pair.

    - "She founded the company in 1998" → who is "she"? → resolve and extract:
    - "Who founded [Company Name]?" → [Full Name]

    ### 2d. Temporal Calculations
    Resolve all relative time expressions using the document date ({doc_extracted_date}):
    - "last month" → calculate the specific month and year
    - "last year" → calculate the specific year
    - "two decades ago" → calculate the specific decade or year
    - "founded 10 years ago" → calculate the founding year
    - "the project took 3 years" + completion date → calculate start date

    ### 2e. Action Relationships & Enablement
    When one entity's action leads to another entity's outcome, extract the full relationship from ALL angles.

    For every action chain, identify:
    - **Enabler**: who performed the action
    - **Action**: what specific action was performed (use precise verbs: provided, delivered, transferred, presented, contributed, sent, gave, passed)
    - **Beneficiary**: who received or benefited
    - **Outcome**: what resulted from the action

    Then generate questions from ALL of these directions:

    | Direction | Example Question |
    |---|---|
    | Forward (enabler → beneficiary) | "Who did [Enabler] provide [Item] to?" |
    | Backward (beneficiary ← enabler) | "Who provided [Item] to [Beneficiary]?" |
    | Action | "What did [Enabler] provide to [Beneficiary]?" |
    | Outcome | "What did [Beneficiary] achieve after receiving [Item] from [Enabler]?" |
    | Sequence | "What did [Beneficiary] do after receiving [Item] from [Enabler]?" |

    **Look for these patterns in the document:**
    - "X provided/gave/sent Y to Z, who then achieved/completed..."
    - "X's contribution allowed/enabled Z to..."
    - "After X delivered Y, Z was able to..."
    - Sequential chains: A contributed → B received → B accomplished

    **Be specific with verbs — never vague:**
    - GOOD: "What did David Liu deliver to Maria Chen during TechStart's 2019 negotiations?" → The market analysis
    - GOOD: "Who received the market analysis from David Liu in 2019?" → Maria Chen
    - BAD: "Who was involved in Maria Chen's success?" → too vague, not indirect

    ---

    ## STEP 3: SELF-CONTAINMENT — THE MOST IMPORTANT RULE
    Every QA pair must be **fully understandable on its own**, without access to the original document, any other QA pair, or any external context.

    A reader should be able to read the question and answer in isolation and fully understand:
    - **Who** is being referred to (full names, never pronouns or vague references)
    - **What** happened or is being described (complete action or fact)
    - **When** it occurred (absolute dates, not relative ones)
    - **Where** it occurred (if relevant)
    - **Why** it matters or what context surrounds it (if needed for understanding)

    ### Self-containment rules:
    - **Never use pronouns**: Replace every "he", "she", "it", "they", "the company", "the organization", "the team" with the actual entity name
    - **Never use relative references**: Replace "the former", "the latter", "the above", "the following" with explicit names
    - **Never reference the document**: Do not say "the document states", "according to the article", "as mentioned", "the author says"
    - **Never reference metadata**: Do not mention the document date, title, source, or author inside questions or answers
    - **Never use relative time**: Replace "last year", "recently", "currently", "at the time", "formerly" with absolute dates derived from the document date
    - **Always name entities explicitly**: Every person, place, organization, product, or event must be named in full each time it is referenced
    - **Include all context needed**: If a fact only makes sense with background context, include that context in the answer

    ### Self-containment test — ask yourself for every QA pair:
    - If I showed only this question and answer to someone who had never seen the document, would they fully understand it?
    - Is there any word or reference that requires the reader to look elsewhere to understand?
    - Are there any pronouns, vague references, or relative time expressions?

    If the answer to any of these is "no" or "yes" respectively → rewrite until it passes.

    ### Self-containment examples:

    **BAD** (requires document):
    Q: "What did he provide to her that helped close the deal?"
    A: "He provided the financial model, which she used to secure the contract."

    **GOOD** (fully standalone):
    Q: "What did CFO James Lim provide to CEO Sarah Park that enabled TechStart to close its Series A deal in March 2021?"
    A: "CFO James Lim provided CEO Sarah Park with a detailed financial model in March 2021. Sarah Park used this model to demonstrate TechStart's profitability trajectory to investors, which enabled TechStart to successfully close its Series A funding round."

    ---

    ## STEP 4: CRAFT SIMPLE, FOCUSED QUESTIONS
    Each question should:
    - Ask about **one specific indirect fact**
    - Be clear and direct
    - Include enough context to be understood completely without the document
    - Use full names and explicit references (never pronouns or vague references)
    - Be answerable with a concise, standalone response

    ### Good question patterns:
    - "Who owned [specific named entity]?"
    - "What did [Full Name] own that [did something specific]?"
    - "Who provided [specific item] to [Full Name] during [named event] in [date]?"
    - "What did [Full Name] do after receiving [specific item] from [Full Name] in [date]?"
    - "When did [specific named event] begin, given that it lasted [duration] and ended in [date]?"
    - "How old was [Full Name] when [specific named event] occurred in [date]?"
    - "Who received [specific item] from [Full Name] in [date]?"

    ### Bad question patterns (never use):
    - "What did he give her?" → missing names
    - "Who was involved?" → too vague
    - "What happened as a result?" → no context
    - "What did the company do?" → which company?
    - "Who did this enable?" → "this" is not grounded

    ---

    ## STEP 5: CRAFT CONCISE, SELF-CONTAINED ANSWERS
    Answers should:
    - Address the question directly and only the question
    - Be 1-4 sentences typically
    - Contain **all context needed to be understood standalone**
    - Use full entity names every time — never pronouns
    - Include absolute dates, full names, and explicit references throughout
    - For action chains, include the full sequence: who did what, to whom, and what resulted

    ---

    ## STEP 6: APPLY TEMPORAL GROUNDING
    Convert all relative time expressions to absolute dates using the document date ({doc_extracted_date}):
    - "last year" → convert to specific year
    - "last month" → convert to specific month and year
    - "recently" → convert to approximate month/year
    - "two decades ago" → convert to specific decade or year
    - "currently" / "now" → "as of [document date]"
    - "35 years old" → "35 years old as of [document date]"
    - Calculated start dates → state the derived absolute date explicitly

    ---

    ## STEP 7: EXTRACTION CHECKLIST FOR ACTION SEQUENCES
    For every action chain found in the document, verify you have extracted questions covering ALL of:
    - [ ] Who performed the action? (enabler)
    - [ ] What specific action was performed? (precise verb)
    - [ ] Who received or benefited? (beneficiary)
    - [ ] What outcome resulted?
    - [ ] Forward direction: enabler → beneficiary
    - [ ] Backward direction: beneficiary ← enabler
    - [ ] Action: what was transferred/provided/delivered
    - [ ] Sequence: what happened next as a result

    ---

    ## STEP 8: FINAL OUTPUT
    Generate a QA pair for **every relevant, distinct indirect fact** found in the document:
    - Each covers a **single, distinct indirect fact**
    - Every QA pair is **fully self-contained** — readable and understandable in complete isolation
    - Are temporally grounded with absolute dates
    - Are **concise** — answers should be 1-4 sentences
    - No two QA pairs cover the same fact
    - **Do not limit yourself** — extract every indirect fact the document supports
    - **Skip irrelevant content** — no generic, boilerplate, or vague information
    - **Do not skip minor entities** — if a possessive, pronoun, or action chain involves a minor entity, extract it

    Output as parseable JSON only — no preamble, postamble, or markdown fences:
    {{
    "qa_pairs": [
        {{
        "question": "...",
        "answer": "..."
        }}
    ]
    }}

    ---

    ## FINAL CHECKLIST:
    Before submitting, verify every QA pair passes ALL of the following:
    - Contains zero pronouns (he, she, it, they, the company, the organization)
    - Contains zero relative time expressions (recently, last year, currently, formerly)
    - Contains zero document references (the article, the document, as mentioned)
    - Contains zero metadata references (document date, source, author, title)
    - Is fully understandable by someone who has never seen the original document
    - Asks about one specific indirect fact
    - Answer is 1-4 sentences
    - No two QA pairs cover the same fact
    - All action chains have been extracted from all directions (forward, backward, action, outcome, sequence)
    - All possessives have been extracted bidirectionally
    - All pronouns have been resolved to actual entity names
    - All relative dates have been converted to absolute dates

    **GUIDING PRINCIPLE**: One indirect fact, one question, one concise answer. Every QA pair must stand completely alone — if it requires anything outside itself to be understood, rewrite it.
    """
    return prompt


def prepare_prompt_for_direct_fact_extraction_v3(chunk_content, doc_extracted_date):
    prompt = f"""
    You are tasked with extracting high-quality, self-contained QA pairs from a given document.

    ## INPUT FORMAT
    <document>
    {chunk_content}
    </document>

    <document_metadata>
    Document date: {doc_extracted_date}
    </document_metadata>

    Use the document date from metadata for temporal grounding throughout your QA pairs.

    ---

    ## STEP 1: DOCUMENT ANALYSIS
    - Read the entire document carefully
    - Note the document date — use it as your reference point for all temporal grounding
    - Identify ALL entities mentioned — major or minor, named or unnamed (people, organizations, places, dates, numbers, products, roles, events, concepts specific to the document)
    - Note important facts, claims, and relationships
    - Map out the temporal structure (when events occurred)

    ---

    ## STEP 2: IDENTIFY INDIVIDUAL DIRECT FACTS
    Direct facts are explicitly and clearly stated in the document.

    Extract **every piece of information**, no matter how minor:
    - Facts about major AND minor entities
    - Specific facts about people, places, organizations, or events — even if mentioned only once or briefly
    - Individual claims or statements
    - Single cause-and-effect relationships
    - Specific numerical data or measurements
    - Individual attributions or contributions
    - Roles, titles, or positions mentioned in passing
    - Dates, locations, or figures cited incidentally

    **Do not skip an entity just because it appears minor or is only mentioned once. If it's in the document and specific, extract it.**

    ### Skip a fact if it is:
    - Generic background knowledge not specific to the document (e.g., common definitions, well-known historical facts)
    - Boilerplate or filler content (e.g., publication disclaimers, author bios, ad copy)
    - Redundant restatements of another fact already captured
    - Too vague or ambiguous to form a clear, answerable question

    ---

    ## STEP 3: SELF-CONTAINMENT — THE MOST IMPORTANT RULE
    Every QA pair must be **fully understandable on its own**, without access to the original document, any other QA pair, or any external context.

    A reader should be able to read the question and answer in isolation and fully understand:
    - **Who** is being referred to (full names, never pronouns or vague references)
    - **What** happened or is being described (complete action or fact)
    - **When** it occurred (absolute dates, not relative ones)
    - **Where** it occurred (if relevant)
    - **Why** it matters or what context surrounds it (if needed for understanding)

    ### Self-containment rules:
    - **Never use pronouns**: Replace every "he", "she", "it", "they", "the company", "the organization", "the team" with the actual entity name
    - **Never use relative references**: Replace "the former", "the latter", "the above", "the following" with explicit names
    - **Never reference the document**: Do not say "the document states", "according to the article", "as mentioned", "the author says"
    - **Never reference metadata**: Do not mention the document date, title, source, or author inside questions or answers
    - **Never use relative time**: Replace "last year", "recently", "currently", "at the time", "formerly" with absolute dates derived from the document date
    - **Always name entities explicitly**: Every person, place, organization, product, or event must be named in full each time it is referenced
    - **Include all context needed**: If a fact only makes sense with background context, include that context in the answer

    ### Self-containment test — ask yourself for every QA pair:
    - If I showed only this question and answer to someone who had never seen the document, would they fully understand it?
    - Is there any word or reference that requires the reader to look elsewhere to understand?
    - Are there any pronouns, vague references, or relative time expressions?

    If the answer to any of these is "no" or "yes" respectively → rewrite until it passes.

    ---

    ## STEP 4: CRAFT SIMPLE, FOCUSED QUESTIONS
    Each question should:
    - Ask about **one specific thing**
    - Be clear and direct
    - Include enough context to be understood completely without the document
    - Use full names and explicit references (never pronouns or vague references)
    - Be answerable with a concise, standalone response

    ### Good question patterns:
    - "What role did [Full Name] hold at [Organization] as of [date]?"
    - "What was the outcome of [specific named event]?"
    - "How much did [specific metric] change between [Year A] and [Year B]?"
    - "What did [Full Name] say about [specific topic] in [time period]?"
    - "When did [specific named event] occur?"
    - "Where was [specific named entity] located as of [date]?"

    ### Bad question patterns (never use):
    - "What did he say?" → missing name
    - "What happened next?" → no context
    - "What was the outcome?" → unclear what event
    - "What did the company announce?" → which company?
    - "What was recently revealed?" → "recently" is not grounded

    ---

    ## STEP 5: CRAFT CONCISE, SELF-CONTAINED ANSWERS
    Answers should:
    - Address the question directly and only the question
    - Be 1-4 sentences typically
    - Contain **all context needed to be understood standalone** — never assume the reader knows anything
    - Use full entity names every time — never pronouns
    - Include absolute dates, full names, and explicit references throughout
    - Not repeat the question verbatim, but naturally incorporate its context

    ### Answer self-containment examples:

    **BAD** (requires document):
    Q: "What did he announce?"
    A: "He announced the merger would proceed in Q3."

    **GOOD** (fully standalone):
    Q: "What did Apple CEO Tim Cook announce regarding the Intel acquisition in June 2023?"
    A: "Apple CEO Tim Cook announced in June 2023 that Apple's planned acquisition of Intel's modem division would proceed in Q3 2023, marking Apple's largest hardware acquisition in five years."

    ---

    ## STEP 6: APPLY TEMPORAL GROUNDING
    Convert all relative time expressions to absolute dates using the document date ({doc_extracted_date}):
    - "last year" → convert to specific year
    - "last month" → convert to specific month and year
    - "recently" → convert to approximate month/year
    - "two decades ago" → convert to specific decade or year
    - "currently" / "now" → "[as of document date]"
    - "35 years old" → "35 years old as of [document date]"
    - "holds three patents" → "holds three patents as of [document date]"

    ---

    ## STEP 7: FINAL OUTPUT
    Generate a QA pair for **every relevant, distinct direct fact** found in the document:
    - Each covers a **single, distinct fact**
    - Every QA pair is **fully self-contained** — readable and understandable in complete isolation
    - Are temporally grounded with absolute dates
    - Are **concise** — answers should be 1-4 sentences
    - No two QA pairs cover the same fact
    - **Do not limit yourself** — if the document contains 30 relevant facts, generate 30 QA pairs
    - **Skip irrelevant content** — no generic, boilerplate, or vague information
    - **Do not skip minor entities** — if a person, place, number, or role is specifically mentioned, extract it

    Output as parseable JSON only — no preamble, postamble, or markdown fences:
    {{
    "qa_pairs": [
        {{
        "question": "...",
        "answer": "..."
        }}
    ]
    }}

    ---

    ## FINAL CHECKLIST:
    Before submitting, verify every QA pair passes ALL of the following:
    - Contains zero pronouns (he, she, it, they, the company, the organization)
    - Contains zero relative time expressions (recently, last year, currently, formerly)
    - Contains zero document references (the article, the document, as mentioned)
    - Contains zero metadata references (document date, source, author, title)
    - Is fully understandable by someone who has never seen the original document
    - Asks about one specific fact
    - Answer is 1-4 sentences
    - No two QA pairs cover the same fact
    - No minor entity was skipped just because it seemed insignificant

    **GUIDING PRINCIPLE**: One fact, one question, one concise answer. Every QA pair must stand completely alone — if it requires anything outside itself to be understood, rewrite it.
    """
    return prompt


def prepare_prompt_for_consolidation(qa_pairs, doc_extracted_date):
    prompt = f"""
    You are tasked with consolidating a set of QA pairs by identifying commonalities and combining related pairs into richer, more comprehensive QA pairs. You will generate as many meaningful combinations as possible.

    ## INPUT FORMAT
    You will receive a list of QA pairs:
    <qa_pairs>
    {qa_pairs}
    </qa_pairs>

    <document_metadata>
    Document date: {doc_extracted_date}
    </document_metadata>

    ---

    ## WHAT IS CONSOLIDATION?
    Consolidation means identifying QA pairs that share a common thread and merging them into a single, more informative QA pair. The goal is to surface relationships and patterns across QA pairs that are not visible when each pair stands alone.

    ### Example:
    **Originals:**
    - Q: "Who founded GeoCities and when?" A: "GeoCities was founded in 1994 by David Bohnett and John Rezner."
    - Q: "Who founded RealNetworks and when?" A: "RealNetworks was founded in 1994 by Rob Glaser."

    **Consolidated:**
    - Q: "Which two early internet companies were both founded in 1994?"
    A: "GeoCities, founded by David Bohnett and John Rezner, and RealNetworks, founded by Rob Glaser, were both founded in 1994."

    ---

    ## STEP 1: INDEX ALL QA PAIRS
    Before combining anything:
    - Assign a number to every QA pair (1, 2, 3, ... N)
    - For each QA pair, note its key attributes:
    - Entities involved (people, organizations, places, dates)
    - Relationship type (founding, ownership, action, attribute, temporal, location, causation)
    - Key values (years, amounts, roles, outcomes)

    This index will be used to systematically find all possible combinations.

    ---

    ## STEP 2: IDENTIFY COMBINATION OPPORTUNITIES
    Look for QA pairs that share ANY of the following commonalities:

    ### Commonality types:
    - **Same time period**: Two or more QA pairs refer to events in the same year, decade, or era
    - **Same entity**: Two or more QA pairs involve the same person, organization, or place
    - **Same relationship type**: Two or more QA pairs describe the same kind of relationship (e.g. multiple founding events, multiple acquisitions, multiple contributions)
    - **Same attribute**: Two or more QA pairs share a common attribute value (e.g. same funding amount, same location, same role title)
    - **Same action type**: Two or more QA pairs describe the same type of action performed by different entities (e.g. multiple people who provided something to someone)
    - **Same outcome type**: Two or more QA pairs describe similar outcomes (e.g. multiple companies that went public, multiple projects that failed)
    - **Cause and effect across pairs**: One QA pair describes a cause and another describes its effect
    - **Sequential events**: Two or more QA pairs describe events that form a chronological sequence
    - **Contrasting information**: Two or more QA pairs describe contrasting facts about similar entities (e.g. one company succeeded, another failed, both in the same year)
    - **Shared context**: Two or more QA pairs share a broader context (e.g. both relate to the same industry, movement, or era)

    ---

    ## STEP 3: GENERATE ALL PERMUTATIONS OF COMBINATIONS
    For every set of QA pairs that share a commonality, generate consolidated versions at EVERY combination size:

    - **Pairs (2)**: Combine every valid pair of QA pairs that share a commonality
    - **Triplets (3)**: Combine every valid group of 3 QA pairs that share a commonality
    - **Quadruplets (4)**: Combine every valid group of 4 QA pairs that share a commonality
    - **Larger groups**: Continue for 5, 6, and beyond if meaningful combinations exist

    ### Rules for permutations:
    - Every combination must have a genuine, identifiable commonality — do not force combinations that have no natural connection
    - A QA pair can appear in multiple combinations (it may share different commonalities with different pairs)
    - Larger combinations are only valid if ALL members share the same commonality thread — do not combine just because some members overlap
    - If combining 4 QA pairs produces an answer that is too unwieldy (more than 150 words), split into smaller combinations instead

    ### How to systematically find all permutations:
    1. Start with all possible pairs (QA1+QA2, QA1+QA3, QA1+QA4... QA2+QA3, etc.)
    2. For each valid pair, check if a third QA pair shares the same commonality → form triplet
    3. For each valid triplet, check if a fourth QA pair shares the same commonality → form quadruplet
    4. Continue until no further valid additions exist
    5. Record which original QA pairs (by index number) contributed to each combination

    ---

    ## STEP 4: CRAFT CONSOLIDATED QUESTIONS
    Each consolidated question should:
    - Reflect the shared commonality that connects the combined QA pairs
    - Be phrased to naturally invite a multi-part answer
    - Use full names and explicit references (never pronouns)
    - Be clear and direct despite covering multiple facts

    ### Good consolidated question patterns:
    - **Same time period**: "Which [entities] were both/all [action] in [year]?"
    - **Same entity**: "What roles did [Person X] play in both [Event A] and [Event B]?"
    - **Same relationship type**: "Which companies were founded by [common attribute] founders in [era]?"
    - **Same attribute**: "Which [entities] shared [common attribute] and what distinguished them?"
    - **Sequential**: "What sequence of events led from [Event A] to [Event B] to [Event C]?"
    - **Contrasting**: "How did [Entity A] and [Entity B] differ in their approach to [common topic]?"
    - **Cause and effect**: "What caused [Event A] and what was the resulting outcome [Event B]?"

    ---

    ## STEP 5: CRAFT CONSOLIDATED ANSWERS
    Each consolidated answer should:
    - Address all components of the combined QA pairs
    - Make the commonality explicit and clear
    - Be structured logically (chronological, comparative, or causal as appropriate)
    - Use full entity names throughout — never pronouns
    - Be 2-6 sentences typically, depending on the number of combined pairs
    - Not exceed 150 words — if it would, reduce the combination size

    ### Answer structure patterns:
    - **List pattern**: "[Entity A], which [detail], and [Entity B], which [detail], were both [commonality]."
    - **Sequential pattern**: "[Event A] occurred in [date], followed by [Event B] in [date], culminating in [Event C]."
    - **Comparative pattern**: "While [Entity A] [did X], [Entity B] [did Y], both sharing [commonality]."
    - **Causal pattern**: "[Event A] led directly to [Event B], which in turn resulted in [Event C]."

    ---

    ## STEP 6: SELF-CONTAINMENT — THE MOST IMPORTANT RULE
    Every consolidated QA pair must be **fully understandable on its own**, without access to the original document, the original QA pairs, or any external context.

    ### Self-containment rules:
    - **Never use pronouns**: Replace every "he", "she", "it", "they", "the company" with the actual entity name
    - **Never use relative references**: No "the former", "the latter", "as mentioned above"
    - **Never reference the document or original QA pairs**: No "as stated earlier", "the document mentions"
    - **Never reference metadata**: Do not mention document date, title, source, or author
    - **Never use relative time**: Replace "last year", "recently", "currently" with absolute dates using document date ({doc_extracted_date})
    - **Always name entities explicitly**: Every person, place, organization, product, or event must be named in full
    - **Include all context needed**: The consolidated answer must be self-sufficient

    ### Self-containment test — for every consolidated QA pair ask:
    - If I showed only this question and answer to someone who had never seen the original QA pairs or document, would they fully understand it?
    - Are there any pronouns, vague references, or relative time expressions?
    - Does the answer contain everything needed to understand it completely?

    ---

    ## STEP 7: APPLY TEMPORAL GROUNDING
    Convert all relative time expressions to absolute dates using the document date ({doc_extracted_date}):
    - "last year" → convert to specific year
    - "last month" → convert to specific month and year
    - "recently" → convert to approximate month/year
    - "currently" / "now" → "as of [document date]"

    ---

    ## STEP 8: FINAL OUTPUT
    Output ALL valid consolidated QA pairs across all combination sizes. For each consolidated QA pair, record which original QA pair indices were combined.

    Output as parseable JSON only — no preamble, postamble, or markdown fences:
    {{
    "consolidated_qa_pairs": [
        {{
        "question": "...",
        "answer": "...",
        "combined_from": [1, 2],
        "commonality": "...",
        "combination_size": 2
        }},
        {{
        "question": "...",
        "answer": "...",
        "combined_from": [1, 2, 3],
        "commonality": "...",
        "combination_size": 3
        }}
    ]
    }}

    ### Output fields:
    - **question**: The consolidated question
    - **answer**: The consolidated answer
    - **combined_from**: List of original QA pair index numbers that were combined
    - **commonality**: A brief label describing what the combined pairs share (e.g. "same founding year", "same person", "cause and effect", "sequential events")
    - **combination_size**: Number of original QA pairs combined (2, 3, 4, etc.)

    ---

    ## FINAL CHECKLIST:
    Before submitting, verify every consolidated QA pair passes ALL of the following:
    - Has a genuine, identifiable commonality connecting all combined pairs
    - Is a meaningfully different combination from other consolidated pairs
    - Contains zero pronouns (he, she, it, they, the company, the organization)
    - Contains zero relative time expressions (recently, last year, currently, formerly)
    - Contains zero document or metadata references
    - Is fully understandable by someone who has never seen the original QA pairs or document
    - Answer does not exceed 150 words
    - All permutation sizes (2, 3, 4+) have been explored
    - combined_from field accurately lists the indices of all contributing original QA pairs

    **GUIDING PRINCIPLE**: Find every meaningful way to combine QA pairs across every possible group size. If two or more QA pairs share any genuine commonality — time, entity, relationship type, attribute, sequence, contrast — combine them. Generate every valid permutation.
    """
    return prompt


def prepare_prompt_for_entity_surfacing(qa_pairs_str: str, doc_extracted_date: str) -> str:
    """
    Given ALL self-contained QA pairs for a document, produce entity-surfacing
    QA pairs where:
    - The question describes an entity through its facts without naming it
    - The answer reveals the entity's full name and confirms the match

    Takes the full doc-level QA pair group to enable multi-fact entity profiling.
    """
    return f"""You are tasked with generating entity-surfacing QA pairs from a set of existing QA pairs about named entities.

An entity-surfacing QA pair is one where:
- The QUESTION describes an entity using its attributes, relationships, and facts — without naming the entity
- The ANSWER provides the entity's name, along with enough context to confirm why the description matches

These QA pairs are specifically designed to allow someone to identify an unknown entity from clues alone,
without prior knowledge of the entity's name.

## INPUT FORMAT
You will receive a set of existing QA pairs:
<qa_pairs>
{qa_pairs_str}
</qa_pairs>

<document_metadata>
Document date: {doc_extracted_date}
</document_metadata>

---

## STEP 1: IDENTIFY ALL NAMED ENTITIES
Go through every QA pair and identify every named entity that appears:
- People (full names)
- Organizations (full names)
- Places (full names)
- Products, works, or events (full names)

For each named entity, collect ALL facts known about it across ALL QA pairs — not just the pair it
first appeared in. The richer the fact set, the more discriminating the surfacing question can be.

---

## STEP 2: BUILD A FACT PROFILE FOR EACH ENTITY
For each named entity, list every known fact from the QA pairs:
- Roles or titles held
- Actions performed or received
- Relationships to other named entities
- Attributes or properties
- Locations associated with
- Dates or time periods associated with
- Outcomes or achievements

Only include facts explicitly stated in the QA pairs — do not infer or add external knowledge.

---

## STEP 3: GENERATE ENTITY-SURFACING QUESTIONS
For each named entity, generate surfacing questions by replacing the entity's name with a
descriptive phrase constructed from its fact profile.

### Surfacing question rules:
- The entity's name must NEVER appear in the question
- The descriptive phrase must be specific enough that most other entities would not qualify
- Use facts from the entity's profile to construct the description
- Combine multiple facts for richer, more discriminating questions
- Questions must be fully self-contained — no pronouns, no vague references, no relative dates
- All dates must be absolute

### Question complexity levels — generate ALL of the following for each entity:

**Single-fact surfacing** (one attribute identifies the entity):
- Only viable when a single fact is highly distinctive on its own
- Example: "What is the name of the architect who designed a famous opera house in Sydney?"

**Two-fact surfacing** (two attributes combined):
- Combine two facts that together narrow the answer space significantly
- Example: "What is the name of the architect who designed a famous opera house in Sydney
  and was born in Denmark in 1918?"

**Multi-fact surfacing** (three or more attributes combined):
- Use the richest possible description to make the question highly specific
- Example: "What is the name of the architect who designed a famous opera house in Sydney,
  was born in Denmark in 1918, and won a major international architecture prize in 2003?"
- These are the most valuable for entity resolution from clues

### Question phrasing patterns:
- "What is the name of the [role] who [fact 1] and [fact 2]?"
- "Who is the [role] that [fact 1], [fact 2], and [fact 3]?"
- "What is the name of the person who [action] in [year] and later [action] in [year]?"
- "Which [entity type] [fact 1] and also [fact 2]?"
- "Who [fact 1] in [year], [fact 2] in [year], and [fact 3]?"

### Bad surfacing question patterns (never generate):
- "Who is the famous person associated with this topic?" — too broad, no discriminating facts
- "What is the name of the author?" — no discriminating attributes
- "Who wrote a novel?" — too vague, millions qualify
- "What is the name of the individual mentioned?" — references the document
- "Who is he?" — pronoun, not self-contained
- "What is the name of the person who did something recently?" — relative time, not grounded

---

## STEP 4: GENERATE RELATIONSHIP-TRAVERSAL SURFACING QUESTIONS
These questions surface a SECONDARY entity by describing it through its relationship to a PRIMARY
named entity, combined with additional attributes of the secondary entity.

These are critical for multi-hop questions where the answer requires first identifying a secondary
entity that is connected to a known primary entity.

### Relationship-traversal rules:
- The secondary entity's name must NOT appear in the question
- The primary entity's name MAY appear in the question — it is the known anchor
- Combine the relationship to the primary entity with at least one additional attribute of the
  secondary entity to make the question discriminating
- Questions must be fully self-contained

### Relationship-traversal phrasing patterns:
- "What is the name of [primary entity]'s [relationship] who [secondary entity fact]?"
- "Who is the [relationship] of [primary entity] that [secondary entity fact] in [year]?"
- "What is the name of the person connected to [primary entity] as [relationship] who also
  [secondary entity fact]?"
- "Which [entity type] is associated with [primary entity] through [relationship] and
  additionally [secondary entity fact]?"

### Bad relationship-traversal patterns (never generate):
- "Who is related to [primary entity]?" — too vague, no secondary entity attributes
- "What is the name of [primary entity]'s associate?" — no discriminating facts about secondary
- "Who worked with [primary entity]?" — too broad

---

## STEP 5: CRAFT SELF-CONTAINED ANSWERS
Each answer must:
- State the entity's full name explicitly
- Briefly confirm why the description in the question matches this entity, using 1-2 facts
- Be 1-3 sentences
- Use full entity names throughout — never pronouns
- Include absolute dates

### Answer structure:
- Lead with the full name of the entity being surfaced
- Follow with 1-2 confirming facts drawn directly from the QA pairs
- Do not introduce any facts not present in the input QA pairs

---

## STEP 6: SELF-CONTAINMENT — THE MOST IMPORTANT RULE
Every QA pair must be fully understandable on its own, without access to the original
document, any other QA pair, or any external context.

### Self-containment rules:
- Never use pronouns: Replace every "he", "she", "it", "they", "the company" with the
  actual entity name or a descriptive phrase
- Never reference the document: No "the document states", "as mentioned", "the article"
- Never use relative time: Replace "last year", "recently", "currently" with absolute dates
  derived from the document date ({doc_extracted_date})
- Always name primary entities explicitly in relationship-traversal questions
- Include all context needed: The question and answer must be fully self-sufficient

### Self-containment test — for every QA pair ask:
- If I showed only this question and answer to someone who had never seen the original QA pairs
  or document, would they fully understand it?
- Are there any pronouns, vague references, or relative time expressions?
- Does the answer contain everything needed to confirm the match?

---

## STEP 7: FINAL OUTPUT
Generate entity-surfacing QA pairs for EVERY named entity found across all QA pairs:
- At least one single-fact surfacing question per entity (where a single fact is distinctive)
- At least one two-fact surfacing question per entity
- At least one multi-fact surfacing question per entity (using 3+ facts where available)
- At least one relationship-traversal question for every named relationship between entities
- Do not skip minor entities — if an entity appears in even one QA pair, generate surfacing
  questions for it
- Do not repeat the same combination of facts across multiple questions
- Do not limit yourself — if an entity has many facts, generate surfacing questions at every
  meaningful combination level

Output as parseable JSON only — no preamble, postamble, or markdown fences:
{{
    "entity_surfacing_qa_pairs": [
        {{
            "entity": "the named entity this question is designed to surface",
            "question": "...",
            "answer": "...",
            "complexity": "single-fact | two-fact | multi-fact | relationship-traversal",
            "facts_used": ["brief description of each fact used in the question"]
        }}
    ]
}}

---

## FINAL CHECKLIST:
Before submitting, verify every QA pair passes ALL of the following:
- The target entity's name does not appear anywhere in the question
- For relationship-traversal questions, the primary entity IS named in the question
- The question is specific enough that most entities in the world would not qualify
- Contains zero pronouns (he, she, it, they, the company, the organization)
- Contains zero relative time expressions (recently, last year, currently, formerly)
- Contains zero document references (the article, the document, as mentioned)
- Is fully understandable by someone who has never seen the original QA pairs or document
- Answer explicitly names the entity and confirms the match with 1-2 supporting facts
- Answer is 1-3 sentences
- All three complexity levels have been generated for each entity where facts support it
- All named relationships between entities have relationship-traversal questions

GUIDING PRINCIPLE: The question describes the entity through its facts — the answer reveals
its name. Every surfacing QA pair must be specific enough to discriminate the target entity from
all others, and general enough to be answerable from clues alone.
"""


def prepare_prompt_for_self_containment_check(question: str, answer: str) -> str:
    return f"""
You are tasked with evaluating whether a QA pair is fully self-contained.

## INPUT
<qa_pair>
Question: {question}
Answer: {answer}
</qa_pair>

---

## WHAT IS SELF-CONTAINMENT?
A QA pair is self-contained if a reader who has never seen any source document, article, or additional context can read the question and answer in isolation and fully understand:
- **Who** is being referred to (no unresolved pronouns or vague references)
- **What** happened or is being described (complete action or fact)
- **When** it occurred (absolute dates, not relative ones)
- **Where** it occurred if relevant (explicit location)
- **Why** it matters if context is needed for understanding

---

## STEP 1: CHECK FOR SELF-CONTAINMENT VIOLATIONS
Evaluate the QA pair against every violation type below. A single violation is enough to mark the pair as NOT self-contained.

### Violation 1: Unresolved Pronouns
The question or answer contains pronouns without explicit antecedents in the same QA pair.
- Failing examples: "he", "she", "it", "they", "the company", "the organization", "the team", "the firm", "the group"
- Ask: Can the reader identify exactly who or what is being referred to without any outside context?

### Violation 2: Relative Time Expressions
The question or answer contains time expressions that are not anchored to an absolute date.
- Failing examples: "recently", "last year", "last month", "currently", "at the time", "formerly", "soon after", "a few years later", "now", "today"
- Ask: Can the reader know exactly when something happened without any outside reference?

### Violation 3: Document or Source References
The question or answer refers to a source document, article, or external text.
- Failing examples: "the document states", "according to the article", "as mentioned", "the report says", "the author notes", "as described above"
- Ask: Does the QA pair stand alone without pointing to an external source?

### Violation 4: Vague or Implicit References
The question or answer uses references that assume shared knowledge not present in the QA pair itself.
- Failing examples: "the former", "the latter", "the above", "the following", "this event", "the incident", "the deal", "the agreement" without naming what it is
- Ask: Is every reference explicit and named within the QA pair itself?

### Violation 5: Missing Context for Understanding
The answer cannot be fully understood without additional background that is not provided in the QA pair.
- Failing examples: An answer that names a person but gives no identifying context when context is needed; an answer that references an event by name but provides no detail about what the event was when that detail is necessary
- Ask: Would a reader with zero prior knowledge fully understand the answer?

### Violation 6: Incomplete Answer
The answer does not fully address what the question asks.
- Failing examples: Question asks "who and when" but answer only provides "who"; question asks for an outcome but answer only describes the action
- Ask: Does the answer address every part of the question?

---

## STEP 2: MAKE A DETERMINATION
After checking all violations:
- If ANY violation is found → `is_self_contained: false`
- If NO violations are found → `is_self_contained: true`

---

## STEP 3: OUTPUT
Output as parseable JSON only — no preamble, postamble, or markdown fences:
{{"is_self_contained": true or false}}
"""


def prepare_prompt_for_self_containment_fix(
    question: str,
    answer: str,
    chunk_content: str,
    doc_extracted_date: str,
) -> str:
    return f"""
You are tasked with fixing a QA pair that is not fully self-contained.

## SOURCE DOCUMENT
<document>
{chunk_content}
</document>

<document_metadata>
Document date: {doc_extracted_date}
</document_metadata>

## QA PAIR TO FIX
<qa_pair>
Question: {question}
Answer: {answer}
</qa_pair>

---

## WHAT IS SELF-CONTAINMENT?
A QA pair is self-contained if a reader who has never seen any source document, article, or additional context can read the question and answer in isolation and fully understand:
- **Who** is being referred to (no unresolved pronouns or vague references)
- **What** happened or is being described (complete action or fact)
- **When** it occurred (absolute dates, not relative ones)
- **Where** it occurred if relevant (explicit location)
- **Why** it matters if context is needed for understanding

---

## COMMON VIOLATIONS TO FIX

### Violation 1: Unresolved Pronouns
Replace every pronoun with the actual entity name from the source document.
- "he", "she", "it", "they", "the company", "the organization", "the team", "the firm" → replace with full name

### Violation 2: Relative Time Expressions
Replace all relative time expressions with absolute dates using the document date ({doc_extracted_date}) as reference.
- "recently" → specific month/year
- "last year" → specific year
- "last month" → specific month and year
- "currently" / "now" → "as of {doc_extracted_date}"
- "at the time" → specific date

### Violation 3: Document or Source References
Remove all references to the source document.
- "the document states", "according to the article", "as mentioned" → rewrite to state the fact directly

### Violation 4: Vague or Implicit References
Replace all vague references with explicit named references from the source document.
- "the former", "the latter", "the incident", "the deal" → replace with the actual named entity or event

### Violation 5: Missing Context
Add any context from the source document needed for the answer to be understood standalone.
- If a person is named but unidentifiable, add their role or affiliation
- If an event is named but unexplained, add a brief description

### Violation 6: Incomplete Answer
Ensure the answer addresses every part of the question. Use the source document to fill in any missing information.

---

## RULES FOR FIXING
- Use the source document ONLY to resolve missing context — do not introduce new facts beyond what is needed for self-containment
- Do not change the core fact being expressed — only fix the self-containment issues
- Keep the answer concise — 1 to 4 sentences
- Use full entity names throughout — never pronouns
- Include absolute dates throughout — never relative time expressions

---

## OUTPUT
Return the fixed QA pair as parseable JSON only — no preamble, postamble, or markdown fences:
{{
  "question": "...",
  "answer": "..."
}}
"""


def _format_anchor_for_prompt(anchor_qa: dict, anchor_doc_id: str) -> str:
    return (
        f"[Doc {anchor_doc_id}]\n"
        f"Q: {anchor_qa.get('question', '')}\n"
        f"A: {anchor_qa.get('answer', '')}"
    )


def _format_other_qa_batch_for_prompt(batch: list[tuple[str, dict]]) -> str:
    """
    Format a batch of (doc_id, qa_pair) tuples into a labeled block.
    Groups consecutive entries by doc_id for readability.
    """
    lines = []
    current_doc = None
    idx_within_doc = 0
    for doc_id, qa in batch:
        if doc_id != current_doc:
            if current_doc is not None:
                lines.append("")
            lines.append(f"[Doc {doc_id}]")
            current_doc = doc_id
            idx_within_doc = 1
        lines.append(f"  {idx_within_doc}. Q: {qa.get('question', '')}")
        lines.append(f"     A: {qa.get('answer', '')}")
        idx_within_doc += 1
    return "\n".join(lines)


def prepare_prompt_for_crossdoc_anchor_combination(
    anchor_qa: dict,
    anchor_doc_id: str,
    other_qa_batch: list[tuple[str, dict]],
) -> str:
    anchor_block = _format_anchor_for_prompt(anchor_qa, anchor_doc_id)
    other_block  = _format_other_qa_batch_for_prompt(other_qa_batch)

    other_doc_ids = sorted({doc_id for doc_id, _ in other_qa_batch})
    other_doc_list = ", ".join(f"Doc {d}" for d in other_doc_ids)

    return f"""You are tasked with finding cross-document entity connections and generating combined QA pairs.

## INPUT

ONE ANCHOR QA PAIR:
<anchor>
{anchor_block}
</anchor>

QA PAIRS FROM OTHER DOCUMENTS ({other_doc_list}):
<candidates>
{other_block}
</candidates>

---

## TASK

Scan the candidate QA pairs for cross-document connections with the anchor.
There are TWO types of connection to identify and generate QA pairs for:

---

### TYPE A — Converging clues → single entity

The anchor and one or more candidates describe DIFFERENT facts about the SAME named entity.

Conditions:
- The same named entity appears in the anchor AND in ≥1 candidate doc
  (may be described differently or from a different angle — still the same real-world entity)
- The facts are distinct (not the same fact repeated verbatim across docs)

Generate one QA pair per unique combination of docs that share the same entity:
- Q: "Who [fact from anchor doc] and [fact from candidate doc(s)]?"
  Include facts from as many docs as contribute to identifying the entity.
  Generate a two-doc version, and a three-doc+ version if ≥3 docs share the entity.
- A: "[Entity full name]. [1-2 confirming facts drawn from the contributing docs]."

The question must describe the entity entirely through accumulated clues.
The answer must lead with the entity's full name.

---

### TYPE B — Parallel property → multiple entities

The anchor entity and one or more candidate entities SHARE the same fact, property,
role, event, or characteristic — but they are DIFFERENT entities.

Conditions:
- The anchor mentions entity X with property P
- A candidate mentions entity Y (Y ≠ X) with the same or closely parallel property P
- The shared property is SPECIFIC enough to be meaningful and produce a bounded answer list
  GOOD: "won the [specific award] in [year]", "held [specific role] at [specific org] in [year]",
        "filed for bankruptcy in [year]", "participated in [specific event] in [year]"
  BAD: "are people", "have done something", "exist", any property so broad it applies to millions

Generate:
- Q: "Which [entity type]s [shared property P]?"
  The question must be specific enough that the answer is an enumerable bounded list.
- A: "[Entity X] and [Entity Y] [and Entity Z if applicable].
  [1 confirming sentence per entity linking them to the shared property]."

The question must not name any of the entities being listed.
The answer must enumerate all matching entities found across the anchor and candidate docs.

---

## RULES (apply to both types)

- Never name the target entity/entities in the question
- No pronouns: replace every "he", "she", "it", "they", "the company" with the actual entity name
- No relative time: replace "recently", "last year", "currently" with absolute dates
- No document references: no "the article", "as mentioned", "the document states"
- All dates must be absolute
- Answers must lead with entity name(s), followed by confirming context
- Do NOT generate QA pairs where all facts come from only one doc
  (single-doc entity surfacing is already handled upstream)
- Return an empty list if no cross-doc connections of either type are found

---

## OUTPUT
Return parseable JSON only — no preamble, postamble, or markdown fences:
{{
  "crossdoc_qa_pairs": [
    {{
      "type": "converging_clues",
      "question": "...",
      "answer": "...",
      "source_doc_ids": ["doc_id_1", "doc_id_2"]
    }},
    {{
      "type": "parallel_property",
      "question": "...",
      "answer": "...",
      "source_doc_ids": ["doc_id_1", "doc_id_2"]
    }}
  ]
}}

Return crossdoc_qa_pairs: [] if no cross-doc connections of either type are found.
"""
