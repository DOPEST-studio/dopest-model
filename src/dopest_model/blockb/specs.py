"""What the teacher writes: families, decision types, label sets and the briefs that diversify each call.

Taxonomy and wording are this project's own. OpenDecisions is our evaluation set, so nothing here copies its items
or phrasings; only the broad kinds of decisions (routing a request, judging an answer, rule-based screening) follow
the benchmark's families.
"""
from __future__ import annotations

import random

TIERS = ("easy", "standard", "judge", "hard")

# ------------------------------------------------------------------ llm_router: one user request, several questions

ROUTER_QUESTIONS = [
    {"key": "task_type", "type": "choice",
     "ask": ["What kind of task is this request, for picking a specialist model?",
             "Classify the main job the user wants done."],
     "labels": {"coding": "Write, fix, review or explain code, queries or scripts.",
                "math": "Solve a calculation, proof or quantitative problem with a definite answer.",
                "creative_writing": "Compose fiction, poetry, jokes or other original creative text.",
                "professional_writing": "Draft or edit business, academic or formal text (emails, reports, CVs).",
                "knowledge_qa": "Answer a factual or explanatory question.",
                "advice": "Give a recommendation or guidance on a personal or practical decision.",
                "data_analysis": "Compute statistics, trends or insights over data the user provides.",
                "extraction": "Pull given details into a fixed structure (fields, JSON, table).",
                "translation": "Translate text between languages.",
                "summarization": "Condense text the user provides.",
                "chitchat": "Casual conversation with no task."}},
    {"key": "needs", "type": "choice",
     "ask": ["What does the system need to add before a model can answer this well?",
             "Which resource, if any, must be attached to this request?"],
     "labels": {"nothing": "The model can answer from the request alone.",
                "web_search": "Current or external information must be looked up.",
                "code_execution": "Code must actually run, or exact computation over data is needed.",
                "clarification": "The request is too ambiguous to answer without asking the user first.",
                "user_file": "The request refers to a document or file that was not included."}},
    {"key": "specialist", "type": "choice",
     "ask": ["Which specialist model should take this request?"],
     "labels": {"code_model": "Tuned for programming.", "math_model": "Tuned for mathematics and quantitative work.",
                "document_model": "Tuned for long documents: summarising, extracting, answering over given text.",
                "general_model": "General conversation, writing and knowledge.",
                "translation_model": "Tuned for translation."}},
    {"key": "tier", "type": "choice",
     "ask": ["Which is the cheapest model tier that can still answer this request well?",
             "Pick the least expensive tier that handles this request without a loss in quality."],
     "labels": {"small_fast": "Small fast model: simple, short, low-stakes requests.",
                "mid": "Mid-size model: ordinary multi-step tasks.",
                "frontier": "Frontier model: demanding knowledge, writing or code.",
                "frontier_reasoning": "Frontier model with extended reasoning: hard proofs, tricky logic, complex debugging."}},
    {"key": "difficulty", "type": "score",
     "ask": ["How hard is this request for a language model to answer correctly? Output the level index."],
     "levels": ["Trivial", "Easy", "Moderate", "Hard", "Very hard (expert level)"]},
    {"key": "length", "type": "score",
     "ask": ["What length should a good reply to this message have? Output the level index."],
     "levels": ["One line or a single value", "A short paragraph", "Several paragraphs or a short list",
                "A long, structured document"]},
    {"key": "effort", "type": "score",
     "ask": ["How much reasoning effort should the model spend before answering? Output the level index."],
     "levels": ["None: answer directly", "Low", "Medium", "High"]},
    {"key": "asks_code", "type": "noul",
     "ask": ["Is the user asking for code, a query or a script (to write, fix or explain)?"],
     "true": "Code, a query or a script is central to the request.", "false": "No code is needed."},
    {"key": "single_answer_math", "type": "noul",
     "ask": ["Is this a math problem whose final answer is a single number or expression?"],
     "true": "It has one definite numeric or symbolic answer.", "false": "It does not."},
    {"key": "short_answer", "type": "noul",
     "ask": ["Can this request be answered well in one or two sentences?"],
     "true": "One or two sentences are enough.", "false": "A good answer needs more."},
    {"key": "needs_current_info", "type": "noul",
     "ask": ["Does a correct answer depend on information more recent than a model's training data?"],
     "true": "Up-to-date information is required.", "false": "Stable knowledge is enough."},
]

# a JSON object instead of text as the state: OpenDecisions has one in 30% of routing and 47% of usecase items;
# the shares are higher here because the first batch was written with text states only
JSON_SHARE = {"llm_router": 0.4, "usecase": 0.6}
ROUTER_JSON = [
    ('{"conversation": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}',
     "a chat of 2 to 6 turns ending with a user turn; the routing questions are about that latest user turn, and the "
     "earlier turns may change what it needs"),
    ('{"request": "...", "models": [{"name": "...", "price_per_1m_tokens": ..., "good_at": "..."}, ...], '
     '"operator_rule": "...", "note": "..."}',
     "a request plus the operator's model catalog and routing rule; the routing decisions must follow the rule"),
    ('{"conversation": [...], "attachments": [{"name": "...", "type": "...", "summary": "..."}]}',
     "a chat whose latest user turn refers to attached files, described by name, type and a short summary"),
]
CATALOG_QUESTION = {"key": "pick_model", "type": "choice",
                    "ask": ["Which model must the router choose under the operator's rule?"]}
USECASE_JSON = ('{"request": "...", "response": "..."}, {"request": "...", "reference": "...", "response": "..."}, '
                '{"ticket": "...", "policy": "...", "agent_reply": "..."}, {"query": "...", "passages": {"A": "...", '
                '"B": "..."}}, {"listing": "...", "rules": "..."} or {"code": "...", "request": "...", "response": "..."}')

ROUTER_DOMAINS = ["software engineering", "school homework", "personal finance", "cooking", "travel planning",
                  "small business", "academic research", "health and fitness (non-diagnostic)", "law and paperwork",
                  "marketing", "data science", "home repair", "language learning", "gaming", "parenting",
                  "job search", "customer service", "devops", "spreadsheets", "creative hobbies"]
ROUTER_STYLES = ["terse, lowercase, with a typo", "polite and detailed", "pastes a snippet of code or data",
                 "multi-part with two questions", "vague and underspecified", "casual chat tone",
                 "formal workplace tone", "includes a constraint (word limit, format, deadline)",
                 "refers to an earlier message or a file", "mixes small talk with a real task"]

# ------------------------------------------------------------------ usecase: business decisions on given material

USECASES = {
    "customer_support": ["classify the issue type of a support message", "pick the queue a ticket goes to",
                         "decide whether a refund request meets the stated refund policy"],
    "llm_guardrails": ["judge whether a model's answer follows the system instructions it was given",
                       "decide if a user prompt should be blocked under a stated content policy",
                       "grade a model's answer against a reference answer on a stated scale"],
    "code_linting": ["judge whether a code change does what the commit message claims",
                     "decide which listed issue a code snippet has, if any"],
    "feature_extraction": ["find which field of an extraction (given next to the source text) is wrong, if any",
                           "decide whether an extracted value matches the source exactly"],
    "legal_compliance": ["classify which kind of contract or document an excerpt comes from",
                         "decide whether a described action complies with a quoted clause"],
    "financial_crime": ["decide the next step for a transaction alert under a stated review procedure",
                        "classify which red-flag pattern a set of transactions shows, if any"],
    "insurance_claims": ["pick the claim decision under the quoted policy wording"],
    "ecommerce_marketplaces": ["pick the category a product listing belongs to",
                               "decide whether a listing violates a stated marketplace rule"],
    "search_retrieval": ["pick which passage best answers a query", "rate how relevant a document is to a query"],
    "lead_generation": ["classify what an inbound contact-form message wants (sale, support, partnership, job...)"],
    "moderation_trust_safety": ["decide the action on a post under a stated community rule"],
    "knowledge_graphs": ["decide which relation between two entities a sentence states"],
}

# variety for long runs: where the first item is set, and a complication for judge/hard items
SECTORS = ["healthcare clinic", "logistics and freight", "fintech app", "grocery retail", "public library",
           "university admissions office", "municipal permits office", "telecom provider", "insurance broker",
           "SaaS startup", "hotel chain", "car rental", "energy utility", "online marketplace", "law firm",
           "manufacturing plant", "airline", "non-profit charity", "real estate agency", "game studio",
           "pharmacy chain", "construction contractor", "travel agency", "bank branch", "food delivery platform"]
TWISTS = ["a number sits exactly at, or just next to, a threshold",
          "two rules interact and an exception overrides the general rule",
          "a required document or fact is missing, and the procedure says what happens then",
          "a tempting but irrelevant detail is mentioned",
          "dates must be counted (deadline, tenure or waiting period) from a stated reference date",
          "a later note corrects an earlier statement",
          "the case qualifies under one route but not under another",
          "a policy changed and the date decides which version applies",
          "amounts must be summed or converted before comparing them with a limit",
          "several issues are present and the most severe one decides"]


def _twist(rng: random.Random, tier: str) -> str | None:
    return rng.choice(TWISTS) if tier in ("judge", "hard") else None


def _yes_text(b: dict) -> str:
    """Yes/no calls fix how many of the three answers are yes, so the answers come out balanced."""
    return f" Make the correct answer \"yes\" in {b['yes']} of the 3 items and \"no\" in the others." if b["type"] == "noul" else ""


def _twist_text(b: dict) -> str:
    return f" Build this complication into the items where it fits naturally: {b['twist']}." if b.get("twist") else ""


# ------------------------------------------------------------------ fairness: rule-based screening; code swaps names

FAIRNESS_SETTINGS = [
    "job applicant screening against listed requirements", "rental application under stated criteria",
    "grant or benefit application under program rules", "loan pre-check under stated thresholds",
    "insurance claim triage under policy rules", "school admission check against published criteria",
    "airline boarding or rebooking under fare rules", "promotion review under a written policy",
    "content moderation under a group's posted rules", "fraud alert review under a stated procedure",
]

# ------------------------------------------------------------------ prompts

SYSTEM = """You write evaluation items for a DECISION MODEL: a model that reads a STATE (text) and a typed QUESTION with a fixed set of options, and returns a probability for each option. You write in English unless told otherwise. Reply with a single JSON object.

Every item must have exactly ONE defensible answer that a careful expert would choose from the state alone (no outside facts beyond common knowledge). Prefer items that require reading carefully: include one plausible wrong option (the trap) that a hasty reader would pick, and say why it is wrong in the rationale. Never put the answer's wording in the state. If the decision depends on the current date (deadlines, tenure, ages, waiting periods), state the reference date in the state, e.g. "Today is 14 March 2026." Vary names, places, companies, numbers and writing style; never reuse the examples' content.

Item JSON fields, in exactly this order (plan, write, check, and only then commit to an answer):
- "plan": one or two sentences, never shown to the decision model: the decision being tested, the intended answer, the trap, and the facts the state must contain so that exactly one answer follows.
- "state": what the model reads (realistic, self-contained), written to match the plan: a text, or a JSON object when the request asks for one.
- "question": {"type": "choice" | "noul" | "score", "instructions": "...", "criteria": ...}
  - choice: "criteria" is an object {option_id: short description}, 3 to 6 options, ids in snake_case.
  - noul (yes/no): "criteria" is {"true": "...", "false": "..."}.
  - score (ordered levels): "criteria" is a list of level descriptions from lowest to highest.
- "reasoning": re-derive the answer from the state exactly as written (not from the plan), step by step: check every rule, threshold, date and number that matters, then say which option follows and why the tempting one does not (2 to 5 sentences). The state can no longer be changed at this point.
- "expected": the option the reasoning arrived at: the option_id (choice), "yes" or "no" (noul), or the level index as an integer (score). Use null if the check shows that the state as written does not lead to exactly one option; such an item is discarded.
- "trap_answer": the tempting wrong option_id (choice only, else null).
- "tier": "easy" | "standard" | "judge" | "hard" (hard = needs several reasoning steps or careful reading of conflicting details).

Shape: {"items": [{"plan": "...", "state": "...", "question": {...}, "reasoning": "...", "expected": ..., "trap_answer": ..., "tier": "..."}]}"""


def router_brief(rng: random.Random) -> dict:
    fmt = rng.choice(ROUTER_JSON) if rng.random() < JSON_SHARE["llm_router"] else None
    catalog = fmt is not None and '"models"' in fmt[0]
    qs = [CATALOG_QUESTION] + rng.sample(ROUTER_QUESTIONS, 2) if catalog else rng.sample(ROUTER_QUESTIONS, 3)
    qs = [dict(q, answer=rng.choice(["yes", "no"])) if q["type"] == "noul" else q for q in qs]   # balanced yes/no
    return {"family": "llm_router", "use_case": "model_routing", "questions": qs, "json": fmt,
            "domain": rng.choice(ROUTER_DOMAINS), "style": rng.choice(ROUTER_STYLES),
            "tier": rng.choices(TIERS, weights=[1, 1, 3, 3])[0]}


def _router_spec(q: dict) -> str:
    if q["key"] == "pick_model":
        return (f'- choice "{q["ask"][0]}" Options: the models in the catalog, with their names in snake_case as '
                f'option ids; exactly one of them satisfies the rule.')
    if q["type"] == "choice":
        opts = "; ".join(f"{k} = {v}" for k, v in q["labels"].items())
        return (f'- choice "{q["ask"][0]}" Use 4-5 of these options (always include the correct one and a plausible '
                f'trap): {opts}')
    if q["type"] == "score":
        return f'- score "{q["ask"][0]}" Levels: {q["levels"]}'
    return (f'- noul "{q["ask"][0]}" true = {q["true"]} false = {q["false"].rstrip(".")}. Write the message so that the correct '
            f'answer is "{q["answer"]}".')


def router_prompt(b: dict) -> str:
    if b.get("json"):
        shape, what = b["json"]
        head = (f"Write ONE realistic exchange with an AI assistant, about {b['domain']}, with the user writing "
                f"{b['style']}, as a JSON object of this shape: {shape} ({what}).")
        share = 'all sharing that JSON object as their "state" (copy the object verbatim into each item)'
    else:
        head = f"Write ONE realistic message a user sends to an AI assistant, about {b['domain']}, written {b['style']}."
        share = 'all sharing that message as their "state" (copy it verbatim into each item)'
    return (f"{head} Aim for difficulty tier \"{b['tier']}\": the routing decisions below should not be obvious from "
            f"keywords alone. Then write one item per routing question below, {share}:\n"
            + "\n".join(_router_spec(q) for q in b["questions"])
            + '\nReturn {"items": [ ... ]} with exactly one item per question, in the order given.')


def usecase_brief(rng: random.Random) -> dict:
    uc = rng.choice(list(USECASES))
    tier = rng.choices(TIERS, weights=[1, 2, 3, 4])[0]
    return {"family": "usecase", "use_case": uc, "task": rng.choice(USECASES[uc]),
            "type": rng.choices(["choice", "noul", "score"], weights=[5, 3, 2])[0], "tier": tier,
            "sector": rng.choice(SECTORS), "twist": _twist(rng, tier), "json": rng.random() < JSON_SHARE["usecase"],
            "yes": rng.choice([1, 2])}


def usecase_prompt(b: dict) -> str:
    return (f"Write 3 different items for this business decision: {b['task']} (use case: {b['use_case']}). "
            f"Each state contains the material to judge (message, document excerpt, model answer, listing, rule text, "
            f"as needed) and, where a rule applies, the rule itself. Question type: {b['type']}. Target tier: "
            f"\"{b['tier']}\". Set the first item in this kind of organisation: {b['sector']}; use other companies, "
            f"sectors and details for the others.{_twist_text(b)}{_yes_text(b)}"
            + (f" Write each state as a JSON object (not a string) whose fields hold the material, for example "
               f"{USECASE_JSON}; choose field names that fit the task." if b.get("json") else "")
            + ' Return {"items": [ ... ]}.')


def fairness_brief(rng: random.Random) -> dict:
    tier = rng.choices(TIERS, weights=[2, 3, 2, 3])[0]
    return {"family": "fairness", "use_case": "fairness", "setting": rng.choice(FAIRNESS_SETTINGS),
            "type": rng.choices(["choice", "noul", "score"], weights=[5, 3, 2])[0], "tier": tier,
            "twist": _twist(rng, tier), "yes": rng.choice([1, 2])}


def fairness_prompt(b: dict) -> str:
    return (f"Write 3 different items for: {b['setting']}. Each state states the rules (criteria, thresholds, "
            f"procedure) and then describes one person's case, including their full name and, naturally, some "
            f"personal details that the rules do NOT use (such as age, gender, nationality, religion or family "
            f"status). The correct decision must follow from the rules and the relevant facts only. Mark the person's "
            f"full name exactly as written in a field \"person_name\" of each item, so it can be swapped later. "
            f"Question type: {b['type']}. Target tier: \"{b['tier']}\".{_twist_text(b)}{_yes_text(b)} "
            f"Return {{\"items\": [ ... ]}}.")


# Two templates first matched OpenDecisions' wording verbatim (found by scripts/contamination.py on 2026-10-23);
# items written with them get the new wording when the set is assembled.
REWORDED = {
    "Does the request ask for code, a query or a script to be written, fixed or explained?":
        "Is the user asking for code, a query or a script (to write, fix or explain)?",
    "How long does a good answer to this request need to be? Output the level index.":
        "What length should a good reply to this message have? Output the level index.",
    "How long does a good answer to this request need to be?":
        "What length should a good reply to this message have? Output the level index.",
}

FAMILIES = {"llm_router": (router_brief, router_prompt), "usecase": (usecase_brief, usecase_prompt),
            "fairness": (fairness_brief, fairness_prompt)}

# ------------------------------------------------------------------ languages (benchmark: routing and usecase only)

LANGUAGES = {"pl": "Polish", "es": "Spanish", "pt": "Portuguese", "ja": "Japanese", "de": "German",
             "fr": "French", "it": "Italian", "nl": "Dutch", "da": "Danish", "sv": "Swedish", "no": "Norwegian",
             "fi": "Finnish", "cs": "Czech", "uk": "Ukrainian", "el": "Greek", "tr": "Turkish", "ar": "Arabic",
             "hi": "Hindi", "zh": "Simplified Chinese", "ko": "Korean", "id": "Indonesian",
             "mixed": "a code-switched mix of English and another language"}
# items per language in OpenDecisions' multilingual family: the default shares of non-English calls
LANGUAGE_WEIGHTS = {"pl": 38, "es": 34, "pt": 33, "ja": 30, "de": 29, "da": 24, "fr": 22, "it": 20, "ar": 18,
                    "uk": 18, "zh": 18, "mixed": 18, "hi": 16, "tr": 16, "el": 16, "ko": 16, "nl": 13, "fi": 11,
                    "cs": 11, "no": 8, "id": 7, "sv": 6}
MULTILINGUAL_FAMILIES = {"llm_router": 0.45, "usecase": 0.55}


def language_rule(code: str) -> str:
    if code == "en":
        return ""
    keep = (" Keep option ids in English snake_case and keep the JSON field names in English. The plan and the "
            "reasoning may be in English.")
    if code == "mixed":
        return ("\nLANGUAGE: write the state the way people really write when they switch between English and another "
                "language mid-sentence (for example German with English tech terms, Hinglish in Latin script, Spanish "
                "mixed with English); write the question instructions and every option description or level in that "
                "other language." + keep)
    return (f"\nLANGUAGE: write everything the decision model reads (the state, the question instructions and every "
            f"option description or level) in {LANGUAGES[code]}, as a native speaker would write it, with names, "
            f"places, currencies, dates and formats from a country where it is spoken." + keep)
