"""Prompts of the replica pipeline: the author brief (what a Jev-class item is, the tier's rules, the family), the
reviewer's blind pass and the reviewer's verdict pass. The rules follow the published JevBench hard-tier authoring spec
in our own words. Every call returns one JSON object after the model's own reasoning."""
from __future__ import annotations

import json

from .spec import FAMILIES

AUTHOR_SYSTEM = """You write evaluation items for Jev-class typed-decision models. Such a model reads a STATE (a text or a JSON object) and a QUESTION with a closed set of options, and returns a probability for every option; it is scored on argmax accuracy and on calibration. You write one item per request, in English. Think it through first, then output a single JSON object and nothing after it.

Rules every item must satisfy:
- Exactly one defensible gold answer that a careful expert picks from the state alone. Difficulty comes from reasoning over the state, never from vague wording, trick grammar or knowledge that is not in the state.
- Self-contained: everything needed is in state + question. Fictional but realistic businesses, systems and people; no real persons, no confidential or harmful content.
- The state never contains the gold answer as a field or a hint. The keys expected, label, answer_key, ground_truth and gold are banned. No sentence may point at the decisive fact or at the trap.
- Each item is its own scenario: never a template with swapped nouns. Specific names, figures and dates.
- Question types, exactly as the harness expects:
  - noul: expected is "yes" or "no"; criteria is {"true": "<when the answer is yes>", "false": "<when it is no>"}.
  - choice: 3-7 options with lowercase snake_case ids; criteria is {id: "<one-line definition>"} covering every option, listed in a neutral order; expected is one id.
  - score: 3-6 ordered levels; criteria is a list of level descriptions from lowest to highest; expected is the gold level index as an integer (0 is the first level).
- Option definitions (criteria) say what an option means, never whether it applies to this state: no "not mentioned", "correct", "the right one" or similar, and never a restatement of the rationale or of the decisive fact (an option "apply the surcharge because the contract overrides the waiver" gives the answer away; "apply the surcharge" does not).
- instructions: the decision to make, in 1-3 sentences, without hints.
- rationale: 2-6 sentences a reviewer can check in two minutes, citing the decisive facts and the arithmetic when there is any.
- surface_answer: the tempting wrong option (an id, "yes"/"no", or a level index), or null. why_hard: one sentence.

Output format, one JSON object with exactly these keys:
{"state": <string or object>, "question": {"type": "noul" | "choice" | "score", "instructions": "...", "criteria": ...}, "expected": <gold>, "rationale": "...", "surface_answer": <option or null>, "why_hard": "...", "gold_probs": <object or null>}"""

TIER_RULES = {
    "easy": "Easy tier: the answer is named or stated explicitly in the state and a careful reader cannot miss it. "
            "A short state, no trap, no distractor.",
    "standard": "Standard tier: one or two conditions must be applied; a careless reader may pick the surface answer, a "
                "careful one cannot. Irrelevant material is realistic and short.",
    "judge": "Judge tier: the state is a JSON object {\"request\": ..., \"response\": ...} (a reference answer only when "
             "the family asks for one). The response is fluent and plausible; the verdict depends on one specific, "
             "checkable property: one error, one unmet constraint, or none. Nothing in the request or the response "
             "telegraphs the verdict.",
    "hard": "Hard tier, strict: a strong small decision model should miss 20-40 % of such items while a careful expert "
            "gets at least 95 % right. Require two or more computation steps or three or more interacting conditions. "
            "The decisive facts sit far apart. No sentence in the state may point at the trap or say which fact is "
            "decisive. Filler is relevant-looking material with near-misses (superseded revisions, same-name records "
            "with other values, a rule for a neighbouring case type) that must be read and ruled out; decoys look like "
            "normal case material. The 'exception applies' answer is the gold in at most half of such items. The state "
            "never contains a summary, decision log, verdict, end-of-day snapshot, note or comment that states the "
            "conclusion, the final status or which rule wins: entries may state facts (a date, an amount, a status "
            "change at a time), and the reader must derive the answer from them.",
}

BLIND_SYSTEM = ("You review evaluation items for typed-decision models. Answer the question from the state alone, "
                "carefully and completely, then output one JSON object and nothing after it.")

VERDICT_SYSTEM = ("You review evaluation items for typed-decision models. You see the author's gold answer and rationale "
                  "(and your own blind answer, when you gave one). Reject when the gold is wrong, when another option is equally "
                  "defensible, when the item needs knowledge that is not in the state, when the state leaks or points at "
                  "the answer or the trap, when an option definition gives the answer away (it restates the rationale, "
                  "names the decisive fact, or describes a candidate's quality instead of naming it), when the state is "
                  "not self-contained, or when the format rules of the question type are broken. Your own blind miss is "
                  "not by itself a reason to reject: an item may simply be hard. "
                  "Output one JSON object and nothing after it.")


def gold_line(b: dict) -> str:
    if b["type"] == "noul":
        return f'The gold answer must be "{b["gold"]}".'
    if b["type"] == "score":
        return f'Use exactly {b["levels"]} levels; the gold level index must be {b["gold"]}.'
    return f"Use {b['n_options']} options; the gold may sit at any position."


def author_prompt(b: dict) -> list[dict]:
    fam = FAMILIES[b["family"]]
    shape = ("as a JSON object with realistic fields" if b["json_state"] else "of plain text")
    lines = [f'Write ONE "{b["type"]}" item of the family "{b["family"]}" at the {b["tier"]} tier.',
             f"Family: {fam['brief']}",
             TIER_RULES[b["tier"]],
             f"Topic area: {b['topic']}. Setting: {b['setting']}. Scenario seed {b['seed']}: invent the specifics "
             "yourself and make them differ from any other item you can think of.",
             f"State length: about {b['words']} words, {shape}.",
             gold_line(b)]
    if b["family"] == "probability":
        lines.append("gold_probs: an object with a probability for every option, summing to 1, derived exactly from the "
                     "evidence in the state (show the arithmetic in the rationale); expected is the most likely option, "
                     "never a tie, with probability between 0.55 and 0.85.")
    else:
        lines.append("gold_probs: null.")
    lines.append("Plan the decision, the decisive facts and the trap; write the state; re-derive the answer from the state "
                 "exactly as written; then output the JSON object.")
    if b["tier"] == "hard":
        lines.append("Final check before output: a reviewer rejects the item if any sentence of the state states the "
                     "conclusion, the final status or which rule wins. Re-read the state and delete such sentences; "
                     "keep only the facts the reader needs to derive the answer.")
    return [{"role": "system", "content": AUTHOR_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def _question_view(item: dict) -> dict:
    return {"state": item["state"], "question": item["question"], "labels": item["labels"]}


TIER_NOTE = {
    "easy": "This is an easy-tier item: the state is meant to state the answer plainly, which is not a leak. Reject it "
            "only when the gold is wrong, another option is equally valid, or the format rules are broken.",
    "standard": "This is a standard-tier item: one or two conditions to apply; a plainly stated decisive fact is "
                "acceptable as long as the answer still needs the condition.",
    "judge": "This is a judge-tier item: the verdict depends on one checkable property of the response; check that "
             "property yourself before judging the gold.",
    "hard": "This is a hard-tier item: the state must not point at the decisive fact or the trap, and the answer must "
            "need two or more steps or three or more interacting conditions.",
}


def blind_prompt(item: dict) -> list[dict]:
    view = _question_view(item)
    labels = ", ".join(f'"{l}"' for l in item["labels"])
    ask = ('\n\nAnswer with one JSON object with exactly these keys: "answer" (one of the labels: ' + labels +
           '), "probs" (an object with a probability for every label, summing to 1), "note" (one sentence on the '
           'decisive fact). Write the values, not placeholders.')
    return [{"role": "system", "content": BLIND_SYSTEM},
            {"role": "user", "content": json.dumps(view, ensure_ascii=False) + ask}]


def verdict_prompt(item: dict, blind: dict | None) -> list[dict]:
    prov = item.get("provenance", {})
    view = _question_view(item)
    if blind is not None:
        view["your_blind_answer"] = blind.get("answer")
    view.update({"gold": item["expected"], "rationale": prov.get("rationale"), "surface_answer": prov.get("surface_answer"),
                 "why_hard": prov.get("why_hard"), "gold_probs": prov.get("gold_probs")})
    ask = ("\n\n" + TIER_NOTE.get(item.get("tier", "standard"), TIER_NOTE["standard"]) +
           '\nAnswer with one JSON object with exactly these keys: "verdict" (the word accept or the word reject), '
           '"reason_class" (one word: gold_wrong when the gold answer is wrong or another option is equally defensible; '
           'outside_knowledge; leak when the state or an option definition gives the answer away; format; other; '
           'or none when you accept), "reason" (one or two sentences), "fix" (what would make the item acceptable, or '
           'null). Write the values, not placeholders.')
    return [{"role": "system", "content": VERDICT_SYSTEM},
            {"role": "user", "content": json.dumps(view, ensure_ascii=False) + ask}]
