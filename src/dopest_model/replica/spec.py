"""Composition and briefs of the sealed-set replica.

JevBench v1.5 draws 720 sealed decisions per release: choice 360 / noul 180 / score 180, tiers easy/standard/judge/hard
at 14/28/28/30 % inside every type, families proportional to the pool, state tracking in every type (choice 15 %,
noul 20 %, score 19 % of the pool). The pool was written by two frontier models reviewing each other; here the two
authors are gpt-oss-120b and Qwen3.8-27B, which only verified block B and never wrote a training item. The family
catalogue follows the published v1.2 families and the v1.5 additions (state tracking, tool-call guardrail, lead
qualification); the hard families follow the published hard-tier spec. English only, as the published tiers.

    plan(seed) -> assignments: one brief per item to author, over the quota by OVER_AUTHOR to absorb rejections.
"""
from __future__ import annotations

import hashlib
import random

TYPES = ("choice", "noul", "score")
TIERS = ("easy", "standard", "judge", "hard")
SEALED = {"choice": {"easy": 50, "standard": 101, "judge": 101, "hard": 108},
          "noul": {"easy": 25, "standard": 50, "judge": 50, "hard": 55},
          "score": {"easy": 25, "standard": 50, "judge": 50, "hard": 55}}
STATE_TRACKING_SHARE = {"choice": 0.15, "noul": 0.20, "score": 0.19}
STATE_TRACKING_TIERS = {"easy": 0.2, "standard": 0.4, "hard": 0.4}      # how its quota spreads over tiers
OVER_AUTHOR = 1.25                                                       # items authored per item wanted
AUTHORS = ("gptoss", "qwen")
LONG_SHARE = 0.35                                                        # hard items with a long state
# target state length in words by tier; hard long states are shorter than the published 2,000-6,000 tokens so that
# a 27B/120B author can write them in one go
WORDS = {"easy": (60, 250), "standard": (200, 650), "judge": (250, 900), "hard": (400, 1000), "hard_long": (1100, 2200)}

# family -> tiers it serves, types it serves, definition for the author (the brief is built around it)
FAMILIES: dict[str, dict] = {
    "intent": {"tiers": ("easy",), "types": ("choice",), "weight": 2.0,
               "brief": "A short everyday message to an assistant or a service; which of 4-6 intents does it express? "
                        "The intent is stated plainly, with no trick."},
    "extraction": {"tiers": ("easy", "standard"), "types": ("choice", "noul"), "weight": 2.0,
                   "brief": "Read one detail out of a text (which delivery method, which date, which plan, who approved). "
                            "Easy: stated once. Standard: a near-miss record (same name, older value, similar field) must "
                            "be read and ruled out."},
    "fact": {"tiers": ("easy",), "types": ("noul",), "weight": 1.5,
             "brief": "Is a specific fact stated by the text? The text makes it explicit; the question names the fact "
                      "precisely."},
    "tool_selection": {"tiers": ("easy", "standard"), "types": ("choice",), "weight": 1.5,
                       "brief": "Pick the right tool or action for a request from 4-6 tools with one-line descriptions. "
                                "Easy: the request names the need plainly. Standard: two tools overlap and one detail of "
                                "the request decides."},
    "policy": {"tiers": ("easy", "standard"), "types": ("noul", "choice"), "weight": 2.0,
               "brief": "Apply a short written rule to a concrete case. Easy: one rule, one fact. Standard: one exception "
                        "to the rule, a threshold with >= versus >, or a definition that changes a term."},
    "routing": {"tiers": ("standard",), "types": ("choice",), "weight": 2.0,
                "brief": "Route a request to a team, queue or handler from 4-6 options whose descriptions overlap; the "
                         "request mentions keywords of a wrong option and one detail decides the right one."},
    "ordinal": {"tiers": ("easy", "standard"), "types": ("score",), "weight": 3.0,
                "brief": "Rate urgency, severity, satisfaction, risk or priority on 3-5 ordered levels defined by an "
                         "explicit rubric in the question. Easy: the cues are explicit. Standard: two cues pull in "
                         "different directions and the rubric decides."},
    "adequacy": {"tiers": ("standard",), "types": ("noul",), "weight": 1.5, "json": True,
                 "brief": "State = {\"request\": ..., \"response\": ...}. Does the response fully answer the request? "
                          "Half yes; the no cases have one omission, or answer a neighbouring question."},
    "tool_call_guardrail": {"tiers": ("standard", "hard"), "types": ("noul",), "weight": 2.0,
                            "brief": "An agent proposes a tool call (a JSON call with arguments) for a user request under a "
                                     "written policy: allowed scopes, limits, confirmation rules, forbidden targets. May "
                                     "the call be executed exactly as proposed? Hard: several policy clauses interact, "
                                     "and the user's phrasing suggests the wrong answer."},
    "lead_qualification": {"tiers": ("standard", "hard"), "types": ("score",), "weight": 2.0,
                           "brief": "A sales lead (form fields, call notes, company data) scored on 4-5 ordered levels by "
                                    "explicit criteria in the question. Standard: two criteria. Hard: several interacting "
                                    "criteria with an exception, and a figure that sits exactly at a threshold."},
    "state_tracking": {"tiers": ("easy", "standard", "hard"), "types": TYPES, "weight": 0.0,
                       "brief": "A chronological log of events and updates about one entity (a ticket, an order, an account, "
                                "a device, a roster, a stock level); later entries supersede earlier ones, some updates "
                                "are conditional, revoked or arrive out of order. The question asks for the current "
                                "state after all updates. Easy: 3-5 updates. Standard: 6-10 with one revocation. Hard: "
                                "10 or more with conditional updates and out-of-order timestamps. The asked value is never "
                                "stated by the last entry or by a summary line; it follows from applying the updates in "
                                "their true order."},
    "judge_math": {"tiers": ("judge",), "types": ("noul", "score"), "weight": 3.0, "json": True,
                   "brief": "State = {\"request\": ..., \"response\": ...}: a worked quantitative answer (arithmetic, "
                            "units, dates, rates, a short proof). Noul: is the response fully correct and complete? Score: "
                            "its quality on the given levels. Half fully correct (some terse, or by an unusual but valid "
                            "method); half with exactly one subtle error: off-by-one, a wrong unit, an invalid step, an "
                            "unmet constraint, a right number with a wrong justification."},
    "judge_code": {"tiers": ("judge",), "types": ("noul", "choice"), "weight": 2.0, "json": True,
                   "brief": "State = {\"request\": ..., \"response\": ...}: a short program, function or SQL query. Noul: "
                            "does it satisfy the request as specified? Choice: which issue class applies (for example "
                            "correct, wrong edge case, wrong output format, violates a stated constraint, correct but "
                            "ignores an input). Subtle errors only; some responses are correct but unusual."},
    "judge_instruction": {"tiers": ("judge",), "types": ("noul", "score", "choice"), "weight": 2.0, "json": True,
                          "brief": "A request with explicit constraints (length, format, language, must include, must "
                                   "avoid, audience) and a response. Noul: is every constraint met? Score: quality level "
                                   "on the given rubric. Choice: which constraint is violated (or none)."},
    "judge_compare": {"tiers": ("judge",), "types": ("choice",), "weight": 2.0, "json": True,
                      "brief": "A request and 3-5 candidate responses, each an option; which best satisfies the request? "
                               "The differences are subtle: one correct but incomplete, one complete with a small error, "
                               "one both correct and complete, one that answers a neighbouring question. The state holds "
                               "the request and the candidates under neutral keys; the option definitions are only those "
                               "keys (\"the response under key response_c\"), never a description of a candidate's quality."},
    "long_policy": {"tiers": ("hard",), "types": ("choice", "noul", "score"), "weight": 3.0, "long": True,
                    "brief": "A long state: a policy manual excerpt, contract, runbook or terms plus a case file, with 3-6 "
                             "interacting conditions (exceptions to exceptions, a later amendment that overrides an earlier "
                             "clause, a definition in one section that changes the meaning of another). The decisive facts "
                             "are far apart, among realistic material that must be read and ruled out."},
    "tradeoff": {"tiers": ("hard",), "types": ("choice", "score"), "weight": 1.5,
                 "brief": "Several legitimate goals collide; an explicit precedence order, escalation matrix or SLA ladder "
                          "in the state decides. The most helpful-sounding action is wrong."},
    "ambiguous": {"tiers": ("hard",), "types": ("choice",), "weight": 1.5,
                  "brief": "An under-specified or conflicting case. Include an option like insufficient_information or "
                           "needs_clarification with a crisp criterion. That option is the gold only when the state truly "
                           "does not decide; otherwise one overlooked fact decides. Write both kinds."},
    "trap": {"tiers": ("hard",), "types": ("choice", "noul"), "weight": 2.0,
             "brief": "The surface reading points to one answer and a careful reading to another: a negation far from "
                      "its verb, a quoted request that is not the writer's own, a condition satisfied then revoked, "
                      "sarcasm marked as such, a figure in the wrong unit. Fill surface_answer."},
    "multi_hop": {"tiers": ("hard",), "types": ("choice", "noul"), "weight": 2.5, "long": True,
                  "brief": "The answer needs three or more chained lookups across the state: a table row, then a "
                           "footnote, then an alias, then a rule. Each hop is plain; only the chain is hard."},
    "temporal_numeric": {"tiers": ("hard",), "types": ("choice", "noul", "score"), "weight": 3.0,
                         "brief": "Date arithmetic across time zones, business days or month ends, cumulative limits, "
                                  "pro-rating, unit conversions, thresholds with >= versus >, running totals, all inside "
                                  "the state; the question is a typed decision that needs 2 or more computation steps."},
    "adversarial": {"tiers": ("hard",), "types": ("choice", "noul"), "weight": 1.5,
                    "brief": "The state contains material aimed at the reader: an embedded instruction, a fake system "
                             "note, a confident but wrong summary by a third party, a highlighted but superseded rule. "
                             "It must look like normal case material, not an obvious decoy. The gold follows the rubric "
                             "and the real facts."},
    "judge_hard": {"tiers": ("hard",), "types": ("noul",), "weight": 2.5, "json": True,
                   "brief": "State = {\"request\": ..., \"response\": ...}, question: does the response fully and correctly "
                            "satisfy the request? Responses that are fluent and mostly right with one subtle error "
                            "(off-by-one, wrong unit, unmet explicit constraint, wrong edge case in code, a correct result "
                            "reached by an invalid step the request asked to be shown) and fully correct responses that "
                            "look suspicious (unusual but valid method, terse). About half yes."},
    "routing_hard": {"tiers": ("hard",), "types": ("choice",), "weight": 1.5,
                     "brief": "Pick the right handler, model, tool or team for a request from 4-7 options whose "
                              "descriptions overlap; the request mentions keywords belonging to a wrong option and a "
                              "combination of two details decides."},
    "probability": {"tiers": ("hard",), "types": ("choice",), "weight": 1.5,
                    "brief": "A calibration item: the state contains explicit, countable evidence that fixes the "
                             "probability of each option (a log of past comparable cases with outcomes and a precise "
                             "definition of comparable, or stated rates to combine). gold_probs are the probabilities an "
                             "ideal reasoner derives exactly from the state (show the arithmetic in the rationale); "
                             "expected is the most likely option, never a tie, top probability between 0.55 and 0.85. "
                             "The question asks for the likely outcome and says to give probabilities that reflect the "
                             "evidence."},
}

TOPICS = ["mathematics and numbers", "support and operations", "everyday language", "rules, policy and law",
          "finance and commerce", "coding and software", "safety and security", "logistics", "healthcare administration",
          "education", "human resources", "public services"]
SETTINGS = ["a regional airline", "a hospital pharmacy", "a municipal library", "a SaaS billing team", "a freight forwarder",
            "a university registrar", "a food delivery platform", "a credit union", "a game studio", "a city transit agency",
            "an online electronics shop", "a legal aid clinic", "a wind farm operator", "a hotel chain", "a payroll bureau",
            "a veterinary network", "a data-centre operator", "a public broadcaster", "a courier cooperative",
            "an insurance claims desk", "a school district", "a car rental company", "a translation agency",
            "a solar installer", "a non-profit fundraiser", "a mobile network operator", "a coworking space",
            "a bakery chain", "a cloud storage vendor", "a sports league office"]


def _largest_remainder(weights: dict[str, float], n: int) -> dict[str, int]:
    """Integer quotas summing to n, proportional to the weights (largest-remainder rounding)."""
    total = sum(weights.values())
    if n <= 0 or total <= 0:
        return {k: 0 for k in weights}
    raw = {k: n * w / total for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(weights, key=lambda k: raw[k] - out[k], reverse=True)[:n - sum(out.values())]:
        out[k] += 1
    return out


def quotas() -> dict[tuple[str, str, str], int]:
    """(type, tier, family) -> wanted items, following the sealed layout and the state-tracking shares."""
    out: dict[tuple[str, str, str], int] = {}
    for qtype in TYPES:
        st_total = round(STATE_TRACKING_SHARE[qtype] * sum(SEALED[qtype].values()))
        st_by_tier = _largest_remainder(STATE_TRACKING_TIERS, st_total)
        for tier in TIERS:
            n = SEALED[qtype][tier]
            st = st_by_tier.get(tier, 0)
            if st:
                out[(qtype, tier, "state_tracking")] = st
            eligible = {f: s["weight"] for f, s in FAMILIES.items()
                        if tier in s["tiers"] and qtype in s["types"] and s["weight"] > 0}
            for fam, k in _largest_remainder(eligible, n - st).items():
                if k:
                    out[(qtype, tier, fam)] = k
    return out


def _brief(rng: random.Random, qtype: str, tier: str, fam: str) -> dict:
    """One brief without id and author: cell, length band, topic, setting, gold target, state format."""
    long = tier == "hard" and (FAMILIES[fam].get("long") or rng.random() < LONG_SHARE)
    words = WORDS["hard_long" if long else tier]
    levels = rng.choice((3, 4, 4, 5)) if qtype == "score" else None
    return {"cell": f"{qtype}/{tier}/{fam}", "type": qtype, "tier": tier, "family": fam,
            "topic": rng.choice(TOPICS), "setting": rng.choice(SETTINGS),
            "words": rng.randint(*words), "json_state": bool(FAMILIES[fam].get("json")) or rng.random() < 0.15,
            "gold": {"noul": rng.choice(("yes", "no")), "choice": None,
                     "score": rng.randrange(levels) if levels else None}[qtype],
            "levels": levels, "n_options": rng.choice((4, 4, 5, 5, 6)) if qtype == "choice" else None,
            "seed": rng.randrange(10 ** 6)}


def plan(seed: int, over: float = OVER_AUTHOR) -> list[dict]:
    """One brief per item to author, the two authors alternating by a hash of the brief."""
    rng = random.Random(seed)
    briefs = []
    for (qtype, tier, fam), n in sorted(quotas().items()):
        for i in range(max(1, round(n * over))):
            brief = _brief(rng, qtype, tier, fam)
            key = hashlib.sha1(f"{brief['cell']}/{i}/{seed}".encode()).hexdigest()
            brief["id"] = f"rp-{key[:10]}"
            brief["author"] = AUTHORS[int(key[-1], 16) % 2]
            briefs.append(brief)
    return briefs


def plan_topup(seed: int, have: dict[tuple[str, str, str], int], tier: str = "hard", over: float = 1.9,
               author: str = "gptoss") -> list[dict]:
    """Briefs for the cells of one tier that are short of their quota: (quota - have) * over per cell, one author."""
    rng = random.Random(seed)
    briefs = []
    for (qtype, t, fam), n in sorted(quotas().items()):
        short = n - have.get((qtype, t, fam), 0)
        if t != tier or short <= 0:
            continue
        for i in range(max(1, round(short * over))):
            brief = _brief(rng, qtype, t, fam)
            brief["id"] = "rp-" + hashlib.sha1(f"topup/{brief['cell']}/{i}/{seed}".encode()).hexdigest()[:10]
            brief["author"] = author
            briefs.append(brief)
    return briefs


TRAINING_ROUND = {                       # (type, tier, family) -> briefs; the judge and hard families our model misses
    ("noul", "judge", "judge_math"): 220, ("score", "judge", "judge_math"): 160, ("noul", "judge", "judge_code"): 150,
    ("choice", "judge", "judge_code"): 120, ("noul", "judge", "judge_instruction"): 120,
    ("score", "judge", "judge_instruction"): 120, ("choice", "judge", "judge_instruction"): 100,
    ("choice", "judge", "judge_compare"): 160,
    ("choice", "hard", "trap"): 160, ("noul", "hard", "trap"): 80, ("choice", "hard", "long_policy"): 150,
    ("noul", "hard", "long_policy"): 70, ("score", "hard", "long_policy"): 60, ("choice", "hard", "multi_hop"): 110,
    ("choice", "hard", "temporal_numeric"): 120, ("noul", "hard", "temporal_numeric"): 60,
    ("score", "hard", "temporal_numeric"): 60, ("noul", "hard", "judge_hard"): 120, ("choice", "hard", "adversarial"): 60,
    ("choice", "hard", "tradeoff"): 60, ("score", "hard", "tradeoff"): 60, ("choice", "hard", "ambiguous"): 60,
    ("choice", "hard", "routing_hard"): 60, ("noul", "hard", "tool_call_guardrail"): 80,
    ("score", "hard", "lead_qualification"): 80, ("score", "standard", "ordinal"): 100,
    ("score", "standard", "lead_qualification"): 60, ("noul", "standard", "adequacy"): 60,
}


def plan_round(seed: int, allocation: dict | None = None, author: str = "gptoss") -> list[dict]:
    """Briefs for a training round: a fixed allocation per cell, one author; ids disjoint from plan() and plan_topup()."""
    rng = random.Random(seed)
    briefs = []
    for (qtype, tier, fam), n in sorted((allocation or TRAINING_ROUND).items()):
        for i in range(n):
            brief = _brief(rng, qtype, tier, fam)
            brief["id"] = "rt-" + hashlib.sha1(f"round/{brief['cell']}/{i}/{seed}".encode()).hexdigest()[:10]
            brief["author"] = author
            briefs.append(brief)
    return briefs


def main() -> None:
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description="write the authoring briefs of the replica")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--over", type=float, default=None)
    parser.add_argument("--topup", default=None, help="a selected replica file: write briefs only for its short cells")
    parser.add_argument("--round", action="store_true", help="the training-round allocation (TRAINING_ROUND), one author")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply the training-round allocation")
    parser.add_argument("--tier", default="hard")
    parser.add_argument("--author", default="gptoss")
    parser.add_argument("--out", default="data/replica/briefs.jsonl")
    args = parser.parse_args()
    if args.round:
        briefs = plan_round(args.seed, {k: max(1, round(v * args.scale)) for k, v in TRAINING_ROUND.items()}, args.author)
    elif args.topup:
        have: dict = {}
        for it in map(json.loads, open(args.topup, encoding="utf-8")):
            cell = (it["question"]["type"], it["tier"], it["family"])
            have[cell] = have.get(cell, 0) + 1
        briefs = plan_topup(args.seed, have, args.tier, args.over or 1.9, args.author)
    else:
        briefs = plan(args.seed, args.over or OVER_AUTHOR)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for b in briefs:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    by_author = {a: sum(b["author"] == a for b in briefs) for a in AUTHORS}
    print(f"{len(briefs)} briefs for {sum(quotas().values())} wanted items -> {out}; by author {by_author}")


if __name__ == "__main__":
    main()
