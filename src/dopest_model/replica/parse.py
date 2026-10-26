"""Reading the models' outputs: the JSON object after the reasoning, and one authored item into the project's item
format with every format rule checked (the published spec's validate step)."""
from __future__ import annotations

import hashlib
import json
import re

from ..blockb.writer import shuffled_options, state_key

BANNED_KEYS = ("expected", "label", "answer_key", "ground_truth", "gold")
BANNED_IN_TEXT = re.compile(r'"(?:expected|answer_key|ground_truth|gold)"\s*:')
SNAKE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
NOUL = {"yes": "yes", "no": "no", "true": "yes", "false": "no", True: "yes", False: "no"}


def strip_thinking(text: str) -> str:
    return text.rsplit("</think>", 1)[-1] if "</think>" in text else text


def extract_json(text: str) -> dict | None:
    """The first complete JSON object in the text (after any thinking block or code fence), or None."""
    text = re.sub(r"```(?:json)?", "", strip_thinking(text or ""))
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def _banned(state) -> bool:
    if isinstance(state, dict):
        return any(k in BANNED_KEYS or _banned(v) for k, v in state.items())
    if isinstance(state, list):
        return any(_banned(v) for v in state)
    return isinstance(state, str) and bool(BANNED_IN_TEXT.search(state))


def _words(state) -> int:
    return len(state.split()) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False)) // 6


def normalise(raw: dict, brief: dict, author: str) -> tuple[dict | None, str]:
    """(item, "") when the authored object follows every rule, else (None, reason)."""
    state, q = raw.get("state"), raw.get("question") or {}
    qtype, crit, exp = q.get("type"), q.get("criteria"), raw.get("expected")
    if not isinstance(state, (str, dict)) or not q.get("instructions"):
        return None, "state or instructions missing"
    if qtype != brief["type"]:
        return None, f"type {qtype!r} is not the brief's {brief['type']!r}"
    if _banned(state):
        return None, "banned key in state"
    words = _words(state)
    if not 0.35 * brief["words"] <= words <= 3.0 * brief["words"]:
        return None, f"state length {words} words, wanted about {brief['words']}"
    surface, gold_probs = raw.get("surface_answer"), raw.get("gold_probs")
    if qtype == "choice":
        if not isinstance(crit, dict) or not 3 <= len(crit) <= 7 or not all(SNAKE.match(str(k)) for k in crit):
            return None, "choice criteria: need 3-7 snake_case options"
        if exp not in crit:
            return None, "expected is not an option"
        if surface is not None and (surface not in crit or surface == exp):
            return None, "surface_answer is not a wrong option"
        labels = list(crit)
    elif qtype == "noul":
        exp = NOUL.get(exp.lower() if isinstance(exp, str) else exp)
        if exp is None or not isinstance(crit, dict) or set(crit) != {"true", "false"}:
            return None, "noul: expected must be yes/no and criteria {true, false}"
        surface = NOUL.get(surface.lower() if isinstance(surface, str) else surface) if surface is not None else None
        if surface == exp:
            surface = None
        labels = ["no", "yes"]
    elif qtype == "score":
        if not isinstance(crit, list) or not 3 <= len(crit) <= 6 or not all(isinstance(c, str) and c for c in crit):
            return None, "score criteria: need 3-6 level descriptions"
        try:
            exp = int(exp)
        except (TypeError, ValueError):
            return None, "score: expected must be a level index"
        if not 0 <= exp < len(crit):
            return None, "score: expected out of range"
        try:
            surface = int(surface) if surface is not None else None
        except (TypeError, ValueError):
            surface = None
        if surface is not None and not (0 <= surface < len(crit) and surface != exp):
            surface = None
        labels, exp = [str(i) for i in range(len(crit))], str(exp)
    else:
        return None, f"unknown type {qtype!r}"
    if brief["family"] == "probability":
        if not isinstance(gold_probs, dict) or set(gold_probs) != set(labels):
            return None, "probability: gold_probs must cover every option"
        try:
            probs = {k: float(v) for k, v in gold_probs.items()}
        except (TypeError, ValueError):
            return None, "probability: gold_probs not numeric"
        top = max(probs, key=probs.get)
        if not 0.98 <= sum(probs.values()) <= 1.02 or top != exp or not 0.5 <= probs[top] <= 0.9:
            return None, "probability: gold_probs must sum to 1, peak at expected, peak between 0.5 and 0.9"
        gold_probs = probs
    else:
        gold_probs = None
    rationale = str(raw.get("rationale") or "")
    if len(rationale) < 40:
        return None, "rationale missing"
    item = {"family": brief["family"], "tier": brief["tier"], "state": state,
            "question": {"type": qtype, "instructions": q["instructions"], "criteria": crit},
            "labels": labels, "expected": exp, "group": None, "split": "replica", "language": "en",
            "brief": brief["id"],
            "provenance": {"source": "dopest-replica of the JevBench v1.5 sealed layout", "license": "Apache-2.0",
                           "author_model": author, "rationale": rationale, "surface_answer": surface,
                           "why_hard": str(raw.get("why_hard") or ""), "gold_probs": gold_probs,
                           "approx_state_words": words}}
    item["id"] = "rp-" + hashlib.sha1(f"{state_key(state)}\x1f{q['instructions']}\x1f{qtype}".encode()).hexdigest()[:12]
    return shuffled_options(item), ""


def parse_blind(raw: dict | None, item: dict) -> dict:
    """The reviewer's blind answer as {answer, probs, note, ok}; answer is one of the item's labels or None."""
    if not raw:
        return {"answer": None, "probs": None, "note": "", "ok": False}
    ans = raw.get("answer")
    if item["question"]["type"] == "noul":
        ans = NOUL.get(ans.lower() if isinstance(ans, str) else ans)
    elif ans is not None:
        ans = str(ans)
    probs = raw.get("probs") if isinstance(raw.get("probs"), dict) else None
    ok = ans in item["labels"]
    return {"answer": ans if ok else None, "probs": probs, "note": str(raw.get("note") or "")[:300], "ok": ok}


REASON_CLASSES = ("gold_wrong", "outside_knowledge", "leak", "format", "other", "none")


def parse_verdict(raw: dict | None) -> dict:
    v = str((raw or {}).get("verdict") or "").strip().lower()
    ok = v in ("accept", "reject")
    cls = str((raw or {}).get("reason_class") or "").strip().lower().replace(" ", "_")
    return {"verdict": v if ok else None, "reason_class": cls if cls in REASON_CLASSES else None,
            "reason": str((raw or {}).get("reason") or "")[:500], "fix": (raw or {}).get("fix"), "ok": ok}
