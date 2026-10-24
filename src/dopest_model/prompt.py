"""Decision item -> lettered multiple-choice chat prompt.

Mirrors predict_letters.py from OpenDecisions (MIT, jgeuter/OpenDecisions), which follows JevBench's
semif_direct adapter, so zero-shot numbers stay comparable with the Jeff paper (arXiv 2610.17205).
"""
from __future__ import annotations

import json

SYSTEM = ("Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
          "Respond with only its uppercase letter, with no explanation or reasoning.")
LETTERS = "ABCDEFGHIJKLMNOP"


def options(question: dict, show_keys: bool = True) -> list[dict]:
    """Option ids and descriptions of one question, in prompt order (show_keys=False: TypeSafe's option_keys=hide)."""
    qtype, criteria = question["type"], question.get("criteria")
    if qtype == "noul":
        criteria = criteria or {}
        opts = [(key, criteria.get(key, f"The proposition is {key}.")) for key in ("true", "false")]
    elif qtype == "choice" and isinstance(criteria, dict):
        opts = [(key, text or key) for key, text in criteria.items()]
    elif qtype == "choice":
        opts = [(str(key), str(key)) for key in criteria]
    else:
        opts = [(str(i), text) for i, text in enumerate(criteria)]
    return [{"id": key, "description": f"{key}: {text}" if show_keys else str(text)} for key, text in opts]


def messages_for(state, instructions: str, opts: list[dict]) -> list[dict]:
    """Chat messages for a state, a criterion and an explicit option list (the serving path reads long choice
    lists in groups, so the options need not be the question's full list)."""
    if len(opts) > len(LETTERS):
        raise ValueError(f"more than {len(LETTERS)} options")
    payload = {"evidence": state, "criterion": instructions,
               "options": [{"letter": LETTERS[i], "description": o["description"]} for i, o in enumerate(opts)]}
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def messages(item: dict) -> tuple[list[dict], list[str]]:
    """Chat messages for one item and the option ids in letter order."""
    opts = options(item["question"])
    try:
        msgs = messages_for(item["state"], item["question"]["instructions"], opts)
    except ValueError:
        raise ValueError(f"{item['id']}: more than {len(LETTERS)} options")
    return msgs, [o["id"] for o in opts]


def probs_by_label(qtype: str, ids: list[str], probs: list[float]) -> dict[str, float]:
    """Map letter-order probabilities to the scorer's label space (yes/no for noul)."""
    by_id = dict(zip(ids, probs))
    if qtype == "noul":
        return {"yes": by_id["true"], "no": by_id["false"]}
    return by_id
