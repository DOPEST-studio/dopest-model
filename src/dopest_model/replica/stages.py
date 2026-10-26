"""The three model stages of the replica as pure functions over rows: conversations in, parsed rows out. The Modal
entrypoint (blockb/verify_modal.py::replica) runs the generations; assemble.py selects the final set."""
from __future__ import annotations

from .parse import extract_json, normalise, parse_blind, parse_verdict
from .prompts import author_prompt, blind_prompt, verdict_prompt


def author_convs(briefs: list[dict]) -> list[list[dict]]:
    return [author_prompt(b) for b in briefs]


def author_rows(briefs: list[dict], outputs: list[dict], author: str) -> list[dict]:
    """One row per brief: the item when it passed every check, else the reason; keyed by the brief id."""
    rows = []
    for b, o in zip(briefs, outputs):
        raw = extract_json(o.get("text", ""))
        item, why = normalise(raw, b, author) if raw else (None, "no JSON object in the output")
        rows.append({"id": b["id"], "ok": item is not None, "why": why, "item": item,
                     "tokens": o.get("tokens"), "finish": o.get("finish")})
    return rows


def blind_convs(items: list[dict]) -> list[list[dict]]:
    return [blind_prompt(it) for it in items]


def blind_rows(items: list[dict], outputs: list[dict], reviewer: str) -> list[dict]:
    rows = []
    for it, o in zip(items, outputs):
        row = {"id": it["id"], "reviewer": reviewer, **parse_blind(extract_json(o.get("text", "")), it),
               "tokens": o.get("tokens"), "finish": o.get("finish")}
        if not row["ok"]:
            row["raw"] = (o.get("text") or "")[:600]                      # what the model wrote, for diagnosis
        rows.append(row)
    return rows


def verdict_convs(items: list[dict], blind_by_id: dict[str, dict]) -> list[list[dict]]:
    return [verdict_prompt(it, blind_by_id.get(it["id"])) for it in items]


def verdict_rows(items: list[dict], outputs: list[dict], reviewer: str) -> list[dict]:
    rows = []
    for it, o in zip(items, outputs):
        row = {"id": it["id"], "reviewer": reviewer, **parse_verdict(extract_json(o.get("text", ""))),
               "tokens": o.get("tokens"), "finish": o.get("finish")}
        if not row["ok"]:
            row["raw"] = (o.get("text") or "")[:600]
        rows.append(row)
    return rows
