"""Shared helpers for code-labelled decision generators.

Items follow the OpenDecisions format, so the evaluation pipeline reads them unchanged. Labels come from a rule
engine, never from a language model; generators keep a case only when exactly one answer is defensible.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# Narrative styles of the case file are the unit of the train/dev/calib/test split, so test items come from
# renderings the model never saw in training.
SPLIT_BY_STYLE = {"memo": "train", "record": "train", "email": "train", "timeline": "calib", "transcript": "test"}

DATE_STYLES = ("dmy", "mdy", "iso", "dmy_short")


def fmt_date(d: date, style: str) -> str:
    if style == "dmy":
        return f"{d.day} {d:%B} {d.year}"
    if style == "mdy":
        return f"{d:%B} {d.day}, {d.year}"
    if style == "iso":
        return d.isoformat()
    return f"{d.day} {d:%b} {d.year}"


@dataclass(frozen=True)
class Nar:
    """Narrative choices drawn once per case (names, ids, wording), so a twin differs in one fact only."""
    seed: int
    style: str
    ds: str

    def rng(self, salt: int) -> random.Random:
        return random.Random(self.seed * 1009 + salt)

    def d(self, x: date) -> str:
        return fmt_date(x, self.ds)

    def dt(self, local: datetime, offset: int) -> str:
        return f"{local:%a} {self.d(local.date())} {local:%H:%M} (UTC{offset:+d})"


CURRENCIES = {"USD": "$", "EUR": "EUR ", "GBP": "£"}


def fmt_money(amount: int, currency: str) -> str:
    return f"{CURRENCIES[currency]}{amount:,}"


def days_between(a: date, b: date) -> int:
    return (b - a).days


def add_days(d: date, n: int) -> date:
    return d + timedelta(days=n)


def stable_id(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:10]


def render_file(style: str, title: str, header: str, facts: list[tuple[date | None, str]]) -> str:
    """A case file in one of the split styles; facts are (date or None, sentence)."""
    if style == "memo":
        return f"{title}\n{header}\n\n" + " ".join(text for _, text in facts)
    if style == "record":
        return f"{title}\n{header}\n" + "\n".join(f"- {text}" for _, text in facts)
    if style == "email":
        half = max(1, len(facts) // 2)
        first = " ".join(text for _, text in facts[:half])
        second = " ".join(text for _, text in facts[half:])
        return (f"From: case handler\nTo: review desk\nSubject: {title}\n\n{header}\n\n{first}\n\n"
                f"--- follow-up from the case handler ---\n{second}")
    if style == "timeline":
        dated = sorted((f for f in facts if f[0]), key=lambda f: f[0])
        undated = [f for f in facts if not f[0]]
        rows = [f"{d.isoformat()}: {text}" for d, text in dated] + [f"Note: {text}" for _, text in undated]
        return f"{title} (TIMELINE)\n{header}\n" + "\n".join(rows)
    return f"{title}: CALL NOTES\n{header}\n" + "\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(facts, 1))


def choice_item(*, rng: random.Random, options: dict[str, str], expected: str, trap: str | None, **common) -> dict:
    """A choice item with options in random order, so the gold position carries no signal."""
    if expected not in options:
        raise ValueError(f"gold {expected!r} is not an option")
    pairs = list(options.items())
    rng.shuffle(pairs)
    criteria = dict(pairs)
    return _item(qtype="choice", criteria=criteria, labels=list(criteria), expected=expected, trap=trap, **common)


def noul_item(*, true_text: str, false_text: str, answer: bool, **common) -> dict:
    return _item(qtype="noul", criteria={"true": true_text, "false": false_text}, labels=["no", "yes"],
                 expected="yes" if answer else "no", trap=None, **common)


def score_item(*, levels: list[str], expected: int, **common) -> dict:
    if not 0 <= expected < len(levels):
        raise ValueError(f"level {expected} out of range")
    return _item(qtype="score", criteria=levels, labels=[str(i) for i in range(len(levels))], expected=expected,
                 trap=None, **common)


def _item(*, qtype: str, criteria, labels: list[str], expected, trap, family: str, generator: str, state: str,
          instructions: str, group: str, style: str, rationale: str, use_case: str, topic: str,
          tier: str = "hard", language: str = "en", target_probs: dict | None = None) -> dict:
    return {
        "id": f"dg-{family}-{stable_id(generator, state, instructions, qtype)}",
        "family": family,
        "state": state,
        "question": {"type": qtype, "instructions": instructions, "criteria": criteria},
        "labels": labels,
        "expected": expected,
        "trap_answer": trap,
        "group": group,
        "provenance": {"source": f"dopest-gen/{generator}", "license": "Apache-2.0", "rationale": rationale,
                       "style": style},
        "tier": tier,
        "language": language,
        "use_case": use_case,
        "topic": topic,
        "label_source": "programmatic",
        "target_probs": target_probs,
        "split": SPLIT_BY_STYLE[style],
    }
