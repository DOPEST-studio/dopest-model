"""Judging a worked answer on ordered quality levels: an invoice request with explicit requirements and a response
with zero, one or two deliberate defects, rated by a rubric that separates a wrong main figure from presentation
misses.

The request: line items, a voucher deducted before tax, a tax rate, and four requirements (state the subtotal,
deduct the voucher before tax, round the total to the cent, state the currency code). The response is generated from
the same numbers with a chosen defect set: a skipped line item (main figure far off), a small arithmetic slip, the
voucher deducted after tax (a small order error), a missing subtotal line, a missing currency code, a total rounded
to whole units. The engine computes the deviation of the response's figure before its rounding and counts the
presentation misses; labels:
  score    the quality level 0-4 under the rubric in the question
  noul     whether the response's final total is correct to the cent
  choice   the defect present (single-defect and defect-free responses only)
Twins change the defect set so that the level moves.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

from .common import SPLIT_BY_STYLE, Nar, choice_item, noul_item, score_item, stable_id

FAMILY = "judge_quality"
GENERATOR = "judge_quality_v1"
LEVELS = ["0: the final total is off by more than 5%, or a line item is ignored",
          "1: the method is recognisable but the final total is wrong by up to 5%",
          "2: the final total is right, but two or more of the presentation requirements are missed",
          "3: the final total is right and exactly one presentation requirement is missed",
          "4: the final total is right and every requirement is met"]
DEFECTS = {"skipped_item": "A line item is left out of the calculation.",
           "arithmetic_slip": "The method is right but one figure is miscalculated by a small amount.",
           "voucher_after_tax": "The voucher is deducted after tax instead of before.",
           "missing_subtotal": "The subtotal before the voucher is not stated.",
           "missing_currency": "The currency code is not stated.",
           "whole_unit_rounding": "The final total is rounded to whole units instead of the cent.",
           "none": "The response has no defect."}
PRESENTATION = ("missing_subtotal", "missing_currency", "whole_unit_rounding")
PRODUCTS = ["ergonomic chair", "standing desk", "monitor arm", "webcam", "desk lamp", "cable tray", "whiteboard",
            "filing cabinet", "noise-cancelling headset", "docking station", "label printer", "paper shredder"]


@dataclass(frozen=True)
class Invoice:
    lines: tuple              # (name, qty, unit_cents)
    voucher: int              # cents, deducted before tax
    tax_pct: int
    currency: str
    defects: tuple            # subset of DEFECTS without "none"
    slip: int                 # cents added by an arithmetic slip (used only when the defect is present)


def subtotal(inv: Invoice) -> int:
    return sum(q * p for _, q, p in inv.lines)


def total(inv: Invoice) -> int:
    """Correct final total in cents: (subtotal - voucher) * (1 + tax), rounded to the cent."""
    return round((subtotal(inv) - inv.voucher) * (100 + inv.tax_pct) / 100)


def response_figures(inv: Invoice) -> tuple[int, int, int, str]:
    """(figure before the response's own rounding, figure as stated, subtotal used, arithmetic shown)."""
    lines = inv.lines[1:] if "skipped_item" in inv.defects else inv.lines
    sub = sum(q * p for _, q, p in lines)
    if "voucher_after_tax" in inv.defects:
        fig = round(sub * (100 + inv.tax_pct) / 100) - inv.voucher
        shown = (f"{sub / 100:.2f} x {1 + inv.tax_pct / 100:.2f} = {sub * (100 + inv.tax_pct) / 10000:.2f}, minus the "
                 f"voucher {inv.voucher / 100:.2f}")
    else:
        fig = round((sub - inv.voucher) * (100 + inv.tax_pct) / 100)
        shown = f"({sub / 100:.2f} - {inv.voucher / 100:.2f}) x {1 + inv.tax_pct / 100:.2f}"
    if "arithmetic_slip" in inv.defects:
        fig += inv.slip
    stated = round(fig / 100) * 100 if "whole_unit_rounding" in inv.defects else fig
    return fig, stated, sub, shown


def level(inv: Invoice) -> int:
    fig, _, _, _ = response_figures(inv)
    correct = total(inv)
    if "skipped_item" in inv.defects or abs(fig - correct) > 0.05 * correct:
        return 0
    if fig != correct:
        return 1
    missed = sum(d in inv.defects for d in PRESENTATION)
    return 4 if missed == 0 else 3 if missed == 1 else 2


def sample(rng: random.Random) -> Invoice:
    while True:
        names = rng.sample(PRODUCTS, rng.randint(3, 5))
        lines = tuple((nm, rng.randint(1, 6), rng.randint(1500, 42000)) for nm in names)
        voucher = rng.choice([2500, 4000, 5000, 7500, 10000])
        if sum(q * p for _, q, p in lines) - voucher >= 30000:
            break
    pick = rng.choice(["none", "none", "skipped_item", "arithmetic_slip", "voucher_after_tax", "missing_subtotal",
                       "missing_currency", "whole_unit_rounding", "two"])
    defects = () if pick == "none" else tuple(rng.sample(PRESENTATION, 2)) if pick == "two" else (pick,)
    return Invoice(lines=lines, voucher=voucher, tax_pct=rng.choice([5, 8, 19, 20, 23]),
                   currency=rng.choice(["EUR", "USD", "GBP"]), defects=defects,
                   slip=rng.choice([-1, 1]) * rng.choice([90, 150, 300, 420]))


def mutate(rng: random.Random, inv: Invoice) -> Invoice:
    pool = [d for d in DEFECTS if d != "none"]
    if inv.defects and rng.random() < 0.5:
        return replace(inv, defects=tuple(inv.defects[1:]))
    extra = rng.choice([d for d in pool if d not in inv.defects])
    return replace(inv, defects=tuple(inv.defects) + (extra,))


def render(inv: Invoice, n: Nar) -> str:
    r = n.rng(47)
    client = r.choice(["Marlowe Design Studio", "Kettle & Co", "Northlight Architects", "Juniper Labs", "Saltmarsh Books"])
    listed = "\n".join(f"- {q} x {nm} at {p / 100:.2f} each" for nm, q, p in inv.lines)
    request = (f"Prepare the invoice total for {client}. Line items:\n{listed}\n"
               f"A voucher of {inv.voucher / 100:.2f} is deducted from the subtotal before tax. Tax is {inv.tax_pct}% on the "
               f"amount after the voucher. Requirements: (1) state the subtotal before the voucher; (2) deduct the voucher "
               f"before applying tax; (3) give the final total rounded to the cent; (4) state the currency code {inv.currency}.")
    _, stated, sub, shown = response_figures(inv)
    cur = "" if "missing_currency" in inv.defects else f" {inv.currency}"
    used = inv.lines[1:] if "skipped_item" in inv.defects else inv.lines
    parts = ", ".join(f"{q} x {p / 100:.2f} = {q * p / 100:.2f}" for _, q, p in used)
    body = ([f"Subtotal before the voucher: {sub / 100:.2f}{cur} ({parts})."] if "missing_subtotal" not in inv.defects
            else [f"Line totals: {parts}."])
    body.append(f"Calculation: {shown}.")
    body.append(f"Final total due: {stated / 100:.0f}{cur}." if "whole_unit_rounding" in inv.defects
                else f"Final total due: {stated / 100:.2f}{cur}.")
    return f"REQUEST:\n{request}\n\nRESPONSE:\n{' '.join(body)}"


def items(rng: random.Random, inv: Invoice, n: Nar, base: dict) -> list[dict]:
    state = render(inv, n)
    lvl = level(inv)
    correct = total(inv)
    _, stated, _, _ = response_figures(inv)
    why = f"Correct total {correct / 100:.2f}, response states {stated / 100:.2f}; defects {list(inv.defects)} -> level {lvl}."
    common = dict(base, state=state, rationale=f"{base['rationale']} {why}", tier="judge")
    out = [score_item(instructions="Rate the response against the request on the rubric below.", levels=LEVELS,
                      expected=lvl, **common),
           noul_item(instructions="Is the final total stated in the response correct to the cent?",
                     true_text="The stated final total equals (subtotal - voucher) x (1 + tax), rounded to the cent.",
                     false_text="The stated final total differs from the correct amount.", answer=stated == correct,
                     **common)]
    if len(inv.defects) <= 1:
        expected = inv.defects[0] if inv.defects else "none"
        out.append(choice_item(rng=rng, options=dict(DEFECTS), expected=expected,
                               trap="arithmetic_slip" if expected == "voucher_after_tax" else None,
                               instructions="Which defect, if any, does the response have?", **common))
    return out


def generate_case(seed: int) -> list[dict]:
    rng = random.Random(seed)
    inv = sample(rng)
    n = Nar(seed=seed, style=rng.choice(list(SPLIT_BY_STYLE)), ds="iso")
    group = f"dg-{FAMILY}-{stable_id(GENERATOR, str(seed))}"
    base = {"family": FAMILY, "generator": GENERATOR, "use_case": "answer_judging", "topic": "finance_ops",
            "group": group, "style": n.style}
    out = items(rng, inv, n, dict(base, rationale="[base:invoice]"))
    for _ in range(10):
        m = mutate(rng, inv)
        if level(m) != level(inv):
            out += items(rng, m, n, dict(base, rationale="[twin:invoice]"))
            break
    return out
