"""Judging a model's answer against an explicit request (JevBench hard tier, family judge_hard).

Shape taken from the public items, wording our own: the state is {"request": ..., "response": ...}; the request
sets a small quantitative task with explicit requirements (an exact factor, decimal places, a unit, the order of
steps, a figure that must be carried through); the response is a short worked answer. Question: does the response
fully and correctly satisfy the request? The answer is "no" when the response carries exactly one planted flaw:
  arithmetic   one intermediate or final number is off
  rounding     the final value is truncated, or given to the wrong number of places
  requirement  an explicit requirement is ignored (method step missing, steps out of order, unit dropped)
  retained     a figure given in the request is silently changed in the response
A correct and a flawed version of one request form a twin; both come from the same engine.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

from .common import SPLIT_BY_STYLE, Nar, noul_item, stable_id

FAMILY = "judge_hard"
GENERATOR = "judge_response_v1"
FLAWS = ("arithmetic", "rounding", "requirement", "retained")


def q(x: Decimal, places: int, mode=ROUND_HALF_UP) -> Decimal:
    return x.quantize(Decimal(1).scaleb(-places), mode)


def fmt(x: Decimal, places: int) -> str:
    return f"{q(x, places):,.{places}f}"


@dataclass(frozen=True)
class Task:
    request: str
    steps: list            # correct worked steps (strings), in the required order
    final: Decimal
    places: int
    unit: str
    numbers: list          # Decimals that appear in the steps (for the arithmetic flaw)
    retained: tuple        # (figure in the request, wrong replacement) for the retained-figure flaw
    method_step: str | None   # a step that shows a required method, dropped by the requirement flaw


def cistern(rng: random.Random) -> Task:
    start = Decimal(rng.randint(40, 300))
    rate = Decimal(rng.randint(8, 45)) / 10
    hours = Decimal(rng.choice(["2.5", "3", "4.5", "6", "7.5", "8", "10.5"]))
    added = Decimal(rng.randint(5, 40))
    lost = rate * hours
    after = start - lost
    final = after + added
    req = (f"A storage cistern holds {start} litres at the start. Over {hours} hours it loses {rate} litres per hour "
           f"through a leak; afterwards {added} litres are pumped in. State the volume at the end in litres, rounded to "
           f"one decimal, and lay out the arithmetic in the order the events happened.")
    steps = [f"Lost to the leak: {rate} × {hours} = {fmt(lost, 1)} litres.",
             f"Left after the leak: {start} − {fmt(lost, 1)} = {fmt(after, 1)} litres.",
             f"Pumping in {added} litres: {fmt(after, 1)} + {added} = {fmt(final, 1)} litres."]
    return Task(req, steps, final, 1, "litres", [lost, after], (str(added), str(added + 5)), None)


def floor_area(rng: random.Random) -> Task:
    sqft = Decimal(rng.randint(120, 2400))
    f = Decimal("0.3048")
    final = sqft * f * f
    req = (f"A floor measures {sqft} square feet. Express it in square metres with the exact factor 1 ft = 0.3048 m, "
           f"make clear that the factor is squared for an area, and round the result to three decimals.")
    steps = [f"Area needs the factor twice: {sqft} ft² × (0.3048 m/ft)² = {sqft} × {f * f} m² = {final} m².",
             f"Rounded to three decimals: {fmt(final, 3)} m²."]
    return Task(req, steps, final, 3, "m²", [final], (str(sqft), str(sqft + 10)), steps[0])


def price_chain(rng: random.Random) -> Task:
    price = Decimal(rng.randint(30, 900)) + Decimal(rng.choice([0, 50, 99, 25, 75])) / 100
    d1, d2 = rng.choice([10, 15, 20, 25]), rng.choice([5, 10, 12])
    tax = rng.choice([5, 8, 20, 23])
    p1 = price * (100 - d1) / 100
    p2 = p1 * (100 - d2) / 100
    final = p2 * (100 + tax) / 100
    req = (f"List price {fmt(price, 2)} EUR. Take {d1}% off, then another {d2}% off the already reduced amount, then "
           f"put {tax}% tax on top. Round only the final figure, to two decimals, and present the three stages one "
           f"after another.")
    steps = [f"Stage 1, {d1}% off: {fmt(price, 2)} × {Decimal(100 - d1) / 100} = {p1} EUR.",
             f"Stage 2, {d2}% off that: {p1} × {Decimal(100 - d2) / 100} = {p2} EUR.",
             f"Stage 3, {tax}% tax: {p2} × {Decimal(100 + tax) / 100} = {final} EUR, which is {fmt(final, 2)} EUR."]
    return Task(req, steps, final, 2, "EUR", [p1, p2], (f"{d2}% off", f"{d2 + 5}% off"), None)


def train_speed(rng: random.Random) -> Task:
    km = Decimal(rng.randint(40, 420))
    h = Decimal(rng.randint(1, 5))
    m = Decimal(rng.choice([0, 15, 20, 30, 40, 45, 50]))
    hours = h + m / 60
    final = km / hours
    req = (f"A train travels {km} km in {h} h {m} min. Express the time in hours as a decimal first, then work out "
           f"the average speed in km/h to one decimal.")
    steps = [f"Time as a decimal: {h} h {m} min = {h} + {m}/60 = {fmt(hours, 4)} h.",
             f"Average speed: {km} / {fmt(hours, 4)} = {fmt(final, 1)} km/h."]
    return Task(req, steps, final, 1, "km/h", [hours], (str(km), str(km + 10)), steps[0])


def fahrenheit(rng: random.Random) -> Task:
    c = Decimal(rng.randint(-25, 45)) + Decimal(rng.choice([0, "0.5"]))
    scaled = c * 9 / 5
    final = scaled + 32
    req = (f"Express {c} °C in degrees Fahrenheit with F = C × 9/5 + 32: do the multiplication step first, then the "
           f"addition, and give one decimal with the unit.")
    steps = [f"Multiplication: {c} × 9/5 = {fmt(scaled, 1)}.", f"Addition: {fmt(scaled, 1)} + 32 = {fmt(final, 1)} °F."]
    return Task(req, steps, final, 1, "°F", [scaled], (f"Express {c} °C", f"Express {c + 1} °C"), steps[0])


def sales_change(rng: random.Random) -> Task:
    old = Decimal(rng.randint(120, 9000))
    new = q(old + Decimal(rng.randint(-60, 90)) * old / 1000, 0)
    final = (new - old) / old * 100
    req = (f"Unit sales moved from {old} to {new}. Compute the percentage change relative to the earlier figure, to "
           f"two decimals, including the sign.")
    steps = [f"Difference: {new} − {old} = {new - old} units.",
             f"Against the earlier figure: {new - old} / {old} × 100 = {fmt(final, 2)}%."]
    return Task(req, steps, final, 2, "%", [new - old], (f"to {new}", f"to {new + 7}"), None)


TASKS = [cistern, floor_area, price_chain, train_speed, fahrenheit, sales_change]


def _replace_whole(text: str, old: str, new: str) -> str | None:
    pattern = r"(?<![\d.])" + re.escape(old) + r"(?![\d.])"
    return re.sub(pattern, new, text, count=1) if re.search(pattern, text) else None


def render(task: Task, flaw: str | None, rng: random.Random) -> tuple[str, str]:
    """The response text and a short description of the flaw (empty for a correct response)."""
    steps = list(task.steps)
    if flaw is None:
        return " ".join(steps), ""
    if flaw == "arithmetic":
        x = rng.choice(task.numbers)
        for token in (fmt(x, task.places), fmt(x, 1), fmt(x, 4), str(x)):
            shown = next((s for s in steps if _replace_whole(s, token, token) is not None), None)
            if shown is not None:
                places = len(token.split(".")[-1]) if "." in token else 0
                unit = Decimal(1).scaleb(-places) * rng.choice([1, 10])
                wrong = fmt(Decimal(token.replace(",", "")) + rng.choice([-1, 1]) * unit, places)
                steps[steps.index(shown)] = _replace_whole(shown, token, wrong)
                return " ".join(steps), f"arithmetic: {token} shown as {wrong}"
        flaw = "rounding"
    if flaw == "rounding":
        right = fmt(task.final, task.places)
        truncated = f"{q(task.final, task.places, ROUND_DOWN):,.{task.places}f}"
        wrong, why = ((fmt(task.final, task.places + 1), f"rounding: {task.places} places asked, {task.places + 1} given")
                      if truncated == right else (truncated, f"rounding: truncated {truncated} instead of {right}"))
        steps[-1] = steps[-1].replace(right, wrong)
        return " ".join(steps), why
    if flaw == "requirement":
        if task.method_step and len(steps) > 1:
            steps.remove(task.method_step)
            return " ".join(steps), "requirement: the requested method step is not shown"
        if len(steps) > 1:
            return " ".join(steps[::-1]), "requirement: stages not in the required order"
        return " ".join(steps).replace(task.unit, "").strip(), "requirement: unit omitted"
    old, new = task.retained
    num_old = re.search(r"-?\d+(?:\.\d+)?", old).group(0)        # the figure inside the request phrase
    num_new = re.search(r"-?\d+(?:\.\d+)?", new).group(0)
    text = _replace_whole(" ".join(steps), num_old, num_new)
    if text is not None:
        return text, f"retained: {num_old} from the request appears as {num_new}"
    return render(task, "rounding", rng)


def generate_case(seed: int) -> list[dict]:
    rng = random.Random(seed)
    maker = TASKS[seed % len(TASKS)]
    task = maker(rng)
    n = Nar(seed=seed, style=rng.choice(list(SPLIT_BY_STYLE)), ds="iso")
    group = f"dg-{FAMILY}-{stable_id(GENERATOR, str(seed))}"
    base = {"family": FAMILY, "generator": GENERATOR, "use_case": "llm_guardrails", "topic": "math", "group": group,
            "style": n.style}
    flaw = rng.choice(FLAWS)
    correct = " ".join(task.steps)
    items = []
    for label, fl in (("base", None), ("twin", flaw)):
        response, why = render(task, fl, rng)
        if label == "twin" and response == correct:
            continue
        items.append(noul_item(
            instructions="Does the response do everything the request asks, correctly?",
            true_text="The response meets every explicit requirement of the request (method, order, units, rounding, "
                      "figures carried through) with no substantive error.",
            false_text="The response contains a substantive error or skips an explicit requirement.",
            answer=fl is None, state=json.dumps({"request": task.request, "response": response}, ensure_ascii=False),
            rationale=f"[{label}:{maker.__name__}] {why or 'correct: ' + fmt(task.final, task.places) + ' ' + task.unit}",
            **base))
    return items
