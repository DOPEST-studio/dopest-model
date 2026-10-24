"""judge_response: the correct response's final number is recomputed from the REQUEST text with a separate
implementation; a flawed twin must differ from the correct response and be labelled 'no'."""
import json
import re
from decimal import ROUND_HALF_UP, Decimal

import pytest

from dopest_model.gen import judge_response as jr


def final_from_request(req: str) -> tuple[Decimal, int]:
    n = lambda s: Decimal(s.replace(",", ""))
    if m := re.match(r"A storage cistern holds ([\d.]+) litres at the start. Over ([\d.]+) hours it loses ([\d.]+) litres per hour through a leak; afterwards ([\d.]+) litres", req):
        s, h, r, a = map(n, m.groups())
        return s - r * h + a, 1
    if m := re.match(r"A floor measures ([\d.]+) square feet", req):
        return n(m.group(1)) * Decimal("0.3048") ** 2, 3
    if m := re.match(r"List price ([\d.,]+) EUR. Take (\d+)% off, then another (\d+)% off the already reduced amount, then put (\d+)% tax", req):
        p, d1, d2, t = map(n, m.groups())
        return p * (100 - d1) / 100 * (100 - d2) / 100 * (100 + t) / 100, 2
    if m := re.match(r"A train travels ([\d.]+) km in (\d+) h (\d+) min", req):
        km, h, mi = map(n, m.groups())
        return km / (h + mi / 60), 1
    if m := re.match(r"Express (-?[\d.]+) °C in degrees Fahrenheit", req):
        return n(m.group(1)) * 9 / 5 + 32, 1
    if m := re.match(r"Unit sales moved from ([\d.]+) to ([\d.]+)\.", req):
        o, w = map(n, m.groups())
        return (w - o) / o * 100, 2
    raise AssertionError(req)


def cases(n: int = 240):
    return [jr.generate_case(seed) for seed in range(n)]


@pytest.mark.parametrize("items", cases(), ids=lambda its: its[0]["group"] if its else "empty")
def test_correct_response_holds_the_recomputed_final_and_twin_differs(items):
    assert items and items[0]["expected"] == "yes"
    st = json.loads(items[0]["state"])
    final, places = final_from_request(st["request"])
    shown = f"{final.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP):,.{places}f}"
    assert shown in st["response"], (shown, st["response"])
    if len(items) > 1:
        twin = json.loads(items[1]["state"])
        assert items[1]["expected"] == "no" and twin["request"] == st["request"] and twin["response"] != st["response"]


def test_yes_no_balance_and_flaw_mix():
    import collections
    items = [it for its in cases(300) for it in its]
    labels = collections.Counter(it["expected"] for it in items)
    assert 0.4 < labels["yes"] / len(items) < 0.6
    flaws = collections.Counter(it["provenance"]["rationale"].split("] ")[1].split(":")[0] for it in items if it["expected"] == "no")
    assert set(flaws) >= {"arithmetic", "rounding", "requirement", "retained"} and min(flaws.values()) >= 20
