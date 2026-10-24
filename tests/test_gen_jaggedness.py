"""Every jaggedness label recomputed from the rendered text by plain code, plus twin and format checks."""
import re
from datetime import date
from decimal import Decimal

import pytest

from dopest_model.gen import jaggedness as jg
from tests.test_gen_rendered_labels import pd

WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def from_text(it):
    s, q, sub = it["state"], it["question"], it["provenance"]["rationale"].split("]")[0].split(":")[1]
    if sub == "invoice_sum":
        total = sum(int(q_) * int(d) * 100 + int(q_) * int(c) for q_, d, c in
                    re.findall(r": (\d+) x \$([\d,]+)\.(\d{2})", s.replace(",", "")))
        stated = re.search(r"Stated total: \$([\d,]+)\.(\d{2})", s)
        if q["type"] == "choice":
            return f"usd_{total // 100}_{total % 100:02d}"
        st = int(stated.group(1).replace(",", "")) * 100 + int(stated.group(2))
        if q["type"] == "noul":
            return "yes" if st == total else "no"
        gap = abs(st - total)
        return 0 if gap == 0 else 1 if gap < 100 else 2 if gap <= 5000 else 3
    if sub == "weekday":
        ref, ref_day, target = re.match(r"(.+?) is a (\w+)\. .+? is booked for (.+)\.$", s).groups()
        shift = (pd(target) - pd(ref)).days
        return WEEK[(WEEK.index(ref_day.lower()) + shift) % 7]
    if sub == "letter_count":
        word = s.split(": ", 1)[1]
        letter = re.search(r"letter '(\w)'", q["instructions"]).group(1)
        c = word.lower().count(letter)
        if q["type"] == "noul":
            m = re.search(r"exactly (once|\d+ times)", q["instructions"]).group(1)
            k = 1 if m == "once" else int(m.split()[0])
            return "yes" if c == k else "no"
        return min(4, c)
    if sub == "palindrome":
        t = s.split(": ", 1)[1]
        return "yes" if t == t[::-1] else "no"
    if sub == "digit_present":
        return "yes" if re.search(r"\d", s) else "no"
    if sub == "list_category":
        words = s.split(": ", 1)[1].split(", ")
        cat = next(c for c in jg.CATEGORIES if c in q["instructions"])
        c = sum(w in jg.CATEGORIES[cat] for w in words)
        if q["type"] == "noul":
            return "yes" if c >= int(re.search(r"at least (\d+)", q["instructions"]).group(1)) else "no"
        return 0 if c == 0 else 1 if c <= 2 else 2 if c <= 4 else 3
    if sub == "decimal_extreme":
        vals = {nm: Decimal(v) for nm, v in re.findall(r"^(\w+): ([\d.]+)$", s, re.M)}
        pick = max if "highest" in q["instructions"] else min
        return pick(vals, key=vals.get)
    if sub == "unit_compare":
        lb, kg = re.search(r"crate_c: ([\d.]+) lb; crate_b: ([\d.]+) kg", s).groups()
        return "yes" if Decimal(lb) * Decimal("0.4536") > Decimal(kg) else "no"
    if sub == "return_window":
        d, r_, n = re.search(r"delivered on (.+?)\. The customer asked to return it on (.+?)\. .*?no more than (\d+) days",
                             s).groups()
        late = (pd(r_) - pd(d)).days - int(n)
        return 0 if late <= 0 else 1 if late <= 2 else 2 if late <= 7 else 3
    raise KeyError(sub)


@pytest.fixture(scope="module")
def groups():
    return [jg.generate_case(seed) for seed in range(900)]


def test_text_reproduces_every_label(groups):
    seen = set()
    for items in groups:
        for it in items:
            assert from_text(it) == it["expected"], (it["provenance"]["rationale"], it["state"])
            seen.add(it["provenance"]["rationale"].split("]")[0].split(":")[1])
    assert seen == {t.name for t in jg.TASKS}


def test_twins_flip_and_items_are_well_formed(groups):
    flipped = 0
    for items in groups:
        assert items
        for it in items:
            if it["question"]["type"] == "choice":
                assert it["expected"] in it["labels"] and len(set(it["labels"])) == len(it["labels"]) >= 3
        twins = [it for it in items if it["provenance"]["rationale"].startswith("[twin")]
        if twins:
            assert twins[0]["expected"] != items[0]["expected"]
            flipped += 1
    assert flipped > 0.85 * len(groups)


def test_decimal_trap_and_weekday_engine():
    w = jg.Decimals((("glade", Decimal("3.11")), ("atlas", Decimal("3.9")), ("beacon", Decimal("3.19"))), True)
    assert jg.DecimalExtreme().solve(w) == "atlas"
    assert jg.WeekdayOf().solve(jg.Weekday(date(2028, 1, 5), date(2028, 1, 22))) == "saturday"
