"""Exact engines against hand-computed cases (from OpenDecisions and JevBench rationales), item invariants, and an
independent recomputation of every target distribution from the rendered text."""
import math
import re
from datetime import date
from fractions import Fraction

import pytest

from dopest_model.gen import probability as pb


def test_bayes_base_rate():
    # 2% fraud, flags 95% of fraud and 1% of legitimate -> 0.019 / (0.019 + 0.0098) = 0.6597
    w = pb.BayesWorld(0, (("travel", Fraction(2, 100)),), Fraction(95, 100), Fraction(1, 100), "travel", "travel")
    assert abs(float(pb.Bayes.posterior(w)) - 0.6597222) < 1e-6


def test_sampling_without_replacement():
    # 12 units, 3 defective, draw 3: P(none) = 9/12 * 8/11 * 7/10 = 21/55
    w = pb.LotWorld(12, 3, 2, 3, date(2026, 9, 1), date(2026, 9, 18), False)
    assert pb.Sampling().solve(w)["yes"] == Fraction(34, 55)
    assert pb.Sampling().solve(pb.replace(w, inspection=date(2026, 8, 31)))["yes"] == 1 - Fraction(9 * 8, 12 * 11)
    lv = pb.Sampling().levels(pb.replace(w, units=13, defective=1, k_new=2))
    assert lv["0"] == Fraction(66, 78) and lv["2"] == 0 and sum(lv.values()) == 1


def test_connection_rescue_term():
    # P(make) = 34/50 + 6/50 x 10/40 = 0.71, so P(miss) = 0.29
    w = pb.ConnWorld(50, 34, 6, 10, 40, 60, 40, 10)
    assert pb.Connection().solve(w)["yes"] == Fraction(29, 100)


def test_urn_two_stage():
    w = pb.UrnWorld(2, (3, 1, 0), (0, 1, 3))
    d = pb.Urn.solve(w)
    assert d["gold"] == Fraction(1, 3) * Fraction(3, 4) and sum(d.values()) == 1


def test_reference_class_filters():
    rows = (("T-1", "Enterprise", "SSO login", date(2026, 7, 20), True, "escalated"),
            ("T-2", "Enterprise", "SSO login", date(2026, 7, 15), True, "reopened"),     # on the cutoff day: excluded
            ("T-3", "Enterprise", "SSO login", date(2026, 8, 1), False, None),           # open: excluded
            ("T-4", "Standard", "SSO login", date(2026, 8, 2), True, "resolved_first_contact"),
            ("T-5", "Enterprise", "SSO login", date(2026, 8, 3), True, "resolved_first_contact"),
            ("T-6", "Enterprise", "SSO login", date(2026, 8, 4), True, "escalated"))
    w = pb.RefWorld(0, rows, date(2026, 7, 15), "Enterprise", "SSO login", date(2026, 9, 1), "Standard")
    assert pb.ReferenceClass().solve(w) == {"resolved_first_contact": Fraction(1, 3), "escalated": Fraction(2, 3),
                                            "reopened": 0}


# ------------------------------------------------------------------ generated items

@pytest.fixture(scope="module")
def groups():
    return [pb.generate_case(seed) for seed in range(500)]


def test_items_carry_exact_targets_and_twins_flip(groups):
    flipped = 0
    for items in groups:
        assert items
        for it in items:
            tp = it["target_probs"]
            assert abs(sum(tp.values()) - 1) < 1e-9 and set(tp) == set(it["labels"])
            gold = max(tp, key=tp.get)
            assert str(it["expected"]) == gold
            ordered = sorted(tp.values(), reverse=True)
            assert ordered[0] - ordered[1] >= 0.06 - 1e-12
        tags = [it["provenance"]["rationale"].split("]")[0] for it in items]
        if any(t.startswith("[twin") for t in tags):
            assert items[0]["expected"] != items[-1]["expected"]
            flipped += 1
    assert flipped > 0.9 * len(groups)


# ------------------------------------------------------------------ independent recomputation from the text

def num(s):
    return Fraction(s.replace(",", ""))


def from_text(it):
    s, sub = it["state"], it["provenance"]["rationale"].split("]")[0].split(":")[1]
    if sub == "bayes":
        m = re.search(r"\((?:merchant category|production line|user group|inbox): ([^)]+)\) was", s)
        seg = m.group(1)
        base = num(re.search(rf"{re.escape(seg)}: ([\d.]+)%", s).group(1)) / 100
        hit, fa = (num(x) / 100 for x in re.search(r"flags ([\d.]+)% of .*? and ([\d.]+)% of", s).groups())
        p = base * hit / (base * hit + (1 - base) * fa)
        return {"yes": p, "no": 1 - p}
    if sub == "sampling":
        k_old, = re.search(r"version 2: draw (\d+) unit", s).groups()
        eff, k_new = re.search(r"on or after (.+?): draw (\d+) units", s).groups()
        units, d = re.search(r"has (\d+) units\. The supplier's verified test found exactly (\d+)", s).groups()
        insp = re.search(r"scheduled for (.+?)\.(?:\s|$)", s).group(1)
        from tests.test_gen_rendered_labels import pd
        k = int(k_new) if pd(insp) >= pd(eff) else int(k_old)
        n, d = int(units), int(d)
        pmf = [Fraction(math.comb(d, x) * math.comb(n - d, k - x), math.comb(n, k)) for x in range(k + 1)]
        if it["question"]["type"] == "noul":
            return {"yes": 1 - pmf[0], "no": pmf[0]}
        return {"0": pmf[0], "1": sum(pmf[1:2], Fraction(0)), "2": sum(pmf[2:3], Fraction(0)), "3": sum(pmf[3:], Fraction(0))}
    if sub == "connection":
        days, within, middle = re.search(r"last (\d+) days: at most \d+ minutes late \(or early\) (\d+) times; "
                                         r"\d+ to \d+ minutes late (\d+) times", s).groups()
        out_days, late = re.search(r"last (\d+) days: departed at least 20 minutes late on (\d+) days", s).groups()
        make = Fraction(int(within), int(days)) + Fraction(int(middle), int(days)) * Fraction(int(late), int(out_days))
        return {"yes": 1 - make, "no": make}
    if sub == "urn":
        t = int(re.search(r"On 1 to (\d+) the ticket", s).group(1))
        a, b = (tuple(map(int, x)) for x in re.findall(r"holds (\d+) gold, (\d+) silver and (\d+) bronze", s))
        pa = Fraction(t, 6)
        return {k: pa * Fraction(a[i], sum(a)) + (1 - pa) * Fraction(b[i], sum(b))
                for i, k in enumerate(("gold", "silver", "bronze"))}
    if sub == "reference_class":
        from tests.test_gen_rendered_labels import pd
        a_name, b_name = re.search(r"with the same (.+?) and the same (.+?) that were", s).groups()
        cutoff = pd(re.search(r" after (.+?), when ", s).group(1))
        m = re.search(rf"New \w+ [\w-]+: {re.escape(a_name)} (.+?), {re.escape(b_name)} (.+?), ", s)
        a, b = m.groups()
        rows = re.findall(r"^\| ([\w-]+) \| (.+?) \| (.+?) \| (\d{4}-\d{2}-\d{2}) \| (closed|open) \| (.+?) \|$", s, re.M)
        comp = [r for r in rows if r[1] == a and r[2] == b and date.fromisoformat(r[3]) > cutoff and r[4] == "closed"]
        return {o: Fraction(sum(r[5] == o for r in comp), len(comp)) for o in it["labels"]}
    raise KeyError(sub)


def test_rendered_text_reproduces_target_probs(groups):
    seen = set()
    for items in groups:
        for it in items:
            dist = from_text(it)
            if it["question"]["type"] == "noul" and "yes" not in it["target_probs"]:
                continue
            for k, v in it["target_probs"].items():
                assert abs(float(dist[k]) - v) < 1e-9, (it["provenance"]["rationale"], k)
            seen.add(it["provenance"]["rationale"].split("]")[0].split(":")[1])
    assert seen == {"bayes", "sampling", "connection", "urn", "reference_class"}
