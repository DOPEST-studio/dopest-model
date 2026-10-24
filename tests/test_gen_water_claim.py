import collections
import json

import pytest

from dopest_model.gen import water_claim as wc

CASES = 400


@pytest.fixture(scope="module")
def groups():
    return [wc.generate_case(seed) for seed in range(CASES)]


def test_deterministic():
    assert json.dumps(wc.generate_case(7)) == json.dumps(wc.generate_case(7))


def test_items_are_well_formed(groups):
    for items in groups:
        for it in items:
            q = it["question"]
            assert it["label_source"] == "programmatic" and it["provenance"]["license"] == "Apache-2.0"
            if q["type"] == "choice":
                assert list(q["criteria"]) == it["labels"] and it["expected"] in it["labels"]
                assert 4 <= len(it["labels"]) <= 6 or it["expected"].startswith("amount_")
            elif q["type"] == "noul":
                assert it["labels"] == ["no", "yes"] and it["expected"] in ("no", "yes")
                assert set(q["criteria"]) == {"true", "false"}
            else:
                assert 0 <= it["expected"] < len(q["criteria"])
            assert len(it["state"]) > 4000


def test_one_group_per_case_and_twins_flip(groups):
    flipped = 0
    for items in groups:
        assert len({it["group"] for it in items}) <= 1
        mains = [it for it in items if it["question"]["type"] == "choice"
                 and not str(it["expected"]).startswith("amount_")]
        base = [it for it in mains if it["provenance"]["rationale"].startswith("[base]")]
        twins = [it for it in mains if it["provenance"]["rationale"].startswith("[twin")]
        if base and twins:
            assert base[0]["expected"] != twins[0]["expected"]
            assert base[0]["provenance"]["style"] == twins[0]["provenance"]["style"]
            flipped += 1
    assert flipped > 0.8 * CASES


def test_twins_are_minimal_edits(groups):
    """A twin's file differs from the base file in a few lines only, so the flipped fact is what decides."""
    for items in groups[:100]:
        states = {it["provenance"]["rationale"].split("]")[0]: it["state"] for it in items}
        base = states.get("[base")
        for role, state in states.items():
            if role.startswith("[twin") and base:
                diff = set(state.splitlines()) ^ set(base.splitlines())
                assert len(diff) <= 6, (role, sorted(diff))


def test_no_gold_position_bias(groups):
    pos = collections.Counter()
    for items in groups:
        for it in items:
            if it["question"]["type"] == "choice":
                pos[it["labels"].index(it["expected"]) / len(it["labels"]) < 0.5] += 1
    total = sum(pos.values())
    assert abs(pos[True] / total - 0.5) < 0.12


def test_earlier_sign_falls_within_the_leak():
    """A sign noticed before the leak began would make the story incoherent, also after a twin mutation."""
    import random
    rng = random.Random(0)
    checked = 0
    for _ in range(3000):
        p, c = wc.sample_case(rng, wc.sample_policy(rng))
        cases = [c] + [t for t in (wc.mutate(rng, p, c, w) for w in wc.MUTATIONS) if t is not None]
        for case in cases:
            if case.noticed_on:
                assert 3 <= wc.days_between(case.noticed_on, case.discovery) < case.leak_days[0]
                checked += 1
    assert checked > 300


def test_engine_settlement_order():
    p = wc.Policy(insurer="X", form="HX-1", currency="USD", deductible=1000, seepage_days=14, vacancy_days=60,
                  notice_days=30, sublimit_old=10000, sublimit_new=15000, endorsement="HE-2291",
                  effective=wc.date(2026, 1, 1))
    c = wc.Case(discovery=wc.date(2026, 9, 1), reported=wc.date(2026, 9, 5), leak_days=(28, 42), concealed=True,
                noticed_on=None, away_days=75, furnished=True, renewal=wc.date(2026, 3, 15), estimate=22400)
    assert wc.decide(p, c) == ("pay_subject_to_15000_sublimit", 15000)          # deductible first, then the cap
    assert wc.decide(p, wc.replace(c, renewal=wc.date(2025, 12, 31)))[0] == "pay_subject_to_10000_sublimit"
    assert wc.decide(p, wc.replace(c, furnished=False))[0] == wc.DENY_VACANCY
    assert wc.decide(p, wc.replace(c, noticed_on=wc.date(2026, 8, 20)))[0] == wc.DENY_SEEPAGE
    assert wc.decide(p, wc.replace(c, missing=frozenset({"renewal"})))[0] == wc.MISSING
    assert wc.decide(p, wc.replace(c, leak_days=(0, 0), missing=frozenset({"renewal"})))[0] == wc.PAY_FULL
