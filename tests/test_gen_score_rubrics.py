import random

import pytest

from dopest_model.gen import judge_quality as jq
from dopest_model.gen import lead_scoring as ls
from dopest_model.gen import priority_rubric as pr

LEAD = ls.Lead(employees=40, budget=20000, budget_old=None, weeks=6, role="decision_maker",
               required=("API access", "offline mode", "custom reports"),
               product=("API access", "offline mode", "custom reports", "SMS alerts"), region="Poland",
               relationship="new", churn_months=0, pricing=True, withdrawn=False, currency="EUR")


def test_lead_points_follow_the_guide_thresholds_and_floors():
    assert ls.points(LEAD) == {"budget": 2, "timeline": 2, "contact": 2, "fit": 2, "engagement": 1, "relationship": 0}
    assert ls.level(LEAD) == (3, 9, "")
    assert ls.points(ls.Lead(**{**LEAD.__dict__, "budget": 19999}))["budget"] == 1
    assert ls.points(ls.Lead(**{**LEAD.__dict__, "weeks": 7}))["timeline"] == 1
    assert ls.points(ls.Lead(**{**LEAD.__dict__, "withdrawn": True}))["engagement"] == 0
    assert ls.points(ls.Lead(**{**LEAD.__dict__, "relationship": "churned", "churn_months": 24}))["relationship"] == -1
    assert ls.points(ls.Lead(**{**LEAD.__dict__, "relationship": "churned", "churn_months": 25}))["relationship"] == 0
    assert ls.level(ls.Lead(**{**LEAD.__dict__, "region": "Brazil"})) == (0, 9, "region")
    assert ls.level(ls.Lead(**{**LEAD.__dict__, "employees": 9})) == (1, 9, "size")
    assert ls.level(ls.Lead(**{**LEAD.__dict__, "budget": 4500, "role": "end_user", "pricing": False})) == (1, 4, "")


@pytest.mark.parametrize("seed", range(0, 40, 7))
def test_lead_cases_have_a_twin_and_no_computed_total_in_the_state(seed):
    items = ls.generate_case(seed)
    assert items and len({it["id"] for it in items}) == len(items)
    roles = {it["provenance"]["rationale"].split("]")[0][1:] for it in items}
    assert "base:lead" in roles
    for it in items:
        assert "LEAD SCORING GUIDE" in it["state"] and "Points {" not in it["state"] and "-> level" not in it["state"]
        assert it["question"]["type"] in ("score", "noul", "choice") and it["split"] in ("train", "calib", "test")
        if it["question"]["type"] == "score":
            assert 0 <= it["expected"] < 4 and len(it["question"]["criteria"]) == 4


def test_priority_rules_apply_in_order_with_floors_and_exceptions():
    base = dict(users=100, total=1000, revenue=0, workaround=False, tier="standard", security=False, data_loss=False,
                since_update_h=4)
    assert pr.apply(pr.Ticket(**base))[0] == 2                                           # 10% -> P2
    assert pr.apply(pr.Ticket(**{**base, "users": 99}))[0] == 1                           # just under 10% -> P3
    assert pr.apply(pr.Ticket(**{**base, "workaround": True}))[0] == 1                    # R3 lowers
    lvl, trail = pr.apply(pr.Ticket(**{**base, "workaround": True, "security": True}))
    assert lvl == 2 and [r for r, _, _ in trail] == ["R1"]                               # R3 skipped, R4 floor already met
    assert pr.apply(pr.Ticket(**{**base, "users": 20, "workaround": True, "security": True}))[0] == 2   # R4 raises P4 -> P2
    assert pr.apply(pr.Ticket(**{**base, "revenue": 5000}))[0] == 3 and pr.apply(pr.Ticket(**{**base, "revenue": 4999}))[0] == 2
    assert pr.apply(pr.Ticket(**{**base, "users": 50, "tier": "enterprise", "workaround": True}))[0] == 2  # R5 floor after R3
    assert pr.apply(pr.Ticket(**{**base, "since_update_h": 73}))[0] == 3 and pr.apply(pr.Ticket(**{**base, "since_update_h": 72}))[0] == 2
    assert pr.apply(pr.Ticket(**{**base, "users": 400, "revenue": 9000, "since_update_h": 100}))[0] == 3  # capped at P1


@pytest.mark.parametrize("seed", range(1, 40, 7))
def test_priority_cases_are_consistent(seed):
    items = pr.generate_case(seed)
    assert items and any("twin" in it["provenance"]["rationale"] for it in items)
    for it in items:
        assert "SUPPORT-OPS 2.1" in it["state"] and "Trail" not in it["state"]
        if it["question"]["type"] == "choice":
            assert it["expected"] in pr.RULES and set(it["labels"]) == set(pr.RULES)


def test_judge_levels_separate_the_main_figure_from_presentation():
    inv = jq.Invoice(lines=(("webcam", 2, 8000), ("desk lamp", 1, 4500), ("cable tray", 3, 1500)), voucher=2500,
                     tax_pct=20, currency="EUR", defects=(), slip=150)
    assert jq.subtotal(inv) == 25000 and jq.total(inv) == 27000
    assert jq.level(inv) == 4
    assert jq.level(jq.Invoice(**{**inv.__dict__, "defects": ("missing_currency",)})) == 3
    assert jq.level(jq.Invoice(**{**inv.__dict__, "defects": ("missing_currency", "missing_subtotal")})) == 2
    assert jq.level(jq.Invoice(**{**inv.__dict__, "defects": ("whole_unit_rounding",)})) == 3
    assert jq.level(jq.Invoice(**{**inv.__dict__, "defects": ("arithmetic_slip",)})) == 1
    assert jq.level(jq.Invoice(**{**inv.__dict__, "defects": ("skipped_item",)})) == 0
    order = jq.Invoice(**{**inv.__dict__, "defects": ("voucher_after_tax",)})
    fig = jq.response_figures(order)[0]
    assert fig == 30000 - 2500 and jq.level(order) == 1                                   # 27,500 vs 27,000: 1.9% off


@pytest.mark.parametrize("seed", range(2, 40, 7))
def test_judge_cases_render_request_and_response_and_agree_with_the_engine(seed):
    items = jq.generate_case(seed)
    assert items
    for it in items:
        assert it["state"].startswith("REQUEST:") and "RESPONSE:" in it["state"] and "-> level" not in it["state"]
        if it["question"]["type"] == "score":
            assert 0 <= it["expected"] < 5
        if it["question"]["type"] == "choice":
            assert it["expected"] in jq.DEFECTS
    stated_total_line = [l for l in items[0]["state"].splitlines() if "Final total due" in l]
    assert stated_total_line


def test_mutations_change_the_label_when_accepted():
    rng = random.Random(3)
    for _ in range(20):
        inv = jq.sample(rng)
        m = jq.mutate(rng, inv)
        assert set(m.defects) != set(inv.defects)
