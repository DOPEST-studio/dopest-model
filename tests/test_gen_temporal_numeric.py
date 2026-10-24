"""Engines checked against hand-computed cases, several taken from the rationales of public JevBench items."""
import json
from datetime import date, datetime
from decimal import Decimal

import pytest

from dopest_model.gen import temporal_numeric as tn


def test_period_end_leap_february_and_winter_offset():
    # 18 months after 31 Aug 2026 is 'February 31, 2028' -> 29 Feb 2028 (leap); 24:00 Rotterdam = 23:00 UTC
    w = tn.PeriodWorld(date(2026, 8, 31), 18, "Rotterdam", 1, datetime(2028, 2, 29, 22, 30), -4)
    end, deadline, offset = tn.PeriodEnd.deadline(w)
    assert (end, deadline, offset) == (date(2028, 2, 29), datetime(2028, 2, 29, 23, 0), 1)
    assert tn.PeriodEnd().solve(w) is True                        # 18:30 UTC-4 = 22:30 UTC, before 23:00 UTC
    assert tn.add_months_overflow(date(2026, 8, 31), 18) == date(2028, 3, 2)
    summer = tn.PeriodWorld(date(2027, 1, 31), 6, "Leeds", 0, datetime(2027, 7, 31, 22, 30), 3)
    assert tn.PeriodEnd.deadline(summer)[1:] == (datetime(2027, 7, 31, 23, 0), 1)


def test_working_days_year_end_closure():
    # Fri 11 Dec 2026 + 15 working days, closure 24 Dec-1 Jan -> Tue 12 Jan 2027
    w = tn.AppealWorld(date(2026, 12, 11), 10, 15, date(2026, 11, 1),
                       (date(2026, 12, 25), date(2026, 12, 26), date(2027, 1, 1)),
                       (date(2026, 12, 24), date(2027, 1, 1)), date(2027, 1, 12))
    wd = tn.WorkingDays()
    assert wd.deadline(w) == date(2027, 1, 12) and wd.solve(w)
    assert not wd.solve(tn.replace(w, lodged=date(2027, 1, 13)))
    # published before the amendment: the old 10-day limit -> 8 days in December, then Mon 4 and Tue 5 Jan
    assert wd.deadline(tn.replace(w, amended_from=date(2026, 12, 12))) == date(2027, 1, 5)
    assert wd.deadline(w, closure=False) < wd.deadline(w)


def test_threshold_day_credits_tax_and_equality():
    w = tn.AlertWorld(date(2026, 9, 1), 1000, 80, tuple(map(Decimal, ("300", "300", "250", "100"))),
                      ((1, Decimal("100")),), 21, "EUR")
    t = tn.ThresholdDay()
    assert t.fire(w) == 3 and t.fire(w, credits=False) == 2 and t.fire(w, tax=True) == 2
    exact = tn.replace(w, usage=tuple(map(Decimal, ("300", "300", "250", "50"))))      # net reaches exactly 800.00
    assert t.fire(exact) == 3 and t.fire(exact, strict=True) is None


def test_rest_across_offsets_is_real_elapsed_time():
    # Sat 28 Mar 09:00 UTC-4 (13:00 UTC) to Sun 29 Mar 01:00 UTC+1 (00:00 UTC) = exactly 11 h: complies
    w = tn.RosterWorld(((datetime(2026, 3, 28, 5), datetime(2026, 3, 28, 13), 0),
                        (datetime(2026, 3, 29, 0), datetime(2026, 3, 29, 10), 1)), (("NY", -4), ("Berlin", 1)), 11)
    assert tn.RestPeriods.rests(w) == [11.0] and tn.RestPeriods().solve(w) == 0
    assert tn.RestPeriods.rests(w, naive=True) == [16.0]


def test_prorata_refund_leap_term():
    # notice received 10 Feb 2028 + 30 days = 11 Mar 2028; term 1 Sep 2027-31 Aug 2028 = 366 days; 173 days left
    w = tn.RefundWorld(Decimal("1464.00"), "EUR", date(2027, 9, 1), date(2028, 2, 10), 30, Decimal(0))
    p = tn.ProrataRefund()
    assert p.parts(w) == (date(2028, 8, 31), date(2028, 3, 11), 366, 173)
    assert p.solve(w) == Decimal("692.00")
    assert p.refund(w, term_days=365) == Decimal("693.90")
    early = tn.replace(w, received=date(2027, 12, 1), early_fee=Decimal("100"))
    assert p.solve(early) == (Decimal("1464") * 244 / 366).quantize(tn.CENT) - 100


def test_unit_limit_restraint_and_rounding():
    # 5,280 lb -> 2,394.97 kg + 105 tare + 12.1 restraint = 2,512.07 -> 2,513 kg > 2,500; without restraint 2,500
    w = tn.LoadWorld(5280, Decimal("105"), 1, Decimal("2.1"), Decimal("10.0"), 2500)
    u = tn.UnitLimit()
    assert u.declared(w) == 2513 and not u.solve(w)
    assert u.declared(w, restraint=False) == 2500


@pytest.fixture(scope="module")
def groups():
    return [tn.generate_case(seed) for seed in range(600)]


def test_generated_items_well_formed_and_twins_flip(groups):
    flipped = 0
    for items in groups:
        assert items, "every seed yields a case"
        for it in items:
            q = it["question"]
            if q["type"] == "choice":
                assert list(q["criteria"]) == it["labels"] and it["expected"] in it["labels"]
                assert len(set(it["labels"])) == len(it["labels"]) >= 3
            elif q["type"] == "noul":
                assert it["expected"] in ("yes", "no")
            else:
                assert 0 <= it["expected"] < len(q["criteria"])
        tags = [it["provenance"]["rationale"].split("]")[0] for it in items]
        base = next(it for it, t in zip(items, tags) if t.startswith("[base"))
        twin = next((it for it, t in zip(items, tags) if t.startswith("[twin")), None)
        if twin:
            assert twin["question"]["instructions"] == base["question"]["instructions"]
            assert twin["expected"] != base["expected"]
            flipped += 1
    assert flipped > 0.85 * len(groups)


def test_deterministic():
    assert json.dumps(tn.generate_case(11), default=str) == json.dumps(tn.generate_case(11), default=str)
