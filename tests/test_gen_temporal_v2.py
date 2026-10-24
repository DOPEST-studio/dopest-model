"""Independent label check for temporal_v2: every answer is recomputed from the RENDERED TEXT with plain regexes and
the stdlib, never with the generator's engine. Catches engine bugs and render/engine mismatches."""
import re
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from dopest_model.gen import temporal_v2 as tv

ROW = re.compile(r"^(\d{2}) (\w{3}): usage ([\d,]+\.\d{2}); credits ([\d,]+\.\d{2}); tax (\d+)% ([\d,]+\.\d{2})$", re.M)
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def items(task_index: int, n: int = 120) -> list[dict]:
    out = []
    for seed in range(task_index, 2 * n, 2):
        out += tv.generate_case(seed)
    return out


def ledger_fire(state: str, pct: int, at_or_above: bool, credits=True, tax=False) -> int | None:
    budget = Decimal(re.search(r"Spend budget for the month: USD ([\d,]+)\.00", state).group(1).replace(",", ""))
    threshold = budget * pct / 100
    total = Decimal(0)
    for day, mon, usage, credit, rate, tx in ROW.findall(state):
        total += Decimal(usage.replace(",", "")) * ((100 + int(rate)) / Decimal(100)) if tax else Decimal(usage.replace(",", ""))
        if credits:
            total -= Decimal(credit.replace(",", ""))
        if (total >= threshold) if at_or_above else (total > threshold):
            return int(day)
    return None


@pytest.mark.parametrize("it", items(0), ids=lambda it: it["id"])
def test_ledger_alert_labels_from_rendered_text(it):
    state = it["state"]
    pct = int(re.search(r"month (?:reaches or passes|passes) (\d+)% of the budget", state).group(1))
    at_or_above = "reaches or passes" in state
    fire = ledger_fire(state, pct, at_or_above)
    assert fire is not None
    q = it["question"]
    if q["type"] == "choice":
        assert int(it["expected"][-2:]) == fire
    elif q["type"] == "score":
        credit_day = int(re.search(r"Credit applied on (\d+) \w+: USD", state).group(1))
        gap = fire - credit_day
        assert it["expected"] == (0 if gap <= 0 else 1 if gap <= 3 else 2 if gap <= 7 else 3)
    elif "name the right day" in q["instructions"]:
        day_in_note = int(re.search(r"passed \d+% on (\d+) \w+; A1 went off", state).group(1))
        assert (it["expected"] == "yes") == (day_in_note == fire)
    else:
        cut = int(re.search(r"by the close of (\d+) \w+ \(inclusive\)", q["instructions"]).group(1))
        assert (it["expected"] == "yes") == (fire <= cut)


SHIFT = re.compile(r"^Duty (\d+): (\w{3}) (\d{2}) (\w{3}), (\d{2}:\d{2}) to (?:(\w{3}) (\d{2}) (\w{3}) )?(\d{2}:\d{2})$", re.M)
CHANGE = re.compile(r"Daylight-saving switch: on \w+ (\d{2}) (\w+) (\d{4}) at (\d{2}:\d{2}) local \w+ \(UTC([+-]\d+)\) the clocks move "
                    r"(forward|back) to (\d{2}:\d{2}) local \w+ \(UTC([+-]\d+)\)")


def parse_rota(state: str):
    year = int(re.search(r"Roster week examined: Monday (\d{2} \w+ (\d{4}))", state).group(2))
    week_monday = datetime.strptime(re.search(r"Roster week examined: Monday (\d{2} \w+ \d{4})", state).group(1), "%d %B %Y")
    cd, cm, cy, ct, off_b, direction, new_t, off_a = CHANGE.search(state).groups()
    change = datetime.strptime(f"{cd} {cm} {cy} {ct}", "%d %B %Y %H:%M")
    off_b, off_a = int(off_b), int(off_a)
    def to_utc(local):
        off = off_b if local < change else off_a
        return local - timedelta(hours=off)
    shifts = []
    for k, dow, d1, m1, t1, dow2, d2, m2, t2 in SHIFT.findall(state):
        s = datetime(year, MONTHS[m1], int(d1), *map(int, t1.split(":")))
        e_date = datetime(year, MONTHS[m2], int(d2)) if d2 else s.replace(hour=0, minute=0)
        e = e_date.replace(hour=int(t2[:2]), minute=int(t2[3:]))
        if e <= s:                     # same-day rendering but ends past midnight never happens; guard anyway
            e += timedelta(days=1)
        shifts.append((s, e))
    home_off = int(re.search(r"base office's clock \(UTC([+-]\d+)\)", state).group(1))
    return week_monday, to_utc, shifts, home_off


@pytest.mark.parametrize("it", items(1), ids=lambda it: it["id"])
def test_rota_dst_labels_from_rendered_text(it):
    state = it["state"]
    week_monday, to_utc, shifts, home_off = parse_rota(state)
    min_rest = int(re.search(r"at least (\d+) hours must elapse in real time", state).group(1))
    max_shift = int(re.search(r"Longest duty: (\d+) hours", state).group(1))
    weekly_max = int(re.search(r"Hours cap per roster week: (\d+) hours", state).group(1))
    utc = [(to_utc(s), to_utc(e)) for s, e in shifts]
    rests = [(utc[i + 1][0] - utc[i][1]).total_seconds() / 3600 for i in range(len(utc) - 1)]
    lengths = [(e - s).total_seconds() / 3600 for s, e in utc]
    wk_start = week_monday - timedelta(hours=home_off)
    week_hours = sum(l for (s, _), l in zip(utc, lengths) if wk_start <= s < wk_start + timedelta(days=7))
    n_rest, n_shift, wk = sum(r < min_rest for r in rests), sum(l > max_shift for l in lengths), week_hours > weekly_max
    q = it["question"]
    if q["type"] == "score":
        assert it["expected"] == min(3, n_rest + n_shift + int(wk))
    elif q["type"] == "choice":
        hit = [k for k, v in (("2_1", n_rest), ("2_2", n_shift), ("2_3", wk)) if v]
        assert it["expected"] == ("none" if not hit else hit[0] if len(hit) == 1 else "several")
    elif "staff delegate" in q["instructions"]:
        listed = re.findall(r"after duty (\d+)", re.search(r"Staff delegate's comment: \"(.*?)\"", state, re.S).group(1))
        real = [str(k) for k, r in enumerate(rests, 1) if r < min_rest]
        assert (it["expected"] == "yes") == (listed == real)
    else:
        assert (it["expected"] == "yes") == (n_rest == 0)
