"""Independent label check: recompute every temporal_numeric answer from the RENDERED TEXT with a separate, plain
implementation (regexes + stdlib), never from the generator's engine. Catches engine bugs and render/engine
mismatches (a number shown that is not the number used)."""
import calendar
import math
import re
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

import pytest

from dopest_model.gen import temporal_numeric as tn

DATE = r"(\d{4}-\d{2}-\d{2}|\d{1,2} [A-Z][a-z]+ \d{4}|[A-Z][a-z]+ \d{1,2}, \d{4})"


def pd(s: str) -> date:
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(s)


def num(s: str) -> Decimal:
    return Decimal(s.replace(",", ""))


def months_later(d: date, n: int) -> date:
    y, m = d.year + (d.month - 1 + n) // 12, (d.month - 1 + n) % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def one(pattern: str, text: str):
    m = re.search(pattern, text)
    assert m, pattern
    return m.groups()


def check_unit_limit(s: str, it: dict):
    lb, = one(r"manifest: ([\d,]+) lb", s)
    tare, = one(r"\(unit plate\): ([\d.]+) kg", s)
    n, w, net = one(r"Restraint fitted: (\d+) straps at ([\d.]+) kg each and one net of ([\d.]+) kg", s)
    limit, = one(r"at most ([\d,]+) kg", s)
    declared = math.ceil(num(lb) * Decimal("0.45359237") + num(tare) + int(n) * num(w) + num(net))
    return ("yes" if declared <= num(limit) else "no") if it["question"]["type"] == "noul" else f"kg_{declared}"


def check_working_days(s: str, it: dict):
    old, = one(r"within (\d+) working days", s)
    since, new = one(rf"published on or after {DATE}: for those decisions the period in R1 is (\d+) working days", s)
    hol_text, = one(r"Public holidays: (.*?)\. (?:The registry|No registry)", s)
    holidays = {pd(x) for x in re.findall(DATE, hol_text)}
    closure = re.search(rf"closed from {DATE} to {DATE} inclusive", s)
    closed = (pd(closure.group(1)), pd(closure.group(2))) if closure else None
    published = pd(one(rf"published on the portal on \w+ {DATE}", s)[0])
    lodged = pd(one(rf"lodged on {DATE}", s)[0])
    limit = int(new) if published >= pd(since) else int(old)
    d, k = published, 0
    while k < limit:
        d += timedelta(days=1)
        if d.weekday() < 5 and d not in holidays and not (closed and closed[0] <= d <= closed[1]):
            k += 1
    return ("yes" if lodged <= d else "no") if it["question"]["type"] == "noul" else d.strftime("d_%Y_%m_%d")


def check_threshold_day(s: str, it: dict):
    budget, = one(r"monthly budget is [A-Z]{3} ([\d,]+\.\d{2})", s)
    pct, = one(r"at or above (\d+)% of", s)
    threshold, total = num(budget) * int(pct) / 100, Decimal(0)
    rows = re.findall(rf"{DATE}: usage [A-Z]{{3}} ([\d,]+\.\d{{2}})(?:, tax [A-Z]{{3}} [\d,]+\.\d{{2}})?"
                      rf"((?:, credit -[A-Z]{{3}} [\d,]+\.\d{{2}} \([^)]*\) posted)*)\.", s)
    assert len(rows) >= 9, "every daily line must parse"
    for d, usage, credits in sorted(rows, key=lambda r: pd(r[0])):
        total += num(usage) - sum((num(c) for c in re.findall(r"credit -[A-Z]{3} ([\d,]+\.\d{2})", credits)),
                                  Decimal(0))
        if total >= threshold:
            return pd(d).strftime("d_%Y_%m_%d")
    return "does_not_fire"


def check_prorata_refund(s: str, it: dict):
    notice, = one(r"terminate with (\d+) calendar days", s)
    fee_m = re.search(r"early-termination fee of [A-Z]{3} ([\d,]+\.\d{2})", s)
    start, fee = one(rf"Current term started {DATE}; annual fee paid in advance: ([A-Z]{{3}}) ([\d,]+\.\d{{2}})", s)[0::2]
    cur, = one(r"annual fee paid in advance: ([A-Z]{3})", s)
    start, received = pd(start), pd(one(rf"reached our contracts inbox on {DATE}", s)[0])
    end = months_later(start, 12) - timedelta(days=1)
    effective = received + timedelta(days=int(notice))
    refund = (num(fee) * (end - effective).days / ((end - start).days + 1)).quantize(Decimal("0.01"), ROUND_HALF_UP)
    if fee_m and effective < months_later(start, 6):
        refund = max(Decimal(0), refund - num(fee_m.group(1)))
    return f"{cur.lower()}_{int(refund)}_{int((refund - int(refund)) * 100):02d}"


def eu_summer_utc(t: datetime) -> bool:
    def last_sun(y, m):
        d = date(y, m, calendar.monthrange(y, m)[1])
        return d - timedelta(days=(d.weekday() - 6) % 7)
    return datetime.combine(last_sun(t.year, 3), datetime.min.time()) + timedelta(hours=1) <= t < \
        datetime.combine(last_sun(t.year, 10), datetime.min.time()) + timedelta(hours=1)


def check_period_end(s: str, it: dict):
    months, = one(r"Cover runs for (\d+) months", s)
    base, = one(r"time is UTC([+-]\d+) in standard time", s)
    delivery = pd(one(rf"Delivery Date on the signed delivery note: {DATE}", s)[0])
    d, hh, mm, off = one(rf"recorded by the portal: \w+ {DATE} (\d{{2}}):(\d{{2}}) \(UTC([+-]\d+)\)", s)
    end = months_later(delivery, int(months))
    midnight = datetime.combine(end + timedelta(days=1), datetime.min.time())
    offset = int(base) + (1 if eu_summer_utc(midnight - timedelta(hours=int(base) + 1)) else 0)
    claim_utc = datetime.combine(pd(d), datetime.min.time()) + timedelta(hours=int(hh) - int(off), minutes=int(mm))
    if it["question"]["type"] == "noul":
        return "yes" if claim_utc < midnight - timedelta(hours=offset) else "no"
    return end.strftime("d_%Y_%m_%d")


def check_rest_periods(s: str, it: dict):
    need, = one(r"rest of at least (\d+) consecutive hours", s)
    shifts = []
    for m in re.finditer(rf"Shift \d+: [^,]+, \w+ {DATE} (\d{{2}}):(\d{{2}}) \(UTC([+-]\d+)\) to \w+ {DATE} "
                         rf"(\d{{2}}):(\d{{2}}) \(UTC([+-]\d+)\)", s):
        g = m.groups()
        to_utc = lambda d, h, mi, o: datetime.combine(pd(d), datetime.min.time()) + timedelta(hours=int(h) - int(o),
                                                                                             minutes=int(mi))
        shifts.append((to_utc(*g[0:4]), to_utc(*g[4:8])))
    assert len(shifts) >= 5
    short = sum((b[0] - a[1]).total_seconds() / 3600 < int(need) for a, b in zip(shifts, shifts[1:]))
    return ("yes" if short == 0 else "no") if it["question"]["type"] == "noul" else min(3, short)


CHECKS = {"unit_limit": check_unit_limit, "working_days": check_working_days, "threshold_day": check_threshold_day,
          "prorata_refund": check_prorata_refund, "period_end": check_period_end, "rest_periods": check_rest_periods}


@pytest.mark.parametrize("task", sorted(CHECKS))
def test_rendered_text_supports_the_label(task):
    checked = 0
    for seed in range(1500):
        for it in tn.generate_case(seed):
            sub = it["provenance"]["rationale"].split("]")[0].split(":")[1]
            if sub != task:
                continue
            assert CHECKS[task](it["state"], it) == it["expected"], (seed, it["provenance"]["rationale"])
            checked += 1
    assert checked > 150
