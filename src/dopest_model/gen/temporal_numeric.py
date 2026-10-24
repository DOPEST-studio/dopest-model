"""Dates, clock times, units and running totals under stated rules (JevBench hard tier, family temporal_numeric).

Six small worlds, each with a rule engine, a renderer and a one-fact mutation for twins:
  period_end     months after a date with end-of-month clamping and leap years, a deadline at 24:00 local time
                 with summer time, and a claim stamped in another UTC offset
  working_days   an appeal period in working days with holidays, a registry closure and an amended limit
  threshold_day  the day a budget alert fires: credits count, taxes do not, 'at or above'
  rest_periods   daily-rest violations in a roster across UTC offsets (real elapsed time, not wall clocks)
  prorata_refund a pro-rata refund over a 365- or 366-day term with notice, rounding and an early-exit fee
  unit_limit     pounds to kilograms, gross = net + tare + restraint, round up, compare with a limit
Every rule the answer needs is in the state (offsets, holidays, factors, rounding), never in world knowledge.
"""
from __future__ import annotations

import calendar
import math
import random
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal

from .common import DATE_STYLES, SPLIT_BY_STYLE, Nar, choice_item, fmt_date, noul_item, render_file, score_item, \
    stable_id

FAMILY = "temporal_numeric"
GENERATOR = "temporal_numeric_v1"
CENT = Decimal("0.01")
LB = Decimal("0.45359237")


def add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def add_months_overflow(d: date, n: int) -> date:
    """The classic mistake: a day missing in the target month spills into the next month."""
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    last = calendar.monthrange(y, m)[1]
    return date(y, m, last) + timedelta(days=d.day - last) if d.day > last else date(y, m, d.day)


def last_sunday(y: int, m: int) -> date:
    d = date(y, m, calendar.monthrange(y, m)[1])
    return d - timedelta(days=(d.weekday() - 6) % 7)


def eu_summer(t_utc: datetime) -> bool:
    """EU rule: summer time from 01:00 UTC on the last Sunday of March to 01:00 UTC on the last Sunday of October."""
    y = t_utc.year
    return datetime.combine(last_sunday(y, 3), time(1)) <= t_utc < datetime.combine(last_sunday(y, 10), time(1))


def money(x: Decimal, cur: str) -> str:
    return f"{cur} {x:,.2f}"


def money_label(x: Decimal, cur: str) -> str:
    x = x.quantize(CENT)
    return f"{cur.lower()}_{int(x)}_{int((x - int(x)) * 100):02d}"


def date_label(d: date) -> str:
    return d.strftime("d_%Y_%m_%d")


def pick_options(rng: random.Random, gold, candidates, k: int) -> list:
    """Gold plus up to k-1 distinct wrong candidates, wrong ones from typical mistakes first."""
    out = [gold]
    for c in candidates:
        if c not in out:
            out.append(c)
        if len(out) == k:
            break
    return out


# ------------------------------------------------------------------ period_end

@dataclass(frozen=True)
class PeriodWorld:
    start: date
    months: int
    city: str
    base: int                 # standard-time UTC offset of the city
    submit_utc: datetime
    offset: int               # UTC offset of the claimant's timestamp


class PeriodEnd:
    name, use_case = "period_end", "customer_support"
    CITIES = [("Rotterdam", 1), ("Lyon", 1), ("Kraków", 1), ("Turin", 1), ("Dublin", 0), ("Porto", 0), ("Leeds", 0)]
    OFFSETS = [-8, -7, -6, -5, -4, -3, 3, 4, 5, 8, 9, 10]

    @staticmethod
    def deadline(w: PeriodWorld) -> tuple[date, datetime, int]:
        end = add_months(w.start, w.months)
        local = datetime.combine(end + timedelta(days=1), time(0))
        off = w.base + 1 if eu_summer(local - timedelta(hours=w.base + 1)) else w.base
        return end, local - timedelta(hours=off), off

    def solve(self, w: PeriodWorld) -> bool:
        return w.submit_utc < self.deadline(w)[1]

    @staticmethod
    def naive(w: PeriodWorld) -> bool:
        """Reading the claimant's wall clock against 24:00 on the end date, ignoring the offsets."""
        end = add_months(w.start, w.months)
        return w.submit_utc + timedelta(hours=w.offset) < datetime.combine(end + timedelta(days=1), time(0))

    def sample(self, rng: random.Random) -> PeriodWorld:
        months = rng.choice([6, 12, 18, 24, 30, 36])
        while True:
            y, m = rng.choice([2025, 2026, 2027]), rng.randint(1, 12)
            day = rng.choice([29, 30, 31, 31, 30, 29, rng.randint(1, 28)])
            if day <= calendar.monthrange(y, m)[1]:
                start = date(y, m, day)
                if add_months(start, months).day != day or rng.random() < 0.3:
                    break
        city, base = rng.choice(self.CITIES)
        w = PeriodWorld(start, months, city, base, datetime(2000, 1, 1), 0)
        dl = self.deadline(w)[1]
        submit = dl + timedelta(minutes=rng.randint(15, 240) * rng.choice([-1, 1]))
        offsets = rng.sample(self.OFFSETS, len(self.OFFSETS))
        w = replace(w, submit_utc=submit)
        tricky = [o for o in offsets if self.naive(replace(w, offset=o)) != self.solve(w)]
        return replace(w, offset=tricky[0] if tricky and rng.random() < 0.7 else offsets[0])

    def mutate(self, rng: random.Random, w: PeriodWorld) -> PeriodWorld:
        dl = self.deadline(w)[1]
        sign = 1 if w.submit_utc < dl else -1
        return replace(w, submit_utc=dl + timedelta(minutes=sign * rng.randint(15, 180)))

    def items(self, rng: random.Random, w: PeriodWorld, n: Nar, base: dict) -> list[dict]:
        r = n.rng(1)
        cert, claim = f"EW-{r.randint(1000000, 9999999)}", f"CL-{r.randint(10, 99)}-{r.randint(10000, 99999)}"
        product = r.choice(["espresso machine", "washing machine", "e-bike", "laptop", "heat pump controller"])
        purchase = w.start - timedelta(days=r.randint(2, 20))
        end, dl, off = self.deadline(w)
        local = w.submit_utc + timedelta(hours=w.offset)
        rules = (f"EXTENDED WARRANTY CERTIFICATE {cert}: TERMS\n"
                 f"1. Cover runs for {w.months} months from the Delivery Date.\n"
                 f"2. A period of months ends on the date in its final month that has the same day number as the "
                 f"Delivery Date. If the final month has no such date, the period ends on the last day of that month.\n"
                 f"3. The period ends at 24:00 {w.city} time on its final day.\n"
                 f"4. {w.city} time is UTC{w.base:+d} in standard time and UTC{w.base + 1:+d} in summer time. Summer "
                 f"time starts at 01:00 UTC on the last Sunday of March and ends at 01:00 UTC on the last Sunday of "
                 f"October.\n"
                 f"5. A claim is reported when the claims portal records it. Portal timestamps show the claimant's "
                 f"local time and its UTC offset.")
        facts = [(purchase, f"Invoice date (purchase of the {product}): {n.d(purchase)}."),
                 (w.start, f"Delivery Date on the signed delivery note: {n.d(w.start)}."),
                 (w.start + timedelta(days=r.randint(5, 50)), "The product was registered online after delivery."),
                 (local.date(), f"Claim {claim} recorded by the portal: {n.dt(local, w.offset)}.")]
        state = f"{rules}\n\n" + render_file(n.style, f"WARRANTY CLAIM {claim}", f"Certificate {cert}", facts)
        inside = self.solve(w)
        why = (f"{w.months} months from {w.start} ends {end}; the period ends 24:00 {w.city} time (UTC{off:+d}) = "
               f"{dl:%Y-%m-%d %H:%M} UTC. Claim {local:%Y-%m-%d %H:%M} UTC{w.offset:+d} = "
               f"{w.submit_utc:%Y-%m-%d %H:%M} UTC: {'in time' if inside else 'late'}.")
        common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
        out = [noul_item(instructions=f"Was claim {claim} reported within the extended warranty period of "
                                      f"certificate {cert}?",
                         true_text="The claim was reported before the period ended.",
                         false_text="The claim was reported after the period ended.", answer=inside, **common)]
        if rng.random() < 0.5:
            wrong = [add_months_overflow(w.start, w.months), end - timedelta(days=1), end + timedelta(days=1),
                     add_months(w.start, w.months - 1)]
            dates = pick_options(rng, end, wrong, 4)
            out.append(choice_item(rng=rng, options={date_label(x): f"The period ends on {n.d(x)}." for x in dates},
                                   expected=date_label(end), trap=date_label(dates[1]),
                                   instructions=f"On which date does the cover under certificate {cert} end?",
                                   **common))
        return out


# ------------------------------------------------------------------ working_days

@dataclass(frozen=True)
class AppealWorld:
    published: date
    limit_old: int
    limit_new: int
    amended_from: date
    holidays: tuple
    closure: tuple | None
    lodged: date


class WorkingDays:
    name, use_case = "working_days", "legal_compliance"

    @staticmethod
    def working(w: AppealWorld, d: date, holidays: bool = True, closure: bool = True) -> bool:
        if d.weekday() >= 5:
            return False
        if holidays and d in w.holidays:
            return False
        return not (closure and w.closure and w.closure[0] <= d <= w.closure[1])

    @staticmethod
    def limit(w: AppealWorld) -> int:
        return w.limit_new if w.published >= w.amended_from else w.limit_old

    def deadline(self, w: AppealWorld, limit: int | None = None, from_day: date | None = None, **kw) -> date:
        d, k, n = from_day or w.published, 0, limit or self.limit(w)
        while k < n:
            d += timedelta(days=1)
            k += self.working(w, d, **kw)
        return d

    def solve(self, w: AppealWorld) -> bool:
        return w.lodged <= self.deadline(w)

    def sample(self, rng: random.Random) -> AppealWorld:
        old = rng.choice([10, 15, 20])
        new = rng.choice([x for x in (5, 10, 15, 20, 21) if x != old])
        if rng.random() < 0.5:                                        # year-end: holidays and a registry closure
            y = rng.choice([2025, 2026, 2027])
            published = date(y, 12, rng.randint(1, 18))
            holidays = (date(y, 12, 25), date(y, 12, 26), date(y + 1, 1, 1)) + \
                ((date(y + 1, 1, 6),) if rng.random() < 0.5 else ())
            start = date(y, 12, rng.choice([23, 24]))
            closure = (start, date(y + 1, 1, rng.choice([1, 2])))
        else:
            published = date(rng.choice([2026, 2027]), rng.randint(1, 11), rng.randint(1, 28))
            span = [published + timedelta(days=i) for i in range(2, 30)]
            holidays = tuple(sorted(rng.sample([d for d in span if d.weekday() < 5], rng.randint(1, 3))))
            if rng.random() < 0.5:
                c0 = published + timedelta(days=rng.randint(3, 20))
                closure = (c0, c0 + timedelta(days=rng.randint(1, 3)))
            else:
                closure = None
        amended = published + timedelta(days=rng.randint(-40, 40))
        w = AppealWorld(published, old, new, amended, holidays, closure, published)
        dl = self.deadline(w)
        shift = rng.choice([-3, -1, 0, 0, 1, 1, 2])
        lodged = dl
        step = 1 if shift > 0 else -1
        for _ in range(abs(shift)):
            lodged += timedelta(days=step)
            while not self.working(w, lodged):
                lodged += timedelta(days=step)
        return replace(w, lodged=lodged)

    def mutate(self, rng: random.Random, w: AppealWorld) -> AppealWorld:
        dl = self.deadline(w)
        if rng.random() < 0.3 and self.deadline(replace(w, amended_from=w.published + timedelta(days=1))) != \
                self.deadline(replace(w, amended_from=w.published)):
            flipped = w.published + timedelta(days=1) if w.published >= w.amended_from else w.published
            return replace(w, amended_from=flipped)
        if w.lodged <= dl:
            later = dl + timedelta(days=1)
            while not self.working(w, later):
                later += timedelta(days=1)
            return replace(w, lodged=later)
        return replace(w, lodged=dl)

    def items(self, rng: random.Random, w: AppealWorld, n: Nar, base: dict) -> list[dict]:
        r = n.rng(2)
        dec, apl, amend = (f"DEC-{w.published.year}-{r.randint(1000, 9999)}", f"APL-{r.randint(1000, 9999)}",
                           f"S/{w.amended_from.year}/{r.randint(10, 99)}")
        posted = w.published + timedelta(days=r.randint(2, 4))
        hol = ", ".join(f"{x:%a} {n.d(x)}" for x in w.holidays)
        clo = (f"The registry is closed from {n.d(w.closure[0])} to {n.d(w.closure[1])} inclusive."
               if w.closure else "No registry closures are scheduled in the period.")
        rules = (f"APPEALS AND ADMISSIBILITY RULES (AAR), AS AMENDED\n"
                 f"R1. An appeal against a registry decision must be lodged within {w.limit_old} working days.\n"
                 f"R2. Amendment {amend} applies to decisions published on or after {n.d(w.amended_from)}: for "
                 f"those decisions the period in R1 is {w.limit_new} working days.\n"
                 f"R3. The period starts on the first working day after the day on which the decision is published "
                 f"on the portal. Sending a paper copy does not change the publication date.\n"
                 f"R4. Working days are Monday to Friday, except public holidays and days on which the registry is "
                 f"closed.\nR5. An appeal is in time if it is lodged on or before the last working day of the "
                 f"period.\nPublic holidays: {hol}. {clo}")
        facts = [(w.published, f"Decision {dec} was published on the portal on {w.published:%A} "
                               f"{n.d(w.published)}."),
                 (posted, f"A paper copy was posted to the appellant on {n.d(posted)}."),
                 (w.lodged, f"Appeal {apl} against {dec} was lodged on {n.d(w.lodged)}.")]
        state = f"{rules}\n\n" + render_file(n.style, f"APPEAL {apl}", f"Decision {dec}", facts)
        dl, lim = self.deadline(w), self.limit(w)
        ok = self.solve(w)
        why = (f"Published {w.published} ({'on or after' if w.published >= w.amended_from else 'before'} "
               f"{w.amended_from}): {lim} working days from the next working day, skipping holidays and closures, "
               f"ends {dl}. Lodged {w.lodged}: {'in time' if ok else 'late'}.")
        common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
        out = [noul_item(instructions=f"Was appeal {apl} lodged in time under the AAR as amended?",
                         true_text="The appeal was lodged within the period.",
                         false_text="The appeal was lodged after the period ended.", answer=ok, **common)]
        if rng.random() < 0.5:
            other = w.limit_old if lim == w.limit_new else w.limit_new
            wrong = [self.deadline(w, holidays=False), self.deadline(w, closure=False),
                     self.deadline(w, limit=other), self.deadline(w, from_day=w.published - timedelta(days=1)),
                     w.published + timedelta(days=lim)]
            dates = pick_options(rng, dl, wrong, 4)
            out.append(choice_item(rng=rng, options={date_label(x): f"The last day is {x:%A} {n.d(x)}." for x in dates},
                                   expected=date_label(dl), trap=date_label(dates[1]),
                                   instructions=f"What is the last day for lodging an appeal against {dec}?", **common))
        return out


# ------------------------------------------------------------------ threshold_day

@dataclass(frozen=True)
class AlertWorld:
    first: date
    budget: int
    pct: int
    usage: tuple              # Decimal per day, from `first`
    credits: tuple            # (day index, Decimal)
    tax_rate: int
    currency: str


class ThresholdDay:
    name, use_case = "threshold_day", "risk_assessment"

    @staticmethod
    def fire(w: AlertWorld, credits: bool = True, tax: bool = False, strict: bool = False) -> int | None:
        threshold = Decimal(w.budget) * w.pct / 100
        total = Decimal(0)
        for i, u in enumerate(w.usage):
            total += u * (1 + Decimal(w.tax_rate) / 100) if tax else u
            if credits:
                total -= sum((c for j, c in w.credits if j == i), Decimal(0))
            if total > threshold if strict else total >= threshold:
                return i
        return None

    def solve(self, w: AlertWorld) -> int | None:
        return self.fire(w)

    def sample(self, rng: random.Random) -> AlertWorld:
        n = rng.randint(9, 14)
        f = rng.randint(5, n - 2)
        first = date(rng.choice([2026, 2027]), rng.randint(1, 12), rng.randint(1, 10))
        usage = [Decimal(rng.randint(15000, 90000)) / 100 for _ in range(n)]
        credits = tuple(sorted((rng.randint(1, f - 1), Decimal(rng.randint(20, 90)) * 10)
                               for _ in range(rng.randint(1, 2))))
        net_before = sum(usage[:f]) - sum(c for _, c in credits)
        for budget in sorted(range(1000, 40001, 500), key=lambda _: rng.random()):
            pct = rng.choice([75, 80, 90])
            gap = Decimal(budget) * pct / 100 - net_before
            if Decimal(50) < gap < Decimal(1200):
                break
        else:
            return None
        extra = Decimal(0) if rng.random() < 0.3 else Decimal(rng.randint(1, 30000)) / 100
        usage[f] = gap + extra
        return AlertWorld(first, budget, pct, tuple(usage), credits, rng.choice([0, 8, 19, 21, 23]),
                          rng.choice(["EUR", "USD", "GBP"]))

    def mutate(self, rng: random.Random, w: AlertWorld) -> AlertWorld | None:
        f = self.fire(w)
        if f is None:
            return None
        usage = list(w.usage)
        usage[f] = max(Decimal("1.00"), usage[f] - Decimal(rng.randint(5000, 40000)) / 100)
        return replace(w, usage=tuple(usage))

    def items(self, rng: random.Random, w: AlertWorld, n: Nar, base: dict) -> list[dict]:
        r = n.rng(3)
        project = r.choice(["Helios", "Lumen", "Atlas", "Quarry", "Nimbus"]) + f"-{r.randint(10, 99)}"
        reasons = ["goodwill credit", "SLA credit", "billing correction", "promotional credit"]
        days = [w.first + timedelta(days=i) for i in range(len(w.usage))]
        facts = []
        for i, (d, u) in enumerate(zip(days, w.usage)):
            line = f"{n.d(d)}: usage {money(u, w.currency)}"
            if w.tax_rate:
                line += f", tax {money((u * w.tax_rate / 100).quantize(CENT, ROUND_HALF_UP), w.currency)}"
            for j, c in w.credits:
                if j == i:
                    line += f", credit -{money(c, w.currency)} ({r.choice(reasons)}) posted"
            facts.append((d, line + "."))
        rules = (f"BUDGET ALERT RULE A1 (project {project})\n"
                 f"- The monthly budget is {money(Decimal(w.budget), w.currency)}.\n"
                 f"- A1 fires at the end of the first day on which month-to-date net cost is at or above {w.pct}% of "
                 f"the monthly budget.\n- Net cost is usage charges minus credits. A credit counts from the day it is "
                 f"posted.\n- Taxes are shown for information only and are not part of net cost.\n"
                 f"- Days not listed had no charges.")
        state = f"{rules}\n\n" + render_file(n.style, f"DAILY BILLING LOG, {project}", "Month to date", facts)
        f = self.solve(w)
        label = lambda i: "does_not_fire" if i is None else date_label(days[i])
        text = lambda i: "A1 does not fire on any listed day." if i is None else f"A1 fires on {n.d(days[i])}."
        threshold = Decimal(w.budget) * w.pct / 100
        why = (f"Threshold {w.pct}% of {w.budget} = {threshold:.2f} of net cost (usage minus credits, no tax). "
               f"First day at or above: {'none' if f is None else days[f]}.")
        wrong = [self.fire(w, credits=False), self.fire(w, tax=True), self.fire(w, strict=True)]
        neighbours = [x for x in ((f or 0) - 1, (f or 0) + 1) if 0 <= x < len(days)]
        picked = pick_options(rng, f, wrong + neighbours + [None], rng.randint(4, 5))
        common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
        return [choice_item(rng=rng, options={label(i): text(i) for i in picked}, expected=label(f),
                            trap=label(picked[1]), instructions=f"On which day does alert rule A1 fire for project "
                                                                f"{project}?", **common)]


# ------------------------------------------------------------------ rest_periods

@dataclass(frozen=True)
class RosterWorld:
    shifts: tuple             # (start_utc, end_utc, site index)
    sites: tuple              # (name, UTC offset)
    min_rest: int


class RestPeriods:
    name, use_case = "rest_periods", "legal_compliance"
    SITES = [("Eastport depot", -5), ("Westfield site", -4), ("Northgate office", 1), ("Harbourside hub", 2),
             ("Ridgeview plant", 3), ("Bayline lab", 8), ("Summit yard", 9)]

    @staticmethod
    def rests(w: RosterWorld, naive: bool = False) -> list[float]:
        out = []
        for (_, e, si), (s, _, sj) in zip(w.shifts, w.shifts[1:]):
            if naive:          # wall-clock difference, as if both clocks were the same zone
                s, e = s + timedelta(hours=w.sites[sj][1]), e + timedelta(hours=w.sites[si][1])
            out.append((s - e).total_seconds() / 3600)
        return out

    def solve(self, w: RosterWorld) -> int:
        return min(3, sum(r < w.min_rest for r in self.rests(w)))

    def sample(self, rng: random.Random) -> RosterWorld:
        a, b = rng.sample(range(len(self.SITES)), 2)
        if abs(self.SITES[a][1] - self.SITES[b][1]) < 4:
            return None
        n = rng.randint(5, 7)
        sites = [a] * rng.randint(1, n - 2)
        sites += [b] * (n - len(sites))
        if rng.random() < 0.4:
            sites[-1] = a
        t = datetime(rng.choice([2026, 2027]), rng.randint(1, 12), rng.randint(1, 20), rng.randint(5, 20))
        p_short = rng.choice([0.0, 0.1, 0.2, 0.35])          # per roster, so compliant rosters are common too
        shifts = []
        for i, si in enumerate(sites):
            end = t + timedelta(hours=rng.choice([6, 7, 8, 8.5, 9, 10]))
            shifts.append((t, end, si))
            rest = rng.choice([8.5, 9, 10, 10.5]) if rng.random() < p_short else rng.choice([11, 11, 11.5, 12, 13,
                                                                                            14, 16])
            nxt = sites[i + 1] if i + 1 < len(sites) else si
            diff = self.SITES[nxt][1] - self.SITES[si][1]
            if nxt != si and rng.random() < 0.7:
                # Where the wall clock lies: eastward a short rest looks long, westward a legal rest looks short.
                rest = rng.choice([8.5, 9, 10, 10.5]) if diff > 0 else rng.choice(
                    [x / 2 for x in range(22, 22 + 2 * abs(diff) - 1)])
            t = end + timedelta(hours=rest)
        return RosterWorld(tuple(shifts), tuple(self.SITES), 11)

    def mutate(self, rng: random.Random, w: RosterWorld) -> RosterWorld | None:
        i = rng.randrange(1, len(w.shifts))
        s, e, si = w.shifts[i]
        rest = self.rests(w)[i - 1]
        delta = (w.min_rest - rest + rng.choice([0, 0.5, 1])) if rest < w.min_rest else -(rest - w.min_rest +
                                                                                           rng.choice([0.5, 1, 2]))
        new_start = s + timedelta(hours=delta)
        if (e - new_start).total_seconds() < 3 * 3600:
            return None
        shifts = list(w.shifts)
        shifts[i] = (new_start, e, si)
        return replace(w, shifts=tuple(shifts))

    def items(self, rng: random.Random, w: RosterWorld, n: Nar, base: dict) -> list[dict]:
        r = n.rng(4)
        worker = f"{r.choice(['Priya', 'Tomasz', 'Ines', 'Owen', 'Aiko', 'Rafael'])} " \
                 f"{r.choice(['Anand', 'Novak', 'Moreau', 'Quinn', 'Sato', 'Delgado'])}"
        lines = []
        for k, (s, e, si) in enumerate(w.shifts, 1):
            name, off = w.sites[si]
            ls, le = s + timedelta(hours=off), e + timedelta(hours=off)
            lines.append((ls.date(), f"Shift {k}: {name}, {n.dt(ls, off)} to {n.dt(le, off)}."))
        rules = (f"WORKING TIME STANDARD WTS-4\n"
                 f"4.1 Between the end of one shift and the start of the next there must be a rest of at least "
                 f"{w.min_rest} consecutive hours, measured in real elapsed time. Exactly {w.min_rest} hours "
                 f"complies.\n4.2 Roster times are local times at the site, shown with the site's UTC offset on that "
                 f"date.\n4.3 Travel between sites happens within the rest period.")
        state = f"{rules}\n\n" + render_file(n.style, f"ROSTER, {worker.upper()}", "Rota as published", lines)
        count = self.solve(w)
        rests = self.rests(w)
        why = ("Real rests (h): " + ", ".join(f"{x:g}" for x in rests) + f"; below {w.min_rest}: "
               f"{sum(x < w.min_rest for x in rests)}. Wall-clock rests would be: "
               + ", ".join(f"{x:g}" for x in self.rests(w, naive=True)) + ".")
        common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
        out = [score_item(instructions=f"How many WTS-4 rest violations does {worker}'s roster contain? Output the "
                                       f"level index (0 = none, 1 = one, 2 = two, 3 = three or more).",
                          levels=["No violations", "Exactly one violation", "Exactly two violations",
                                  "Three or more violations"], expected=count, **common)]
        if rng.random() < 0.4:
            out.append(noul_item(instructions=f"Does {worker}'s roster comply with WTS-4 section 4.1?",
                                 true_text="Every rest between shifts meets 4.1.",
                                 false_text="At least one rest between shifts is too short.", answer=count == 0,
                                 **common))
        return out


# ------------------------------------------------------------------ prorata_refund

@dataclass(frozen=True)
class RefundWorld:
    fee: Decimal
    currency: str
    start: date
    received: date
    notice_days: int
    early_fee: Decimal


class ProrataRefund:
    name, use_case = "prorata_refund", "customer_support"

    @staticmethod
    def parts(w: RefundWorld) -> tuple[date, date, int, int]:
        end = add_months(w.start, 12) - timedelta(days=1)
        effective = w.received + timedelta(days=w.notice_days)
        return end, effective, (end - w.start).days + 1, (end - effective).days

    def refund(self, w: RefundWorld, term_days: int | None = None, extra_day: int = 0, from_received: bool = False,
               fee: str = "rule") -> Decimal:
        end, effective, days, remaining = self.parts(w)
        if from_received:
            remaining, effective = (end - w.received).days, w.received
        value = (w.fee * (remaining + extra_day) / (term_days or days)).quantize(CENT, ROUND_HALF_UP)
        early = effective < add_months(w.start, 6)
        if fee == "always" or fee == "rule" and early:
            value -= w.early_fee
        return max(Decimal(0), value)

    def solve(self, w: RefundWorld) -> Decimal:
        return self.refund(w)

    def sample(self, rng: random.Random) -> RefundWorld:
        start = date(rng.choice([2026, 2027, 2027]), rng.randint(1, 12), 1)
        days = (add_months(start, 12) - start).days
        received = start + timedelta(days=rng.randint(60, days - 60))
        return RefundWorld(Decimal(rng.randint(48, 400)) * 6, rng.choice(["EUR", "USD", "GBP"]), start, received,
                           rng.choice([14, 30, 30, 60]), Decimal(rng.choice([0, 50, 75, 120, 150])))

    def mutate(self, rng: random.Random, w: RefundWorld) -> RefundWorld:
        six = add_months(w.start, 6) - timedelta(days=w.notice_days)       # received on this day: effective at 6 months
        received = six + timedelta(days=rng.randint(1, 20)) if w.received < six else six - timedelta(
            days=rng.randint(1, 20))
        return replace(w, received=received)

    def items(self, rng: random.Random, w: RefundWorld, n: Nar, base: dict) -> list[dict]:
        r = n.rng(5)
        customer = r.choice(["Vantorre Landscaping", "Kessel Dental", "Orrin Studio", "Bramley Bakery", "Nord Freight"])
        sent = w.received - timedelta(days=r.randint(1, 4))
        end, effective, days, remaining = self.parts(w)
        early = (f"14.3 If termination takes effect before the end of the first six months of the term, an "
                 f"early-termination fee of {money(w.early_fee, w.currency)} is deducted from the refund (the refund "
                 f"never goes below zero).\n" if w.early_fee else "")
        rules = (f"TERMS OF SERVICE, SECTION 14 (TERMINATION AND REFUNDS)\n"
                 f"14.1 The customer may terminate with {w.notice_days} calendar days' notice. Termination takes "
                 f"effect {w.notice_days} days after the day on which we receive the notice.\n"
                 f"14.2 We refund the annual fee pro rata: fee x (days of the term after the effective date) / (days "
                 f"in the term). The result is rounded to the cent, half up.\n{early}"
                 f"14.4 A term runs for twelve months from its start date; its last day is the day before the "
                 f"anniversary of the start date.")
        facts = [(w.start, f"Current term started {n.d(w.start)}; annual fee paid in advance: "
                           f"{money(w.fee, w.currency)}."),
                 (sent, f"{customer} emailed a termination notice on {n.d(sent)}."),
                 (w.received, f"The notice reached our contracts inbox on {n.d(w.received)} (date of receipt)."),
                 (None, "The customer asked for the refund to be paid to the card on file.")]
        state = f"{rules}\n\n" + render_file(n.style, f"TERMINATION, {customer.upper()}", "Account review", facts)
        gold = self.solve(w)
        wrong = [self.refund(w, term_days=731 - days), self.refund(w, extra_day=1),
                 self.refund(w, from_received=True), self.refund(w, fee="never"), self.refund(w, fee="always")]
        amounts = pick_options(rng, gold, wrong, rng.randint(4, 5))
        why = (f"Effective {effective} ({w.notice_days} days after receipt {w.received}); term {w.start} to {end} = "
               f"{days} days; {remaining} days after the effective date; {w.fee} x {remaining}/{days} = "
               f"{(w.fee * remaining / days).quantize(CENT, ROUND_HALF_UP)}"
               + (f", minus early fee {w.early_fee}" if w.early_fee and effective < add_months(w.start, 6) else "")
               + f" -> {gold}.")
        common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
        return [choice_item(rng=rng, options={money_label(a, w.currency): f"Refund {money(a, w.currency)}."
                                              for a in amounts},
                            expected=money_label(gold, w.currency), trap=money_label(amounts[1], w.currency),
                            instructions=f"What refund does {customer} receive under section 14 of the Terms of "
                                         f"Service?", **common)]


# ------------------------------------------------------------------ unit_limit

@dataclass(frozen=True)
class LoadWorld:
    net_lb: int
    tare: Decimal
    straps: int
    strap_kg: Decimal
    net_kg_restraint: Decimal     # the restraint net
    limit: int


class UnitLimit:
    name, use_case = "unit_limit", "other"

    @staticmethod
    def gross(w: LoadWorld, restraint: bool = True, factor: Decimal = LB) -> Decimal:
        g = Decimal(w.net_lb) * factor + w.tare
        if restraint:
            g += w.straps * w.strap_kg + w.net_kg_restraint
        return g

    def declared(self, w: LoadWorld, **kw) -> int:
        return math.ceil(self.gross(w, **kw))

    def solve(self, w: LoadWorld) -> bool:
        return self.declared(w) <= w.limit

    def sample(self, rng: random.Random) -> LoadWorld:
        limit = rng.choice([1500, 2500, 3200, 4600, 6800])
        tare = Decimal(rng.randint(600, 1200)) / 10
        straps, strap_kg = rng.randint(2, 8), Decimal(rng.randint(8, 16)) / 10
        net_r = Decimal(rng.randint(30, 120)) / 10
        target = limit + rng.choice([-3, -2, -1, 0, 1, 2, 3, 4])
        kg = Decimal(target) - tare - straps * strap_kg - net_r - Decimal("0.5")
        return LoadWorld(int((kg / LB).to_integral_value()), tare, straps, strap_kg, net_r, limit)

    def mutate(self, rng: random.Random, w: LoadWorld) -> LoadWorld:
        step = 1 if self.solve(w) else -1
        for k in range(1, 6):
            t = replace(w, straps=max(1, w.straps + step * k))
            if self.solve(t) != self.solve(w):
                return t
        return replace(w, straps=w.straps + step)

    def items(self, rng: random.Random, w: LoadWorld, n: Nar, base: dict) -> list[dict]:
        r = n.rng(6)
        uld = f"{r.choice(['PMC', 'PAG', 'AKE'])} {r.randint(10000, 99999)} {r.choice(['KX', 'LH', 'AF', 'QR'])}"
        pos = f"{r.randint(11, 44)}{r.choice('LRP')}"
        sheet = self.declared(w, restraint=False)
        rules = (f"LOAD RULE LR-7\n- Gross weight = net cargo weight + tare weight of the unit + weight of all restraint "
                 f"equipment (straps, nets, bars) fitted to it.\n- Convert pounds to kilograms with 1 lb = 0.45359237 kg "
                 f"exactly.\n- The declared gross weight is the gross weight rounded up to the next whole kilogram.\n"
                 f"- Position {pos} accepts a unit whose declared gross weight is at most {w.limit:,} kg.")
        facts = [(None, f"Net cargo weight per the shipper's manifest: {w.net_lb:,} lb."),
                 (None, f"Tare weight of {uld} (unit plate): {w.tare} kg."),
                 (None, f"Restraint fitted: {w.straps} straps at {w.strap_kg} kg each and one net of "
                        f"{w.net_kg_restraint} kg."),
                 (None, f"Load supervisor's worksheet: declared gross {sheet:,} kg, planned for position {pos}.")]
        state = f"{rules}\n\n" + render_file(n.style, f"LOAD CHECK, ULD {uld}", "Pre-departure", facts)
        ok, dec = self.solve(w), self.declared(w)
        why = (f"{w.net_lb} lb x 0.45359237 = {Decimal(w.net_lb) * LB:.3f} kg; + tare {w.tare} + restraint "
               f"{w.straps * w.strap_kg + w.net_kg_restraint} = {self.gross(w):.3f} kg, rounded up {dec} kg vs "
               f"limit {w.limit}: {'complies' if ok else 'over'}. The worksheet left out the restraint ({sheet}).")
        common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
        out = [noul_item(instructions=f"Does {uld} comply with load rule LR-7 for position {pos}?",
                         true_text="The declared gross weight is within the position limit.",
                         false_text="The declared gross weight exceeds the position limit.", answer=ok, **common)]
        if rng.random() < 0.5:
            wrong = [sheet, round(self.gross(w)), self.declared(w, factor=Decimal("0.45")), dec + 1, dec - 1]
            values = pick_options(rng, dec, wrong, 4)
            out.append(choice_item(rng=rng, options={f"kg_{v}": f"Declare {v:,} kg." for v in values},
                                   expected=f"kg_{dec}", trap=f"kg_{values[1]}",
                                   instructions=f"What declared gross weight must be entered for {uld}?", **common))
        return out


# ------------------------------------------------------------------ driver

TASKS = [PeriodEnd(), WorkingDays(), ThresholdDay(), RestPeriods(), ProrataRefund(), UnitLimit()]


def generate_case(seed: int) -> list[dict]:
    """One case: a base world and a twin whose single changed fact flips the main answer."""
    rng = random.Random(seed)
    task = TASKS[seed % len(TASKS)]
    w = None
    for _ in range(20):
        w = task.sample(rng)
        if w is not None:
            break
    if w is None:
        return []
    n = Nar(seed=seed, style=rng.choice(list(SPLIT_BY_STYLE)), ds=rng.choice(DATE_STYLES))
    group = f"dg-{FAMILY}-{stable_id(GENERATOR, str(seed))}"
    base = {"family": FAMILY, "generator": GENERATOR, "use_case": task.use_case, "topic": "math", "group": group,
            "style": n.style}
    items = task.items(rng, w, n, dict(base, rationale=f"[base:{task.name}]"))
    answer = task.solve(w)
    for _ in range(6):
        t = task.mutate(rng, w)
        if t is not None and task.solve(t) != answer:
            items += task.items(rng, t, n, dict(base, rationale=f"[twin:{task.name}]"))
            break
    return items
