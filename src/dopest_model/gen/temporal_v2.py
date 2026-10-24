"""Hard temporal and numeric decisions in the shape of JevBench's public hard items (family temporal_numeric), v2.

What v1 (temporal_numeric.py) lacked, taken from the structure of the public items (never their content):
  - a human note in the state that asserts a wrong answer with a plausible method (tax counted, wall clocks, a
    shift attributed to the wrong week); the model has to trust the rules, not the note
  - clock changes (summer time starts or ends inside the roster) and rules measured in real elapsed time
  - provisions marked "for information" that the question does not need
  - long tabular exports (a whole month of daily rows) and compliance questions over several sections at once
Two worlds, each with a rule engine, a renderer and a one-fact mutation for twins:
  ledger_alert  a month of daily cloud costs; which day a budget alert fires, whether a note is right, how many
                alert levels fire
  rota_dst      a roster across a clock change and two offices; rest, shift length and weekly hours
Labels come from the engine only; the state contains every rule the answer needs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal

from .common import DATE_STYLES, SPLIT_BY_STYLE, Nar, choice_item, noul_item, render_file, score_item, stable_id

FAMILY = "temporal_numeric"
GENERATOR = "temporal_v2"
CENT = Decimal("0.01")
MONTH_DAYS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def money(x: Decimal) -> str:
    return f"{x.quantize(CENT, ROUND_HALF_UP):,.2f}"


# ------------------------------------------------------------------ ledger_alert

@dataclass(frozen=True)
class LedgerWorld:
    year: int
    month: int
    budget: int
    pct: int                 # alert threshold, percent of budget
    usage: tuple             # Decimal per day
    credit_day: int          # 1-based day the credit is applied
    credit: Decimal
    tax_rate: int
    at_or_above: bool        # True: "at or above", False: "above"
    extra_levels: tuple      # other alert thresholds (percent), for the score question


class LedgerAlert:
    name, use_case = "ledger_alert", "risk_assessment"

    @staticmethod
    def fire(w: LedgerWorld, pct: int | None = None, credits: bool = True, tax: bool = False,
             at_or_above: bool | None = None) -> int | None:
        """1-based day the month-to-date net cost first reaches pct% of the budget, or None."""
        pct = w.pct if pct is None else pct
        at_or_above = w.at_or_above if at_or_above is None else at_or_above
        threshold = Decimal(w.budget) * pct / 100
        total = Decimal(0)
        for i, u in enumerate(w.usage, 1):
            total += u * (1 + Decimal(w.tax_rate) / 100) if tax else u
            if credits and i == w.credit_day:
                total -= w.credit
            if (total >= threshold) if at_or_above else (total > threshold):
                return i
        return None

    def solve(self, w: LedgerWorld) -> int | None:
        return self.fire(w)

    def sample(self, rng: random.Random) -> LedgerWorld | None:
        year, month = rng.choice([2026, 2027]), rng.randint(1, 12)
        n = MONTH_DAYS[month] + (1 if month == 2 and year % 4 == 0 else 0)
        base = rng.randint(200, 900)
        usage = [Decimal(rng.randint(base * 85, base * 115)) / 100 for _ in range(n)]
        if rng.random() < 0.5:                                   # one quiet day (weekend or outage)
            usage[rng.randint(5, n - 3)] = Decimal(rng.randint(base * 40, base * 65)) / 100
        credit = Decimal(rng.randint(3, 12)) * 100
        budget = int(sum(usage) * Decimal(rng.randint(85, 125)) / 100 / 100) * 100
        pct = rng.choice([75, 80, 85, 90])
        at_or_above = rng.random() < 0.7
        tax_rate = rng.choice([0, 5, 8, 19, 20, 23])
        w = LedgerWorld(year, month, budget, pct, tuple(usage), n + 5, credit, tax_rate, at_or_above, ())
        f0 = self.fire(w)                                        # firing day before any credit
        if f0 is None or f0 < 6 or f0 >= n - 1:
            return None
        # place the credit so that the distance between credit and alert is spread over the score levels
        credit_day = max(2, min(n - 2, f0 - rng.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13])))
        w = replace(w, credit_day=credit_day)
        f = self.fire(w)
        # the trap methods must give a DIFFERENT day, otherwise the note is not a trap
        if f is None or f >= n - 1:
            return None
        if self.fire(w, tax=True) == f and self.fire(w, credits=False) == f:
            return None
        return w

    def mutate(self, rng: random.Random, w: LedgerWorld) -> LedgerWorld | None:
        """Move the credit by a few days or change its size: the alert day moves (and still exists)."""
        if rng.random() < 0.5:
            t = replace(w, credit=max(Decimal(100), w.credit + Decimal(rng.choice([-300, -200, 200, 300, 400]))))
        else:
            t = replace(w, credit_day=max(2, min(len(w.usage) - 2, w.credit_day + rng.choice([-5, -3, 3, 5]))))
        f = self.fire(t)
        return t if f is not None and f < len(t.usage) - 1 else None

    def items(self, rng: random.Random, w: LedgerWorld, n: Nar, base: dict) -> list[dict]:
        r = n.rng(11)
        vendor = r.choice(["NimbusGrid", "Stratus Cloud", "Kestrel Compute", "Harbor Data", "Orbital Stack"])
        account = r.choice(["analytics-prod", "ml-training", "web-tier", "data-lake", "payments-batch"])
        who = r.choice(["FinOps channel", "cost analyst's note", "team lead's message", "controller's comment"])
        reason = r.choice(["migration incentive", "SLA credit", "committed-use discount true-up", "support credit"])
        first = date(w.year, w.month, 1)
        mon = f"{first:%B} {w.year}"
        rows = []
        for i, u in enumerate(w.usage, 1):
            d = date(w.year, w.month, i)
            tax = (u * w.tax_rate / 100).quantize(CENT, ROUND_HALF_UP)
            rows.append(f"{d:%d %b}: usage {money(u)}; credits {money(w.credit if i == w.credit_day else Decimal(0))};"
                        f" tax {w.tax_rate}% {money(tax)}")
        comparator = "reaches or passes" if w.at_or_above else "passes"
        rules = (f"{vendor.upper()} BILLING: SPEND ALERT SETUP AND DAY-BY-DAY CHARGES, ACCOUNT {account.upper()}, "
                 f"{mon.upper()}, USD\n\n"
                 f"Spend budget for the month: USD {w.budget:,}.00.\n"
                 f"Alert A1 triggers once, after the close of the first day on which cumulative net spend for the month "
                 f"{comparator} {w.pct}% of the budget.\n"
                 f"How net spend is computed ({vendor} billing handbook): net spend = usage charges less any credits "
                 f"applied; taxes are listed for information only and never count toward the budget; a credit counts "
                 f"from the day it is applied.\n"
                 f"Credit applied on {date(w.year, w.month, w.credit_day).day} "
                 f"{first:%B}: USD {money(w.credit)} ({reason}).\n"
                 f"Not in scope of this review: alerts A0 (50%) and A2 (100%) on the same budget.\n\n"
                 f"Day-by-day charges:\n" + "\n".join(rows))
        f = self.solve(w)
        f_tax, f_nocredit = self.fire(w, tax=True), self.fire(w, credits=False)
        trap_day = f_tax if f_tax is not None and f_tax != f else f_nocredit
        d_of = lambda i: date(w.year, w.month, i)
        note_right = r.random() < 0.4 or trap_day == f       # sometimes the human note is simply correct; and a
        if note_right:                                         # note that names the right day is never called wrong
            note = (f"{who}, {d_of(min(len(w.usage), f + 1)).day} {first:%B}: \"Net of the credit and before tax we "
                    f"passed {w.pct}% on {d_of(f).day} {first:%B}; A1 went off that evening.\"")
        elif trap_day == f_tax:
            note = (f"{who}, {d_of(min(len(w.usage), trap_day + 1)).day} {first:%B}: \"Counting tax as well, we passed {w.pct}% on "
                    f"{d_of(trap_day).day} {first:%B}; A1 went off that evening.\"")
        else:
            note = (f"{who}, {d_of(min(len(w.usage), trap_day + 1)).day} {first:%B}: \"Going by usage alone, we "
                    f"passed {w.pct}% on {d_of(trap_day).day} {first:%B}; A1 went off that evening.\"")
        state = rules + "\n\n" + render_file(n.style, "Billing thread", f"Account {account}", [(d_of(f), note)])
        why = (f"Threshold {w.pct}% of {w.budget} = {Decimal(w.budget) * w.pct / 100:.2f} of net cost (usage minus "
               f"the credit on day {w.credit_day}, no tax). First day {comparator}: day {f}. "
               f"Tax-included day {f_tax}, credit-ignored day {f_nocredit}.")
        common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
        lab = lambda i: f"{first:%b}_{i:02d}".lower()
        text = lambda i: f"At the end of {d_of(i).day} {first:%B}."
        lo = max(1, min(f, trap_day) - 1)
        days = [i for i in range(lo, lo + 5) if i <= len(w.usage)]
        if f not in days or trap_day not in days:
            days = sorted({f, trap_day, max(1, f - 1), min(len(w.usage), f + 1), min(len(w.usage), f + 2)})
        out = [choice_item(rng=rng, options={lab(i): text(i) for i in days}, expected=lab(f), trap=lab(trap_day),
                           instructions="After the close of which day does alert A1 trigger?", **common)]
        kind = r.random()
        if kind < 0.5:
            out.append(noul_item(instructions=f"Does the {who} name the right day for alert A1?",
                                 true_text="The note names the day A1 actually fired under the stated rule and "
                                           "definitions.", false_text="The note names a different day than the rule "
                                                                     "gives.", answer=note_right, **common))
        else:
            cut = r.choice([f - 1, f, f + 1])
            out.append(noul_item(instructions=f"Has alert A1 triggered by the close of {d_of(cut).day} {first:%B} "
                                              f"(inclusive)?", true_text="A1 fired on or before that day.",
                                 false_text="A1 had not fired by the end of that day.", answer=f <= cut, **common))
        gap = f - w.credit_day
        level = 0 if gap <= 0 else 1 if gap <= 3 else 2 if gap <= 7 else 3
        out.append(score_item(instructions="How long after the credit is applied does A1 trigger? Output the level index.",
                              levels=["On the credit day or earlier", "1 to 3 days after the credit",
                                      "4 to 7 days after the credit", "More than 7 days after the credit"],
                              expected=level, **common))
        return out


# ------------------------------------------------------------------ rota_dst

@dataclass(frozen=True)
class Zone:
    name: str                # e.g. Europe/Berlin
    abbr_before: str
    abbr_after: str
    off_before: int          # UTC offset before the change
    off_after: int
    change: datetime         # local wall-clock instant of the change, expressed in BEFORE time
    # at `change` (before-time), clocks are set to change + (off_after - off_before)


ZONES = [
    ("Europe/Berlin", "CET", "CEST", 1, 2), ("Europe/Warsaw", "CET", "CEST", 1, 2),
    ("Europe/London", "GMT", "BST", 0, 1), ("America/New_York", "EST", "EDT", -5, -4),
    ("America/Chicago", "CST", "CDT", -6, -5), ("Australia/Sydney", "AEST", "AEDT", 10, 11),
]


def make_zone(rng: random.Random, year: int, spring: bool) -> Zone:
    name, std, dst, off_std, off_dst = rng.choice(ZONES)
    if name.startswith("Australia"):
        # southern hemisphere: summer time starts in October, ends in April
        month = 10 if spring else 4
    else:
        month = 3 if spring else 10
    # change on a Sunday of that month at 02:00 (before-time)
    d = date(year, month, rng.choice([8, 15, 22, 25]))
    d -= timedelta(days=(d.weekday() + 1) % 7)          # back to Sunday
    if spring:
        return Zone(name, std, dst, off_std, off_dst, datetime.combine(d, datetime.min.time()) + timedelta(hours=2))
    return Zone(name, dst, std, off_dst, off_std, datetime.combine(d, datetime.min.time()) + timedelta(hours=3))


def to_utc(zone: Zone, local: datetime) -> datetime:
    """Local wall-clock -> UTC. Times inside the skipped or repeated hour are never generated."""
    after_start = zone.change + timedelta(hours=zone.off_after - zone.off_before)   # first instant in after-time
    if zone.off_after > zone.off_before:                 # spring forward: before-time up to `change`
        off = zone.off_before if local < zone.change else zone.off_after
    else:                                                # fall back: after-time from the repeated hour on
        off = zone.off_before if local < zone.change else zone.off_after
        if after_start <= local < zone.change:           # ambiguous hour: treat as before-time (not generated)
            off = zone.off_before
    return local - timedelta(hours=off)


@dataclass(frozen=True)
class RotaWorld:
    zone: Zone
    home_offset: int                 # offset of the home office for the rota week (fixed in this world)
    shifts: tuple                    # (start_local, end_local) in the roster zone, chronological
    min_rest: int
    max_shift: int
    weekly_max: int
    week_start: date                 # Monday of the rota week, home-office time


class RotaDst:
    name, use_case = "rota_dst", "legal_compliance"

    def utc(self, w: RotaWorld):
        return [(to_utc(w.zone, s), to_utc(w.zone, e)) for s, e in w.shifts]

    def rests(self, w: RotaWorld, naive: bool = False) -> list[float]:
        pairs = [(s, e) for s, e in w.shifts] if naive else self.utc(w)
        return [(pairs[i + 1][0] - pairs[i][1]).total_seconds() / 3600 for i in range(len(pairs) - 1)]

    def lengths(self, w: RotaWorld, naive: bool = False) -> list[float]:
        pairs = [(s, e) for s, e in w.shifts] if naive else self.utc(w)
        return [(e - s).total_seconds() / 3600 for s, e in pairs]

    def week_hours(self, w: RotaWorld) -> float:
        """Hours of shifts that START inside the rota week (home-office time), real elapsed."""
        start = datetime.combine(w.week_start, datetime.min.time()) - timedelta(hours=w.home_offset)   # UTC
        end = start + timedelta(days=7)
        return sum((e - s).total_seconds() / 3600 for s, e in self.utc(w) if start <= s < end)

    def breaches(self, w: RotaWorld) -> dict:
        return {"rest": sum(x < w.min_rest for x in self.rests(w)),
                "shift": sum(x > w.max_shift for x in self.lengths(w)),
                "week": self.week_hours(w) > w.weekly_max}

    def solve(self, w: RotaWorld) -> int:
        b = self.breaches(w)
        return min(3, b["rest"] + b["shift"] + int(b["week"]))

    def sample(self, rng: random.Random) -> RotaWorld | None:
        year = rng.choice([2026, 2027])
        zone = make_zone(rng, year, spring=rng.random() < 0.5)
        change_day = zone.change.date()
        week_start = change_day - timedelta(days=6 + rng.choice([0, 0, 7]))    # Monday before the change
        n = rng.randint(4, 6)
        t = datetime.combine(week_start, datetime.min.time()) + timedelta(hours=rng.choice([6, 7, 8, 13, 14, 22]))
        shifts = []
        p_short = rng.choice([0.0, 0.0, 0.2, 0.35])
        p_long = rng.choice([0.0, 0.0, 0.0, 0.25])
        for i in range(n):
            length = rng.choice([10.5, 11, 12.5]) if rng.random() < p_long else rng.choice([6, 7, 7.5, 8, 8, 8.5, 9, 9.5])
            s, e = t, t + timedelta(hours=length)
            if self._touches_gap(zone, s) or self._touches_gap(zone, e):
                t += timedelta(hours=3)
                s, e = t, t + timedelta(hours=length)
            shifts.append((s, e))
            rest = rng.choice([9, 10, 10.5]) if rng.random() < p_short else rng.choice([11, 11, 12, 13, 14, 15, 24, 36])
            t = e + timedelta(hours=rest)
        # a rest that straddles the clock change: wall clock lies by one hour
        w = RotaWorld(zone, rng.choice([zone.off_before, zone.off_before, -5, 0, 8]), tuple(shifts), 11,
                      rng.choice([10, 12]), rng.choice([44, 48, 48, 52]), week_start)
        rests_utc, rests_wall = self.rests(w), self.rests(w, naive=True)
        if not any(abs(a - b) > 0.5 for a, b in zip(rests_utc, rests_wall)):
            return None                                   # the change did not touch any rest: no trap, discard
        if any(self._touches_gap(zone, x) for s, e in shifts for x in (s, e)):
            return None
        return w

    @staticmethod
    def _touches_gap(zone: Zone, local: datetime) -> bool:
        """Inside the repeated or skipped hour around the change."""
        a = min(zone.change, zone.change + timedelta(hours=zone.off_after - zone.off_before))
        b = max(zone.change, zone.change + timedelta(hours=zone.off_after - zone.off_before))
        return a <= local < b

    def mutate(self, rng: random.Random, w: RotaWorld) -> RotaWorld | None:
        i = rng.randrange(1, len(w.shifts))
        s, e = w.shifts[i]
        rest = self.rests(w)[i - 1]
        delta = (w.min_rest - rest + rng.choice([0, 0.5])) if rest < w.min_rest else -(rest - w.min_rest + rng.choice([0.5, 1]))
        ns = s + timedelta(hours=delta)
        if (e - ns).total_seconds() < 4 * 3600 or self._touches_gap(w.zone, ns):
            return None
        shifts = list(w.shifts)
        shifts[i] = (ns, e)
        return replace(w, shifts=tuple(shifts))

    def items(self, rng: random.Random, w: RotaWorld, n: Nar, base: dict) -> list[dict]:
        r = n.rng(12)
        worker = f"{r.choice(['T. Volkov', 'M. Okafor', 'L. Brennan', 'S. Ishikawa', 'A. Duarte', 'K. Nowicka'])}"
        employer = r.choice(["Regional Ambulance Service", "Kestrel Payments", "Harbour Freight Control",
                             "Northline Rail Operations", "Civic Water Utility"])
        z = w.zone
        fwd = z.off_after > z.off_before
        new_clock = z.change + timedelta(hours=z.off_after - z.off_before)
        change_txt = (f"Daylight-saving switch: on {z.change:%A %d %B %Y} at {z.change:%H:%M} local {z.abbr_before} "
                      f"(UTC{z.off_before:+d}) the clocks move {'forward' if fwd else 'back'} to {new_clock:%H:%M} local "
                      f"{z.abbr_after} (UTC{z.off_after:+d}). Roster times are local wall-clock times.")
        lines = []
        for k, (s, e) in enumerate(w.shifts, 1):
            same_day = s.date() == e.date()
            lines.append((s.date(), f"Duty {k}: {s:%a %d %b}, {s:%H:%M} to {'' if same_day else e.strftime('%a %d %b ')}{e:%H:%M}"))
        rules = (f"{employer.upper()} : DUTY TIME CODE DT-7 AND ROSTER PRINTOUT FOR {worker.upper()}, ZONE {z.name}\n\n"
                 f"DT-7.1 Rest between duties: from the end of one duty to the start of the next, at least "
                 f"{w.min_rest} hours must elapse in real time; exactly {w.min_rest} hours is compliant; every shorter "
                 f"gap is one breach.\n"
                 f"DT-7.2 Longest duty: {w.max_shift} hours of real elapsed time; every longer duty is one breach.\n"
                 f"DT-7.3 Hours cap per roster week: {w.weekly_max} hours. The roster week runs Monday 00:00 to Sunday "
                 f"24:00 in the base office's clock (UTC{w.home_offset:+d}); a duty counts in the week in which it "
                 f"begins; exceeding the cap is one breach.\n{change_txt}\n"
                 f"Also in DT-7 but outside this check: the night premium for hours between 23:00 and 06:00 "
                 f"local, and the 24-hour weekly rest, which is audited quarterly.\n"
                 f"Roster week examined: Monday {w.week_start:%d %B %Y}.\n\nRoster (local times):")
        rests_utc, rests_wall = self.rests(w), self.rests(w, naive=True)
        wall_short = [k for k, x in enumerate(rests_wall, 1) if x < w.min_rest]
        real_short = [k for k, x in enumerate(rests_utc, 1) if x < w.min_rest]
        note_right = r.random() < 0.4                        # sometimes the rep converts the clock change correctly
        listed = real_short if note_right else wall_short
        note_list = ", ".join(f"after duty {k}" for k in listed) or "no short gaps"
        method = "Allowing for the clock switch in real time," if note_right else "Going by the roster clock,"
        note = (f"Staff delegate's comment: \"{method} I make it {len(listed)} short gap(s): {note_list}. "
                f"{'That is ' + str(len(listed)) + ' breach(es) of 7.1.' if listed else 'So nothing under 7.1.'}\"")
        state = (rules + "\n" + "\n".join(t for _, t in lines) + "\n\n"
                 + render_file(n.style, "Audit notes", f"Roster week of {w.week_start:%d %b}", [(w.week_start, note)]))
        b = self.breaches(w)
        total = self.solve(w)
        why = (f"Real rests (h): {', '.join(f'{x:g}' for x in rests_utc)}; wall-clock rests: "
               f"{', '.join(f'{x:g}' for x in rests_wall)}. Short real rests: {real_short}. Shift lengths: "
               f"{', '.join(f'{x:g}' for x in self.lengths(w))} (max {w.max_shift}). Week hours (home time, by start): "
               f"{self.week_hours(w):g} of {w.weekly_max}. Violations: rest {b['rest']}, shift {b['shift']}, "
               f"week {int(b['week'])}.")
        common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
        out = [score_item(instructions=f"Counting DT-7.1, 7.2 and 7.3 together, how many breaches does "
                                       f"{worker}'s roster contain? Output the level index.",
                          levels=["No breaches", "Exactly one breach", "Exactly two breaches",
                                  "Three or more breaches"], expected=total, **common),
               noul_item(instructions=f"Does {worker}'s roster, as printed, satisfy DT-7.1 (rest between duties)?",
                         true_text=f"Every gap between consecutive duties is at least {w.min_rest} hours of real "
                                   f"elapsed time.", false_text="At least one rest is shorter than required.",
                         answer=b["rest"] == 0, **common)]
        if note_right or real_short != wall_short:
            out.append(noul_item(instructions="Is the staff delegate's count of short gaps under DT-7.1 right?",
                                 true_text="The note lists exactly the rests that are too short in actual elapsed "
                                           "time.", false_text="The note is wrong: it counts by wall clock across the "
                                                              "clock change.", answer=note_right, **common))
        sections = {"none": "No section is breached.", "2_1": "Only DT-7.1 (rest between duties).",
                    "2_2": "Only DT-7.2 (duty length).", "2_3": "Only DT-7.3 (weekly cap).",
                    "several": "More than one section is breached."}
        hit = [k for k, v in (("2_1", b["rest"]), ("2_2", b["shift"]), ("2_3", b["week"])) if v]
        gold = "none" if not hit else hit[0] if len(hit) == 1 else "several"
        trap = "2_1" if (wall_short and not real_short) else ("none" if real_short and not wall_short else None)
        if trap == gold:
            trap = None
        out.append(choice_item(rng=rng, options=sections, expected=gold, trap=trap,
                               instructions="Which DT-7 sections does the roster breach?", **common))
        return out


TASKS = [LedgerAlert(), RotaDst()]


def generate_case(seed: int) -> list[dict]:
    """One case: a base world and a twin whose single changed fact flips the main answer."""
    rng = random.Random(seed)
    task = TASKS[seed % len(TASKS)]
    w = None
    for _ in range(30):
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
    for _ in range(8):
        t = task.mutate(rng, w)
        if t is not None and task.solve(t) != answer:
            items += task.items(rng, t, n, dict(base, rationale=f"[twin:{task.name}]"))
            break
    return items
