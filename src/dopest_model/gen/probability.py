"""Probabilities that follow exactly from the state (JevBench hard tier family probability; OpenDecisions
calibration_gen). Items carry target_probs (the exact distribution) and expected (its most likely option).

Five worlds, each with an exact engine (fractions), a renderer and a one-fact twin:
  reference_class  outcome shares among comparable past cases (same attributes, after a cutoff, closed only)
  bayes            posterior of a flag from the segment's base rate, the hit rate and the false-alarm rate
  sampling         lot inspection without replacement under the sampling plan in force on the inspection date
  connection       missing a connection, where a late outbound can rescue a moderately late inbound
  urn              two-stage draw: a die picks the box, then one ticket is drawn from it

JevBench's Capability scores calibration against the expected option (the distance to target_probs is reported
separately), so whether to train on hard or soft targets is a training ablation, not a property of the data.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from datetime import date, timedelta
from fractions import Fraction

from .common import DATE_STYLES, SPLIT_BY_STYLE, Nar, choice_item, noul_item, render_file, score_item, stable_id

FAMILY = "probability"
GENERATOR = "probability_v1"
MARGIN = Fraction(6, 100)          # |p - 1/2| for yes/no, top-two gap for choice and score
ASK = "Give probabilities that reflect the evidence in the state."


def top(dist: dict) -> str:
    return max(dist, key=lambda k: dist[k])


def clear(dist: dict) -> bool:
    """A single most likely option, ahead of the runner-up by at least MARGIN."""
    v = sorted(dist.values(), reverse=True)
    return len(v) < 2 or v[0] - v[1] >= MARGIN


def floats(dist: dict) -> dict:
    return {k: float(v) for k, v in dist.items()}


def pct(x: Fraction) -> str:
    return f"{float(x) * 100:g}%"


# ------------------------------------------------------------------ reference_class

DOMAINS = [
    dict(org="LUMINARC SOFTWARE, SUPPORT OPERATIONS", entity="ticket", prefix="T", a="tier",
         a_vals=["Enterprise", "Standard", "Starter"], b="category", b_vals=["SSO login", "billing", "API errors"],
         when="opened", event="identity connector release 5.2 was deployed", use_case="customer_support",
         outcomes={"resolved_first_contact": "The ticket is resolved at first contact.",
                   "escalated": "The ticket is escalated to engineering.",
                   "reopened": "The ticket is reopened after it was closed."}),
    dict(org="OSTERLAND GARDEN SUPPLY, INBOUND LOGISTICS", entity="shipment", prefix="SH", a="carrier",
         a_vals=["Brisa Freight", "Nordline Cargo", "Kestrel Haulage"], b="service", b_vals=["Express", "Economy"],
         when="picked up", event="the Porto hub moved to its new sorting centre", use_case="other",
         outcomes={"early": "Delivered before the promised day.", "on_time": "Delivered on the promised day.",
                   "late": "Delivered after the promised day."}),
    dict(org="VOLTARIA ONLINE STORE, CHARGEBACK DESK", entity="dispute", prefix="CB", a="product line",
         a_vals=["Electronics", "Apparel", "Home"], b="reason", b_vals=["item not received", "not as described"],
         when="filed", event="the card network's new evidence rules took effect", use_case="financial_crime",
         outcomes={"merchant_wins": "The merchant wins the dispute.", "customer_wins": "The customer wins the dispute.",
                   "customer_withdraws": "The customer withdraws the dispute."}),
    dict(org="ALDERWAY INSURANCE, CLAIMS ANALYTICS", entity="claim", prefix="CL", a="line",
         a_vals=["travel", "home", "motor"], b="region", b_vals=["north", "south", "west"], when="opened",
         event="the new claims-handling guideline took effect", use_case="insurance_claims",
         outcomes={"paid_in_full": "The claim is paid in full.", "paid_partially": "The claim is paid in part.",
                   "denied": "The claim is denied."}),
]


@dataclass(frozen=True)
class RefWorld:
    domain: int
    rows: tuple               # (id, a, b, date, closed, outcome or None)
    cutoff: date
    new_a: str
    new_b: str
    new_date: date
    twin_a: str               # the attribute value whose comparable set has a different most likely outcome


class ReferenceClass:
    name = "reference_class"

    @staticmethod
    def comparable(w: RefWorld, a: str | None = None) -> list:
        a = a or w.new_a
        return [r for r in w.rows if r[1] == a and r[2] == w.new_b and r[3] > w.cutoff and r[4]]

    def solve(self, w: RefWorld) -> dict:
        rows = self.comparable(w)
        outcomes = DOMAINS[w.domain]["outcomes"]
        return {o: Fraction(sum(r[5] == o for r in rows), len(rows)) for o in outcomes}

    @staticmethod
    def counts(rng: random.Random, outcomes: list, n: int, lead: str) -> list:
        """n outcomes whose most likely value is `lead`, clearly ahead."""
        for _ in range(200):
            w = {o: rng.random() + (1.2 if o == lead else 0) for o in outcomes}
            draw = rng.choices(list(w), weights=list(w.values()), k=n)
            dist = {o: Fraction(draw.count(o), n) for o in outcomes}
            if top(dist) == lead and clear(dist):
                return draw
        return [lead] * n

    def sample(self, rng: random.Random) -> RefWorld:
        dom_i = rng.randrange(len(DOMAINS))
        dom = DOMAINS[dom_i]
        outcomes = list(dom["outcomes"])
        a0, a1 = rng.sample(dom["a_vals"], 2)
        b0 = rng.choice(dom["b_vals"])
        cutoff = date(2026, rng.randint(2, 8), rng.randint(1, 28))
        lead0, lead1 = rng.sample(outcomes, 2)
        rows = []
        after = lambda: cutoff + timedelta(days=rng.randint(1, 110))
        before = lambda: cutoff - timedelta(days=rng.randint(0, 200))   # 0: on the cutoff day, not after it
        for o in self.counts(rng, outcomes, rng.randint(10, 22), lead0):
            rows.append((a0, b0, after(), True, o))
        for o in self.counts(rng, outcomes, rng.randint(10, 22), lead1):
            rows.append((a1, b0, after(), True, o))
        old_lead = rng.choice([o for o in outcomes if o != lead0])
        for o in self.counts(rng, outcomes, rng.randint(6, 14), old_lead):       # same attributes, before the cutoff
            rows.append((a0, b0, before(), True, o))
        for _ in range(rng.randint(1, 3)):                                     # open cases: no outcome yet
            rows.append((a0, b0, after(), False, None))
        for _ in range(rng.randint(8, 16)):
            a, b = rng.choice(dom["a_vals"]), rng.choice(dom["b_vals"])
            if (a, b) in ((a0, b0), (a1, b0)):
                continue
            rows.append((a, b, rng.choice([after(), before()]), True, rng.choice(outcomes)))
        rows.sort(key=lambda r: r[2])
        ids = rng.sample(range(10000, 99999), len(rows))
        rows = tuple((f"{dom['prefix']}-{i}",) + r for i, r in zip(ids, rows))
        return RefWorld(dom_i, rows, cutoff, a0, b0, cutoff + timedelta(days=rng.randint(115, 140)), a1)

    def mutate(self, rng: random.Random, w: RefWorld) -> RefWorld:
        return replace(w, new_a=w.twin_a, twin_a=w.new_a)

    def items(self, rng: random.Random, w: RefWorld, n: Nar, base: dict) -> list[dict]:
        dom = DOMAINS[w.domain]
        r = n.rng(11)
        new_id = f"{dom['prefix']}-{r.randint(100000, 999999)}"
        head = f"| id | {dom['a']} | {dom['b']} | {dom['when']} | status | outcome |\n|---|---|---|---|---|---|"
        table = "\n".join(f"| {i} | {a} | {b} | {d.isoformat()} | {'closed' if c else 'open'} | {o or '-'} |"
                          for i, a, b, d, c, o in w.rows)
        rules = (f"{dom['org']}\nFORECAST RULE: estimate the outcome of a new {dom['entity']} from past "
                 f"{dom['entity']}s with the same {dom['a']} and the same {dom['b']} that were {dom['when']} after "
                 f"{n.d(w.cutoff)}, when {dom['event']}; earlier {dom['entity']}s are not comparable. Only closed "
                 f"{dom['entity']}s count: open ones have no outcome yet. The forecast is the share of each outcome "
                 f"among the comparable {dom['entity']}s.\n\nPAST {dom['entity'].upper()}S (export)\n{head}\n{table}")
        lead = r.choice(["team lead", "analyst", "shift supervisor"])
        loose = [x for x in w.rows if x[2] == w.new_b and x[4] and x[3] > w.cutoff]          # ignores the first attribute
        note = ""
        if loose:
            o = top({k: sum(x[5] == k for x in loose) for k in dom["outcomes"]})
            note = (f"The {lead} noted that {sum(x[5] == o for x in loose)} of {len(loose)} recent {w.new_b} "
                    f"{dom['entity']}s ended as {o.replace('_', ' ')}.")
        facts = [(w.new_date, f"New {dom['entity']} {new_id}: {dom['a']} {w.new_a}, {dom['b']} {w.new_b}, "
                              f"{dom['when']} {n.d(w.new_date)}.")]
        if note:
            facts.append((None, note))
        state = f"{rules}\n\n" + render_file(n.style, f"FORECAST REQUEST {new_id}", dom["org"].title(), facts)
        dist = self.solve(w)
        rows = self.comparable(w)
        why = (f"Comparable: {dom['a']} {w.new_a}, {dom['b']} {w.new_b}, {dom['when']} after {w.cutoff}, closed: "
               f"{len(rows)}; " + ", ".join(f"{k} {v}" for k, v in dist.items()) + ".")
        common = dict(base, use_case=dom["use_case"], state=state, rationale=f"{base['rationale']} {why}",
                      target_probs=floats(dist))
        return [choice_item(rng=rng, options=dict(dom["outcomes"]), expected=top(dist), trap=None,
                            instructions=f"What will the outcome of {dom['entity']} {new_id} be? {ASK}", **common)]


# ------------------------------------------------------------------ bayes

BAYES = [
    dict(org="PAYMENTS RISK TEAM", thing="transaction", flag="flagged by rule R-17", short="rule R-17",
         pos="fraudulent", neg="legitimate", seg="merchant category", use_case="financial_crime",
         segs=["travel", "electronics", "groceries", "fuel", "gift cards"]),
    dict(org="PLANT QUALITY ASSURANCE", thing="unit", flag="flagged by the acoustic end-of-line sensor",
         short="the acoustic sensor", pos="defective", neg="sound", seg="production line", use_case="other",
         segs=["line 1", "line 2", "line 3", "line 4"]),
    dict(org="SECURITY OPERATIONS", thing="login", flag="raised by the impossible-travel alert",
         short="the impossible-travel alert", pos="an account takeover", neg="legitimate", seg="user group",
         use_case="risk_assessment", segs=["contractors", "employees", "service accounts", "executives"]),
    dict(org="MAIL SECURITY", thing="email", flag="marked by the phishing classifier", short="the classifier",
         pos="phishing", neg="legitimate", seg="inbox", use_case="llm_guardrails",
         segs=["finance inbox", "support inbox", "HR inbox", "sales inbox"]),
]


@dataclass(frozen=True)
class BayesWorld:
    domain: int
    rates: tuple              # (segment, base rate as Fraction)
    hit: Fraction
    false_alarm: Fraction
    segment: str
    twin_segment: str


class Bayes:
    name = "bayes"

    @staticmethod
    def posterior(w: BayesWorld, segment: str | None = None) -> Fraction:
        b = dict(w.rates)[segment or w.segment]
        return b * w.hit / (b * w.hit + (1 - b) * w.false_alarm)

    def solve(self, w: BayesWorld) -> dict:
        p = self.posterior(w)
        return {"yes": p, "no": 1 - p}

    def sample(self, rng: random.Random) -> BayesWorld | None:
        dom_i = rng.randrange(len(BAYES))
        dom = BAYES[dom_i]
        hit = Fraction(rng.choice([80, 85, 90, 95, 98, 99]), 100)
        fa = Fraction(rng.choice([5, 10, 20, 30, 50, 80]), 1000)
        seg0, seg1 = rng.sample(dom["segs"], 2)
        rates = {}
        for seg, p in ((seg0, Fraction(rng.randint(12, 42), 100)), (seg1, Fraction(rng.randint(58, 90), 100))):
            b = p * fa / (hit * (1 - p) + p * fa)
            rates[seg] = Fraction(max(1, round(b * 1000)), 1000)        # stated to 0.1%
        for seg in dom["segs"]:
            rates.setdefault(seg, Fraction(rng.randint(1, 200), 1000))
        if rng.random() < 0.5:
            seg0, seg1 = seg1, seg0
        w = BayesWorld(dom_i, tuple(rates.items()), hit, fa, seg0, seg1)
        ok = all(abs(self.posterior(w, s) - Fraction(1, 2)) >= MARGIN for s in (seg0, seg1))
        return w if ok and (self.posterior(w, seg0) > Fraction(1, 2)) != (self.posterior(w, seg1) > Fraction(1, 2)) \
            else None

    def mutate(self, rng: random.Random, w: BayesWorld) -> BayesWorld:
        return replace(w, segment=w.twin_segment, twin_segment=w.segment)

    def items(self, rng: random.Random, w: BayesWorld, n: Nar, base: dict) -> list[dict]:
        dom = BAYES[w.domain]
        r = n.rng(12)
        ident = f"{dom['thing'][:2].upper()}-{r.randint(100000, 999999)}"
        rates = "; ".join(f"{s}: {pct(b)}" for s, b in w.rates)
        rules = (f"{dom['org']}: REFERENCE FIGURES (last 12 months, verified)\n"
                 f"- Share of {dom['thing']}s that are {dom['pos']}, by {dom['seg']}: {rates}.\n"
                 f"- {dom['short'][0].upper() + dom['short'][1:]} flags {pct(w.hit)} of {dom['pos']} {dom['thing']}s "
                 f"and {pct(w.false_alarm)} of {dom['neg']} ones.")
        facts = [(None, f"{dom['thing'].capitalize()} {ident} ({dom['seg']}: {w.segment}) was {dom['flag']}.")]
        if r.random() < 0.6:
            facts.append((None, f"An analyst commented: {dom['short']} catches {pct(w.hit)} of {dom['pos']} "
                                f"{dom['thing']}s, so this one is about {pct(w.hit)} likely to be {dom['pos']}."))
        state = f"{rules}\n\n" + render_file(n.style, f"ALERT {ident}", dom["org"].title(), facts)
        p = self.posterior(w)
        why = (f"Base rate {dict(w.rates)[w.segment]} for {w.segment}; P = b*{w.hit} / (b*{w.hit} + (1-b)*"
               f"{w.false_alarm}) = {p} = {float(p):.4f}.")
        common = dict(base, use_case=dom["use_case"], state=state, rationale=f"{base['rationale']} {why}",
                      target_probs=floats(self.solve(w)))
        return [noul_item(instructions=f"Is {dom['thing']} {ident} {dom['pos']}? {ASK}",
                          true_text=f"The {dom['thing']} is {dom['pos']}.",
                          false_text=f"The {dom['thing']} is {dom['neg']}.", answer=p > Fraction(1, 2), **common)]


# ------------------------------------------------------------------ sampling

@dataclass(frozen=True)
class LotWorld:
    units: int
    defective: int
    k_old: int
    k_new: int
    effective: date
    inspection: date
    ask_count: bool           # score question (how many defective drawn) instead of yes/no (any defective)


class Sampling:
    name = "sampling"

    @staticmethod
    def k(w: LotWorld) -> int:
        return w.k_new if w.inspection >= w.effective else w.k_old

    def levels(self, w: LotWorld) -> dict:
        n, d, k = w.units, w.defective, self.k(w)
        pmf = [Fraction(math.comb(d, x) * math.comb(n - d, k - x), math.comb(n, k)) for x in range(k + 1)]
        return {"0": pmf[0], "1": sum(pmf[1:2], Fraction(0)), "2": sum(pmf[2:3], Fraction(0)),
                "3": sum(pmf[3:], Fraction(0))}

    def solve(self, w: LotWorld) -> dict:
        lv = self.levels(w)
        return lv if w.ask_count else {"yes": 1 - lv["0"], "no": lv["0"]}

    def ok(self, w: LotWorld) -> bool:
        dist = self.solve(w)
        return clear(dist) if w.ask_count else abs(dist["yes"] - Fraction(1, 2)) >= MARGIN

    def sample(self, rng: random.Random) -> LotWorld | None:
        units = rng.randint(8, 30)
        w = LotWorld(units, rng.randint(1, units // 2), rng.randint(1, 4), rng.randint(2, 6),
                     date(2026, rng.randint(3, 10), rng.randint(1, 28)), date(2026, 1, 1), rng.random() < 0.4)
        if w.k_old == w.k_new or w.k_new >= units:
            return None
        w = replace(w, inspection=w.effective + timedelta(days=rng.choice([-1, 1]) * rng.randint(0, 25)))
        twin = self.mutate(rng, w)
        same = top(self.solve(w)) == top(self.solve(twin))
        return w if self.ok(w) and self.ok(twin) and not same else None

    def mutate(self, rng: random.Random, w: LotWorld) -> LotWorld:
        if w.inspection >= w.effective:
            return replace(w, inspection=w.effective - timedelta(days=1 + (w.inspection - w.effective).days))
        return replace(w, inspection=w.effective + (w.effective - w.inspection) - timedelta(days=1))

    def items(self, rng: random.Random, w: LotWorld, n: Nar, base: dict) -> list[dict]:
        r = n.rng(13)
        lot = f"LOT-{r.randint(1000, 9999)}"
        org = r.choice(["Korridan Electronics", "Calloway Robotics", "Brightwell Freight", "Ostrava Pumps"])
        rules = (f"{org.upper()}: INCOMING INSPECTION\n"
                 f"- Sampling plan SP-4, version 2: draw {w.k_old} unit{'s' if w.k_old > 1 else ''} at random "
                 f"without replacement and test each; reject the lot if any tested unit is defective.\n"
                 f"- SP-4 version 3 applies to inspections on or after {n.d(w.effective)}: draw {w.k_new} units "
                 f"at random without replacement; otherwise as version 2.\n- Tests are exact.")
        facts = [(None, f"Lot {lot} has {w.units} units. The supplier's verified test found exactly {w.defective} "
                        f"defective unit{'s' if w.defective > 1 else ''}; defective units are unmarked."),
                 (w.inspection, f"Inspection of {lot} is scheduled for {n.d(w.inspection)}.")]
        state = f"{rules}\n\n" + render_file(n.style, f"INSPECTION {lot}", org, facts)
        dist = self.solve(w)
        why = (f"Inspection {w.inspection} {'on or after' if w.inspection >= w.effective else 'before'} "
               f"{w.effective}: draw {self.k(w)} of {w.units} with {w.defective} defective; "
               + ", ".join(f"{k} {v}" for k, v in dist.items()) + ".")
        common = dict(base, use_case="other", state=state, rationale=f"{base['rationale']} {why}",
                      target_probs=floats(dist))
        if w.ask_count:
            return [score_item(instructions=f"How many defective units will the inspection sample of {lot} contain? "
                                            f"Output the level index. {ASK}",
                               levels=["No defective unit drawn", "Exactly one", "Exactly two", "Three or more"],
                               expected=int(top(dist)), **common)]
        return [noul_item(instructions=f"Will the inspection sample contain at least one defective unit, so that "
                                       f"{lot} is rejected? {ASK}",
                          true_text="At least one tested unit is defective.", false_text="No tested unit is defective.",
                          answer=dist["yes"] > Fraction(1, 2), **common)]


# ------------------------------------------------------------------ connection

@dataclass(frozen=True)
class ConnWorld:
    days: int
    within: int               # inbound at most `x` minutes late (or early)
    middle: int               # inbound x+1 .. y minutes late
    beyond: int               # inbound more than y minutes late
    x: int
    y: int
    out_days: int
    out_late: int             # outbound at least 20 minutes late


class Connection:
    name = "connection"

    @staticmethod
    def make(w: ConnWorld) -> Fraction:
        return Fraction(w.within, w.days) + Fraction(w.middle, w.days) * Fraction(w.out_late, w.out_days)

    def solve(self, w: ConnWorld) -> dict:
        m = self.make(w)
        return {"yes": 1 - m, "no": m}

    def sample(self, rng: random.Random) -> ConnWorld | None:
        # P(miss) runs from beyond/days (outbound always late) to (middle + beyond)/days (never late): these
        # ranges let the outbound's lateness alone move the answer across 1/2 with margin, which the twin needs.
        days = rng.choice([40, 48, 50, 60])
        middle = rng.randint(int(0.30 * days), int(0.45 * days))
        beyond = rng.randint(int(0.12 * days), int(0.30 * days))
        if days - middle - beyond < 5:
            return None
        w = ConnWorld(days, days - middle - beyond, middle, beyond, rng.choice([25, 30, 40]), rng.choice([55, 60, 70]),
                      rng.choice([30, 40, 50]), 0)
        target_miss = Fraction(rng.choice([1, -1]), 1)
        for late in rng.sample(range(0, w.out_days + 1), w.out_days + 1):
            t = replace(w, out_late=late)
            miss = 1 - self.make(t)
            if abs(miss - Fraction(1, 2)) >= MARGIN and (miss > Fraction(1, 2)) == (target_miss > 0):
                twin = self.mutate(rng, t)
                if twin and (1 - self.make(twin) > Fraction(1, 2)) != (miss > Fraction(1, 2)):
                    return t
        return None

    def mutate(self, rng: random.Random, w: ConnWorld) -> ConnWorld | None:
        side = 1 - self.make(w) > Fraction(1, 2)
        for late in sorted(range(w.out_days + 1), key=lambda v: abs(v - w.out_late)):
            t = replace(w, out_late=late)
            miss = 1 - self.make(t)
            if (miss > Fraction(1, 2)) != side and abs(miss - Fraction(1, 2)) >= MARGIN:
                return t
        return None

    def items(self, rng: random.Random, w: ConnWorld, n: Nar, base: dict) -> list[dict]:
        r = n.rng(14)
        f_in, f_out = f"SK {r.randint(1000, 1999)}", f"SK {r.randint(2000, 2999)}"
        pax = f"{r.choice(['N.', 'A.', 'M.', 'J.'])} {r.choice(['Abernathy', 'Okonkwo', 'Lindgren', 'Ferraro'])}"
        rules = (f"TRAVEL DESK: CONNECTION RISK\n"
                 f"- Inbound {f_in}, last {w.days} days: at most {w.x} minutes late (or early) {w.within} times; "
                 f"{w.x + 1} to {w.y} minutes late {w.middle} times; more than {w.y} minutes late {w.beyond} times.\n"
                 f"- Outbound {f_out}, last {w.out_days} days: departed at least 20 minutes late on {w.out_late} days.\n"
                 f"- Connection rule: a passenger makes the connection if the inbound is at most {w.x} minutes late. "
                 f"If the outbound departs at least 20 minutes late, a passenger also makes it when the inbound is up "
                 f"to {w.y} minutes late. Inbound and outbound delays are independent. Use the frequencies above as "
                 f"probabilities.")
        naive = Fraction(w.middle + w.beyond, w.days)
        facts = [(None, f"Passenger {pax} is booked on {f_in} connecting to {f_out} today."),
                 (None, f"The booking agent estimated the risk of missing the connection at {float(naive) * 100:.0f}% "
                        f"(every inbound more than {w.x} minutes late counted as a miss).")]
        state = f"{rules}\n\n" + render_file(n.style, f"CONNECTION CHECK {pax.upper()}", "Travel desk", facts)
        dist = self.solve(w)
        why = (f"P(make) = {w.within}/{w.days} + {w.middle}/{w.days} x {w.out_late}/{w.out_days} = {self.make(w)}; "
               f"P(miss) = {dist['yes']} = {float(dist['yes']):.4f}.")
        common = dict(base, use_case="customer_support", state=state, rationale=f"{base['rationale']} {why}",
                      target_probs=floats(dist))
        return [noul_item(instructions=f"Will passenger {pax} miss the connection to {f_out}? {ASK}",
                          true_text="The passenger misses the connection.",
                          false_text="The passenger makes the connection.", answer=dist["yes"] > Fraction(1, 2),
                          **common)]


# ------------------------------------------------------------------ urn

PRIZES = {"gold": "The drawn ticket is a gold ticket.", "silver": "The drawn ticket is a silver ticket.",
          "bronze": "The drawn ticket is a bronze ticket."}


@dataclass(frozen=True)
class UrnWorld:
    threshold: int            # die 1..threshold -> box A
    box_a: tuple              # (gold, silver, bronze)
    box_b: tuple


class Urn:
    name = "urn"

    @staticmethod
    def solve(w: UrnWorld) -> dict:
        pa = Fraction(w.threshold, 6)
        out = {}
        for i, k in enumerate(PRIZES):
            out[k] = pa * Fraction(w.box_a[i], sum(w.box_a)) + (1 - pa) * Fraction(w.box_b[i], sum(w.box_b))
        return out

    def sample(self, rng: random.Random) -> UrnWorld | None:
        w = UrnWorld(rng.randint(1, 5), tuple(rng.randint(0, 9) for _ in range(3)),
                     tuple(rng.randint(0, 9) for _ in range(3)))
        if not sum(w.box_a) or not sum(w.box_b) or not clear(self.solve(w)):
            return None
        twin = self.mutate(rng, w)
        return w if twin else None

    def mutate(self, rng: random.Random, w: UrnWorld) -> UrnWorld | None:
        lead = top(self.solve(w))
        for t in sorted(range(1, 6), key=lambda v: abs(v - w.threshold)):
            u = replace(w, threshold=t)
            if top(self.solve(u)) != lead and clear(self.solve(u)):
                return u
        return None

    def items(self, rng: random.Random, w: UrnWorld, n: Nar, base: dict) -> list[dict]:
        r = n.rng(15)
        shop = r.choice(["Harbour Books", "Linden Bakery", "Castle Cycles", "Mill Street Hardware"])
        box = lambda b: f"{b[0]} gold, {b[1]} silver and {b[2]} bronze tickets"
        rules = (f"{shop.upper()}: PROMOTION DRAW RULES\n- A fair six-sided die is rolled. On 1 to {w.threshold} the "
                 f"ticket is drawn from box A, otherwise from box B.\n- Box A holds {box(w.box_a)}. Box B holds "
                 f"{box(w.box_b)}.\n- One ticket is drawn at random from the chosen box.")
        facts = [(None, "Last week's draw produced a silver ticket from box B."),
                 (None, "Today's draw has not taken place yet.")]
        state = f"{rules}\n\n" + render_file(n.style, f"PROMOTION DRAW, {shop.upper()}", "Shop floor", facts)
        dist = self.solve(w)
        why = f"P(box A) = {w.threshold}/6; " + ", ".join(f"{k} {v}" for k, v in dist.items()) + "."
        common = dict(base, use_case="other", state=state, rationale=f"{base['rationale']} {why}",
                      target_probs=floats(dist))
        return [choice_item(rng=rng, options=dict(PRIZES), expected=top(dist), trap=None,
                            instructions=f"What kind of ticket will today's draw produce? {ASK}", **common)]


# ------------------------------------------------------------------ driver

TASKS = [ReferenceClass(), Bayes(), Sampling(), Connection(), Urn()]


def generate_case(seed: int) -> list[dict]:
    """One case: a base world and a twin whose single changed fact changes the most likely answer."""
    rng = random.Random(seed)
    task = TASKS[seed % len(TASKS)]
    w = None
    for _ in range(200):
        w = task.sample(rng)
        if w is not None:
            break
    if w is None:
        return []
    n = Nar(seed=seed, style=rng.choice(list(SPLIT_BY_STYLE)), ds=rng.choice(DATE_STYLES))
    group = f"dg-{FAMILY}-{stable_id(GENERATOR, str(seed))}"
    base = {"family": FAMILY, "generator": GENERATOR, "topic": "math", "group": group, "style": n.style}
    items = task.items(rng, w, n, dict(base, rationale=f"[base:{task.name}]"))
    twin = task.mutate(rng, w)
    if twin is not None and top(task.solve(twin)) != top(task.solve(w)):
        items += task.items(rng, twin, n, dict(base, rationale=f"[twin:{task.name}]"))
    return items
