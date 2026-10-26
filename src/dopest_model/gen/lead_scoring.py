"""Lead qualification: a sales lead scored on an explicit points guide, the level read off thresholds.

JevBench v1.5 added lead-qualification score items (48 public drafts and more in the sealed pool); none is published,
so this follows the idea only: a company profile, dated call notes (one figure superseded by a later note, a quote
request later withdrawn), the vendor's feature list, and a scoring guide with points per criterion, thresholds with
>= versus >, and two floors (a region the vendor does not serve disqualifies; fewer than ten employees caps the level
at Nurture). The engine adds the points; the labels are:
  score    the qualification level (Unqualified / Nurture / Qualified / Priority)
  noul     whether the lead reaches at least Qualified; whether the vendor serves the lead's region
  choice   the criterion that contributes the most points (skipped on a tie)
Twins move one fact across a threshold so that the level changes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import date, timedelta

from .common import SPLIT_BY_STYLE, Nar, choice_item, fmt_money, noul_item, render_file, score_item, stable_id

FAMILY = "lead_qualification"
GENERATOR = "lead_scoring_v1"
LEVELS = ["Unqualified: fewer than 3 points, or a region the vendor does not serve",
          "Nurture: 3 to 5 points, or a company with fewer than 10 employees whatever its points",
          "Qualified: 6 to 8 points", "Priority: 9 points or more"]
ROLE_POINTS = {"decision_maker": 2, "influencer": 1, "end_user": 0}
ROLE_TITLES = {"decision_maker": ["Chief Operating Officer", "VP of Operations", "Head of Procurement", "Managing Director"],
               "influencer": ["Team Lead, Customer Service", "Senior Analyst", "Project Manager", "IT Administrator"],
               "end_user": ["Support Agent", "Scheduling Coordinator", "Junior Accountant", "Field Technician"]}
FEATURES = ["single sign-on", "audit log export", "API access", "offline mode", "role-based permissions",
            "EU data residency", "custom reports", "two-factor login", "SMS alerts", "bulk import"]
REGIONS_SERVED = ["Germany", "Poland", "the Netherlands", "Spain", "Sweden", "Austria"]
REGIONS_NOT = ["Brazil", "India", "Canada", "Japan", "South Africa"]
CRITERIA = {"budget": "Budget: the confirmed annual budget", "timeline": "Timeline: the time to the decision",
            "contact": "Contact role: who the contact is", "fit": "Product fit: how many required features are offered",
            "engagement": "Engagement: a standing quote request", "relationship": "Relationship: customer history"}


@dataclass(frozen=True)
class Lead:
    employees: int
    budget: int
    budget_old: int | None
    weeks: int
    role: str
    required: tuple
    product: tuple           # features the vendor offers (the guide's fit criterion counts the required ones among them)
    region: str
    relationship: str        # new | customer | churned
    churn_months: int
    pricing: bool
    withdrawn: bool
    currency: str


def points(l: Lead) -> dict[str, int]:
    budget = 3 if l.budget >= 50000 else 2 if l.budget >= 20000 else 1 if l.budget >= 8000 else 0
    timeline = 2 if l.weeks <= 6 else 1 if l.weeks <= 16 else 0
    offered = sum(f in l.product for f in l.required)
    fit = 2 if offered == 3 else 1 if offered == 2 else 0
    engagement = 1 if l.pricing and not l.withdrawn else 0
    relationship = 1 if l.relationship == "customer" else (-1 if l.relationship == "churned" and l.churn_months <= 24 else 0)
    return {"budget": budget, "timeline": timeline, "contact": ROLE_POINTS[l.role], "fit": fit,
            "engagement": engagement, "relationship": relationship}


def level(l: Lead) -> tuple[int, int, str]:
    """(level index, points total, floor that changed the outcome or '')."""
    total = max(0, sum(points(l).values()))
    raw = 0 if total <= 2 else 1 if total <= 5 else 2 if total <= 8 else 3
    if l.region not in REGIONS_SERVED:
        return 0, total, "region" if raw > 0 else ""
    if l.employees < 10 and raw > 1:
        return 1, total, "size"
    return raw, total, ""


def sample(rng: random.Random) -> Lead:
    required = tuple(rng.sample(FEATURES, 3))
    offered_n = rng.choice([3, 3, 2, 2, 1])
    product = tuple(rng.sample(required, offered_n)) + tuple(rng.sample([f for f in FEATURES if f not in required], 3))
    hot = rng.random() < 0.3                     # a share of strong leads, so the Priority level is not rare
    budget = rng.choice([49900, 50000, 64000, 90000] if hot else
                        [4500, 7900, 8000, 12000, 19500, 20000, 26000, 35000, 49900, 50000, 64000, 90000])
    relationship = rng.choice(["new", "new", "customer", "churned"])
    return Lead(employees=rng.choice([6, 9, 10, 14, 40, 120, 350, 900]), budget=budget,
                budget_old=rng.choice([None, None, budget // 2, budget * 2]) if rng.random() < 0.5 else None,
                weeks=rng.choice([3, 6, 7, 10] if hot else [3, 6, 7, 10, 16, 17, 24, 36]),
                role=rng.choice(["decision_maker", "decision_maker", "influencer"] if hot else list(ROLE_POINTS)),
                required=required, product=tuple(sorted(product, key=lambda f: rng.random())),
                region=rng.choice(REGIONS_SERVED * 3 + REGIONS_NOT), relationship=relationship,
                churn_months=rng.choice([6, 18, 24, 25, 40]) if relationship == "churned" else 0,
                pricing=rng.random() < 0.6, withdrawn=rng.random() < 0.3, currency=rng.choice(["EUR", "USD", "GBP"]))


def mutate(rng: random.Random, l: Lead) -> Lead:
    """One fact across a threshold."""
    k = rng.choice(["budget", "weeks", "region", "role", "employees", "withdrawn"])
    if k == "budget":
        return replace(l, budget=rng.choice([7900, 8000, 19500, 20000, 49900, 50000]))
    if k == "weeks":
        return replace(l, weeks=rng.choice([6, 7, 16, 17]))
    if k == "region":
        return replace(l, region=rng.choice(REGIONS_NOT if l.region in REGIONS_SERVED else REGIONS_SERVED))
    if k == "role":
        return replace(l, role=rng.choice([r for r in ROLE_POINTS if r != l.role]))
    if k == "employees":
        return replace(l, employees=9 if l.employees >= 10 else 40)
    return replace(l, withdrawn=not l.withdrawn, pricing=True)


def guide(currency: str) -> str:
    c = fmt_money
    return ("LEAD SCORING GUIDE (SALES-OPS, REVISION 5)\n"
            f"Budget (confirmed annual budget; a later confirmation replaces an earlier indicative figure): {c(50000, currency)} or more = 3 points; "
            f"{c(20000, currency)} to {c(49999, currency)} = 2; {c(8000, currency)} to {c(19999, currency)} = 1; below {c(8000, currency)} = 0.\n"
            "Timeline (time to the purchase decision): 6 weeks or less = 2 points; 7 to 16 weeks = 1; more than 16 weeks = 0.\n"
            "Contact role: decision maker (C-level, managing director, VP, head of a department) = 2 points; influencer "
            "(team lead, project manager, senior analyst, IT administrator) = 1; end user (agent, coordinator, junior staff, "
            "technician) = 0.\n"
            "Product fit: all three required features are in our feature list = 2 points; two of them = 1; one or none = 0.\n"
            "Engagement: a formal quote requested and not withdrawn = 1 point.\n"
            "Relationship: existing customer = 1 point; churned within the last 24 months = minus 1 point; churned longer "
            "ago, or no history = 0. The total never goes below 0.\n"
            "Levels: 0 to 2 points = Unqualified; 3 to 5 = Nurture; 6 to 8 = Qualified; 9 or more = Priority.\n"
            f"Floors: a company in a region we do not serve is Unqualified whatever its points (we serve {', '.join(REGIONS_SERVED)}); "
            "a company with fewer than 10 employees is at most Nurture.\n")


def items(rng: random.Random, l: Lead, n: Nar, base: dict) -> list[dict]:
    r = n.rng(41)
    company = r.choice(["Harlow Logistics", "Mirabel Clinics", "Tessera Retail Group", "Ostrava Tooling", "Blue Fjord Media",
                        "Kestrel Energy Services", "Vantage Legal", "Alder & Finch Accountants"])
    contact = r.choice(["Ms. Dubois", "Mr. Okafor", "Dr. Lindqvist", "Ms. Petrova", "Mr. Tanaka", "Ms. Alvarez"])
    title = r.choice(ROLE_TITLES[l.role])
    d0 = date(2026, r.randint(1, 9), r.randint(1, 26))
    c = l.currency
    facts = [(d0, f"Discovery call with {contact} ({title}, {company}). The company has {l.employees} employees and is "
                  f"headquartered in {l.region}. Required features: {', '.join(l.required)}. A decision is expected in about "
                  f"{l.weeks} weeks.")]
    if l.budget_old is not None:
        facts.append((d0 + timedelta(days=2), f"Indicative budget mentioned on the call: {fmt_money(l.budget_old, c)} per year, pending finance."))
        facts.append((d0 + timedelta(days=r.randint(6, 15)), f"Finance confirmed the annual budget at {fmt_money(l.budget, c)}."))
    else:
        facts.append((d0 + timedelta(days=r.randint(1, 8)), f"Finance confirmed the annual budget at {fmt_money(l.budget, c)}."))
    if l.relationship == "customer":
        facts.append((None, f"{company} has been a customer since {d0.year - r.randint(1, 4)} on the Standard plan."))
    elif l.relationship == "churned":
        facts.append((None, f"{company} was a customer and cancelled {l.churn_months} months ago."))
    if l.pricing:
        facts.append((d0 + timedelta(days=r.randint(3, 12)), f"{contact} requested a formal quote for {l.employees} seats."))
        if l.withdrawn:
            facts.append((d0 + timedelta(days=r.randint(13, 25)), f"{contact} asked us to hold the quote: no decision this quarter on pricing, the evaluation continues."))
    facts.append((None, f"Our current feature list: {', '.join(l.product)}."))
    state = guide(c) + "\n" + render_file(n.style, f"Lead file: {company}", f"Owner: sales desk, prepared {n.d(d0 + timedelta(days=30))}", facts)
    lvl, total, floor = level(l)
    pts = points(l)
    why = (f"Points {pts} = {total}; raw level {['Unqualified', 'Nurture', 'Qualified', 'Priority'][0 if total <= 2 else 1 if total <= 5 else 2 if total <= 8 else 3]}"
           f"{'; floor ' + floor if floor else ''} -> level {lvl}.")
    tier = "hard" if (floor or l.budget_old is not None or l.withdrawn) else "standard"
    common = dict(base, state=state, rationale=f"{base['rationale']} {why}", tier=tier)
    out = [score_item(instructions="Apply the lead scoring guide to the lead file. Which qualification level does this lead get?",
                      levels=LEVELS, expected=lvl, **common),
           noul_item(instructions="Under the guide, does this lead reach at least the Qualified level?",
                     true_text="The lead's level is Qualified or Priority.",
                     false_text="The lead's level is Unqualified or Nurture.", answer=lvl >= 2, **common),
           noul_item(instructions="Is the lead's region one the vendor serves?",
                     true_text="The headquarters region is in the guide's list of served regions.",
                     false_text="The region is not in the list, so the lead is Unqualified whatever its points.",
                     answer=l.region in REGIONS_SERVED, **dict(common, tier="easy"))]
    best = max(pts.values())
    top = max(pts, key=pts.get) if best > 0 else None
    if sum(v == best for v in pts.values()) == 1 and best > 0 and (top != "budget" or r.random() < 0.35):
        top = max(pts, key=pts.get)
        second = max((k for k in pts if k != top), key=pts.get)
        out.append(choice_item(rng=rng, options=dict(CRITERIA), expected=top, trap=second if pts[second] > 0 else None,
                               instructions="Which criterion of the guide contributes the most points to this lead's score?",
                               **common))
    return out


def generate_case(seed: int) -> list[dict]:
    rng = random.Random(seed)
    l = sample(rng)
    n = Nar(seed=seed, style=rng.choice(list(SPLIT_BY_STYLE)), ds=rng.choice(["dmy", "mdy", "iso"]))
    group = f"dg-{FAMILY}-{stable_id(GENERATOR, str(seed))}"
    base = {"family": FAMILY, "generator": GENERATOR, "use_case": "sales", "topic": "sales_ops", "group": group,
            "style": n.style}
    out = items(rng, l, n, dict(base, rationale="[base:lead]"))
    for _ in range(12):
        t = mutate(rng, l)
        if level(t)[0] != level(l)[0]:
            out += items(rng, t, n, dict(base, rationale="[twin:lead]"))
            break
    return out
