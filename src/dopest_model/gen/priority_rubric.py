"""Priority under a written rubric: an incident ticket rated on ordered levels by rules applied in a stated order.

A ticket record (affected users, revenue impact, workaround, customer tier, security or data-loss flags, time since
the last update) and a rubric: a base level from the share of users affected, two adjustments, two floors and an
ageing rule, applied in the written order. Two signals pull in different directions on purpose (a workaround lowers
the level, but not for a security incident; a floor stops the lowering). The engine applies the rules; the labels:
  score    the final priority level (P4 Low .. P1 Critical)
  noul     whether the final priority is at least P2; whether the workaround adjustment applies
  choice   the rule that last changed the level
Twins flip one fact (workaround, security, users across a band, the ageing rule).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .common import SPLIT_BY_STYLE, Nar, choice_item, noul_item, render_file, score_item, stable_id

FAMILY = "priority_rubric"
GENERATOR = "priority_rubric_v1"
LEVELS = ["P4 Low: no customer impact beyond a few users, fix in the normal backlog",
          "P3 Medium: a limited group of users affected, fix within the sprint",
          "P2 High: a significant group affected or a floor rule applies, fix within one business day",
          "P1 Critical: a wide outage or a critical escalation, all hands until resolved"]
RULES = {"R1": "R1 base level from the share of users affected", "R2": "R2 revenue adjustment (raise one level)",
         "R3": "R3 workaround adjustment (lower one level)", "R4": "R4 security or data-loss floor (at least P2)",
         "R5": "R5 enterprise floor (at least P2)", "R6": "R6 ageing rule (raise one level)"}
RUBRIC = ("INCIDENT PRIORITY RUBRIC (SUPPORT-OPS 2.1). Apply the rules in this order; each rule reads the level left by "
          "the previous one. Levels from lowest to highest: P4 Low, P3 Medium, P2 High, P1 Critical.\n"
          "R1 Base level from the share of the customer's users affected: 40% or more = P1; 10% to less than 40% = P2; "
          "less than 10% = P3.\n"
          "R2 Revenue: a documented revenue loss of 5,000 or more per hour raises the level by one (P1 stays P1).\n"
          "R3 Workaround: a documented workaround lowers the level by one (P4 is the lowest), except for a security "
          "incident or data loss, where this rule is skipped.\n"
          "R4 Security or data loss: the level is at least P2.\n"
          "R5 Enterprise tier with 50 or more users affected: the level is at least P2.\n"
          "R6 Ageing: more than 72 hours since the last update raises the level by one (P1 stays P1).\n")


@dataclass(frozen=True)
class Ticket:
    users: int
    total: int
    revenue: int
    workaround: bool
    tier: str
    security: bool
    data_loss: bool
    since_update_h: int


def apply(t: Ticket) -> tuple[int, list[tuple[str, int, int]]]:
    """Final level index and the trail of (rule, before, after) for the rules that changed the level."""
    share = t.users / t.total
    lvl = 3 if share >= 0.4 else 2 if share >= 0.1 else 1
    trail = [("R1", -1, lvl)]
    def step(rule, new):
        nonlocal lvl
        if new != lvl:
            trail.append((rule, lvl, new))
            lvl = new
    if t.revenue >= 5000:
        step("R2", min(3, lvl + 1))
    if t.workaround and not (t.security or t.data_loss):
        step("R3", max(0, lvl - 1))
    if t.security or t.data_loss:
        step("R4", max(2, lvl))
    if t.tier == "enterprise" and t.users >= 50:
        step("R5", max(2, lvl))
    if t.since_update_h > 72:
        step("R6", min(3, lvl + 1))
    return lvl, trail


def sample(rng: random.Random) -> Ticket:
    total = rng.choice([120, 250, 400, 800, 1500, 3000])
    share = rng.choice([0.02, 0.05, 0.08, 0.1, 0.12, 0.25, 0.39, 0.4, 0.55])
    return Ticket(users=max(1, round(total * share)), total=total,
                  revenue=rng.choice([0, 0, 0, 1200, 4900, 5000, 8000, 15000]), workaround=rng.random() < 0.55,
                  tier=rng.choice(["standard", "standard", "enterprise"]), security=rng.random() < 0.15,
                  data_loss=rng.random() < 0.12, since_update_h=rng.choice([4, 20, 48, 60, 72, 73, 96, 140]))


def mutate(rng: random.Random, t: Ticket) -> Ticket:
    k = rng.choice(["workaround", "security", "users", "age", "tier", "revenue"])
    if k == "workaround":
        return replace(t, workaround=not t.workaround)
    if k == "security":
        return replace(t, security=not t.security)
    if k == "users":
        return replace(t, users=max(1, round(t.total * rng.choice([0.08, 0.1, 0.39, 0.4]))))
    if k == "age":
        return replace(t, since_update_h=72 if t.since_update_h > 72 else 96)
    if k == "tier":
        return replace(t, tier="enterprise" if t.tier == "standard" else "standard")
    return replace(t, revenue=4900 if t.revenue >= 5000 else 5000)


def items(rng: random.Random, t: Ticket, n: Nar, base: dict) -> list[dict]:
    r = n.rng(43)
    customer = r.choice(["Nordhavn Freight", "Cedar Valley Clinics", "Lumen Retail", "Aster Payroll", "Brightwater Utilities",
                         "Pinecrest Schools", "Halden Insurance", "Orchard Bank"])
    ticket_id = f"INC-{r.randint(10000, 99999)}"
    opened = datetime(2026, r.randint(1, 10), r.randint(1, 26), r.randint(6, 20), r.choice([0, 15, 30, 45]))
    last = opened + timedelta(hours=r.randint(1, 30))
    now = last + timedelta(hours=t.since_update_h)
    symptom = r.choice(["the invoice export returns an empty file", "login via single sign-on loops back to the start page",
                        "scheduled reports arrive three hours late", "the mobile app shows stale balances",
                        "search results omit records created this week", "the API returns HTTP 500 for bulk updates"])
    facts = [(opened.date(), f"{ticket_id} opened at {opened:%H:%M}: {customer} reports that {symptom}."),
             (opened.date(), f"Scope confirmed by the account team: {t.users} of the customer's {t.total} users are affected."),
             (None, f"Customer tier: {t.tier.capitalize()}."),
             (None, (f"Revenue impact documented by the customer: about {t.revenue:,} per hour of outage." if t.revenue
                     else "No revenue impact has been documented."))]
    if t.workaround:
        facts.append((last.date(), r.choice(["Workaround documented in the ticket: the legacy export endpoint works for all affected users.",
                                             "Workaround documented: clearing the session cookie restores access until the fix ships."])))
    else:
        facts.append((last.date(), "No workaround is known; the team is still reproducing the issue."))
    if t.security:
        facts.append((last.date(), "The security team confirmed the incident involves an authentication bypass and classified it as a security incident."))
    if t.data_loss:
        facts.append((last.date(), "Engineering confirmed that records written during the window were lost and must be restored from backup."))
    facts.append((None, f"Last update on the ticket: {last:%d %b %H:%M}. Current time: {now:%d %b %H:%M} ({t.since_update_h} hours later)."))
    if r.random() < 0.4:
        facts.append((None, r.choice(["A support agent commented: 'feels like a P1 to me, the customer is furious.'",
                                      "The account manager asked for the lowest possible priority to avoid alarming the customer."])))
    state = RUBRIC + "\n" + render_file(n.style, f"Ticket {ticket_id}", f"Customer: {customer}", facts)
    lvl, trail = apply(t)
    why = f"Trail {trail} -> final level {lvl} ({LEVELS[lvl].split(':')[0]})."
    tier = "hard" if len(trail) >= 3 else "standard"
    common = dict(base, state=state, rationale=f"{base['rationale']} {why}", tier=tier)
    r3 = any(rule == "R3" for rule, _, _ in trail)
    out = [score_item(instructions=f"Apply the SUPPORT-OPS 2.1 rubric to ticket {ticket_id}. Which priority level results?",
                      levels=LEVELS, expected=lvl, **common),
           noul_item(instructions=f"Is the final priority of {ticket_id} at least P2 (High)?",
                     true_text="The level after all rules is P2 or P1.", false_text="The level after all rules is P3 or P4.",
                     answer=lvl >= 2, **common),
           noul_item(instructions=f"Does rule R3 (workaround) lower the level of {ticket_id}?",
                     true_text="A workaround is documented, the incident is neither a security incident nor data loss, and the level before R3 was above P4.",
                     false_text="R3 does not apply or changes nothing: no workaround, a security or data-loss incident, or the level was already P4.",
                     answer=r3, **common),
           choice_item(rng=rng, options=dict(RULES), expected=trail[-1][0],
                       trap=trail[-2][0] if len(trail) >= 2 else None,
                       instructions=f"Which rule is the last one that changes the level of {ticket_id}?", **common)]
    return out


def generate_case(seed: int) -> list[dict]:
    rng = random.Random(seed)
    t = sample(rng)
    n = Nar(seed=seed, style=rng.choice(list(SPLIT_BY_STYLE)), ds="dmy")
    group = f"dg-{FAMILY}-{stable_id(GENERATOR, str(seed))}"
    base = {"family": FAMILY, "generator": GENERATOR, "use_case": "incident_management", "topic": "support_ops",
            "group": group, "style": n.style}
    out = items(rng, t, n, dict(base, rationale="[base:ticket]"))
    for _ in range(12):
        m = mutate(rng, t)
        if apply(m)[0] != apply(t)[0]:
            out += items(rng, m, n, dict(base, rationale="[twin:ticket]"))
            break
    return out
