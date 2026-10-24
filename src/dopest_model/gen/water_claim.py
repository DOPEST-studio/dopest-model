"""Homeowners water-damage claim under a long policy form (JevBench hard tier, family long_policy).

The settlement follows from the documents by a rule engine. Items exercise:
- leak duration against the repeated-seepage period (estimates are ranges that never straddle it),
- the concealed-water exception: hidden in a wall, floor or ceiling AND no earlier sign noticed,
- 'vacant' versus 'unoccupied' by the definitions (furnished but empty is unoccupied, not vacant),
- which sublimit binds: the endorsement covers terms issued or renewed on or after its effective date only
  (the loss date and mid-term changes do not matter),
- the notice condition, counted from the date of discovery,
- settlement order: deductible first, then the sublimit.

A case is kept only if at most one ground of denial applies, so every question has one defensible answer.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

from .common import DATE_STYLES, SPLIT_BY_STYLE, add_days, choice_item, days_between, fmt_date, fmt_money, noul_item, \
    score_item, stable_id

FAMILY = "long_policy"
GENERATOR = "water_claim_v1"
COMMON = {"family": FAMILY, "generator": GENERATOR, "use_case": "insurance_claims", "topic": "law_policy"}

DENY_NOTICE, DENY_VACANCY, DENY_SEEPAGE = "deny_late_notice", "deny_vacancy_exclusion", "deny_repeated_seepage"
PAY_FULL, MISSING = "pay_full_estimate_less_deductible", "request_missing_information"
FACTS = ("leak_days", "noticed_on", "furnished", "renewal")   # facts that a file may lack


class Ambiguous(Exception):
    """More than one ground of denial applies: no single defensible settlement label."""


def pay_capped(cap: int) -> str:
    return f"pay_subject_to_{cap}_sublimit"


@dataclass(frozen=True)
class Policy:
    insurer: str
    form: str
    currency: str
    deductible: int
    seepage_days: int
    vacancy_days: int
    notice_days: int
    sublimit_old: int
    sublimit_new: int
    endorsement: str
    effective: date


@dataclass(frozen=True)
class Case:
    discovery: date
    reported: date
    leak_days: tuple[int, int]
    concealed: bool
    noticed_on: date | None          # an earlier visible sign noticed by the insured
    away_days: int                   # consecutive days nobody lived there, up to discovery (0 = lived in)
    furnished: bool
    renewal: date                    # start of the current policy term
    estimate: int
    missing: frozenset = field(default_factory=frozenset)


# ------------------------------------------------------------------ rule engine

def sublimit(p: Policy, c: Case) -> int:
    return p.sublimit_new if c.renewal >= p.effective else p.sublimit_old


def is_seepage(p: Policy, c: Case) -> bool:
    return c.leak_days[0] >= p.seepage_days


def is_restored(c: Case) -> bool:
    return c.concealed and c.noticed_on is None


def grounds(p: Policy, c: Case) -> list[str]:
    out = []
    if days_between(c.discovery, c.reported) > p.notice_days:
        out.append(DENY_NOTICE)
    if c.away_days > p.vacancy_days and not c.furnished:
        out.append(DENY_VACANCY)
    if is_seepage(p, c) and not is_restored(c):
        out.append(DENY_SEEPAGE)
    return out


def settle(p: Policy, c: Case) -> tuple[str, int]:
    """Settlement of a case whose facts are all known: (label, amount payable)."""
    g = grounds(p, c)
    if len(g) > 1:
        raise Ambiguous(g)
    if g:
        return g[0], 0
    if is_seepage(p, c):
        cap = sublimit(p, c)
        return pay_capped(cap), min(c.estimate - p.deductible, cap)
    return PAY_FULL, c.estimate - p.deductible


def alternatives(p: Policy, c: Case, fact: str) -> list:
    """Values a missing fact could take; the engine decides only if every value gives the same label."""
    if fact == "leak_days":
        return [(1, 3), (p.seepage_days + 7, p.seepage_days + 21)]
    if fact == "noticed_on":
        return [None, add_days(c.discovery, -3)]
    if fact == "furnished":
        return [True, False]
    if fact == "renewal":
        return [add_days(p.effective, -30), add_days(p.effective, 30)]
    raise KeyError(fact)


def outcomes(p: Policy, c: Case) -> set[tuple[str, int]]:
    """Settlements over every value the missing facts could take (one world if nothing is missing)."""
    worlds = [c]
    for fact in sorted(c.missing):
        worlds = [replace(w, **{fact: v}) for w in worlds for v in alternatives(p, w, fact)]
    return {settle(p, w) for w in worlds}


def decide(p: Policy, c: Case) -> tuple[str, int]:
    results = outcomes(p, c)
    if len({label for label, _ in results}) == 1:
        return results.pop()
    return MISSING, 0


def explain(p: Policy, c: Case, n: "Narrative") -> str:
    """Short trace of the rules applied, stored as the item's rationale."""
    d = lambda x: fmt_date(x, n.date_style)
    m = lambda x: fmt_money(x, p.currency)
    parts = []
    lag = days_between(c.discovery, c.reported)
    parts.append(f"Reported {lag} days after discovery (limit {p.notice_days}): "
                 f"{'late' if lag > p.notice_days else 'timely'}.")
    if "furnished" in c.missing:
        parts.append(f"Nobody lived there for {c.away_days} days, but the file does not say whether it was furnished.")
    elif c.away_days:
        state = "furnished, so unoccupied, not vacant" if c.furnished else "emptied, so vacant"
        hit = c.away_days > p.vacancy_days and not c.furnished
        parts.append(f"Nobody lived there for {c.away_days} days ({state}; vacancy limit {p.vacancy_days} days): "
                     f"vacancy exclusion {'applies' if hit else 'does not apply'}.")
    if "leak_days" in c.missing:
        parts.append("The file gives no leak duration.")
    else:
        lo, hi = c.leak_days
        parts.append(f"Leak {lo}-{hi} days vs repeated seepage at {p.seepage_days} days: "
                     f"{'repeated seepage' if is_seepage(p, c) else 'not repeated seepage'}.")
    if "noticed_on" in c.missing:
        parts.append("The file does not say whether any earlier sign was noticed.")
    elif is_seepage(p, c) or "leak_days" in c.missing:
        if not c.concealed:
            parts.append("The leak point was exposed, not within a wall, floor or ceiling: no concealed water.")
        elif c.noticed_on:
            parts.append(f"An earlier sign was noticed on {d(c.noticed_on)}: not concealed water.")
        else:
            parts.append("Hidden in the structure and no earlier sign noticed: concealed water, exception applies.")
    if "renewal" in c.missing:
        parts.append(f"The current term's start date is missing, so {p.endorsement} cannot be placed.")
    else:
        when = "on or after" if c.renewal >= p.effective else "before"
        parts.append(f"Current term renewed {d(c.renewal)}, {when} {p.endorsement} effective {d(p.effective)}: "
                     f"sublimit {m(sublimit(p, c))}.")
    label, amount = decide(p, c)
    if label.startswith("pay_subject"):
        parts.append(f"{m(c.estimate)} less {m(p.deductible)} = {m(c.estimate - p.deductible)}, capped at {m(amount)}.")
    elif label == PAY_FULL:
        parts.append(f"{m(c.estimate)} less {m(p.deductible)} = {m(amount)}.")
    parts.append(f"Outcome: {label}.")
    return " ".join(parts)


# ------------------------------------------------------------------ sampling

INSURERS = ["Harrowgate Mutual", "Northvale Insurance", "Calder & Rowe Assurance", "Bluestone General",
            "Kestrel Home Insurance", "Larchmont Mutual", "Ashby Fire & General", "Meridian Household"]
FIRST = ["Laura", "Tomasz", "Priya", "Daniel", "Ines", "Marcus", "Aiko", "Henrik", "Fatima", "Owen", "Chloe",
         "Rafael", "Nadia", "Samuel", "Greta", "Jonah"]
LAST = ["Becker", "Okafor", "Lindqvist", "Moreau", "Castillo", "Novak", "Hughes", "Tanaka", "Albers", "Quinn",
        "Fischer", "Delgado", "Brennan", "Sato", "Kowalczyk", "Ward"]
STREETS = ["Linden Avenue", "Harbour Road", "Elm Close", "Orchard Lane", "Mill Street", "Castle View", "Fern Way"]
TOWNS = ["Westbury", "Redhill", "Ashford", "Millbrook", "Kingsley", "Oakham", "Danbury"]
PLUMBERS = ["Brightwater Plumbing", "Keel & Son Plumbing", "Corbel Pipeworks", "Riverside Plumbing & Heating"]
BUILDERS = ["Arden Restoration", "Hollis Building Services", "Tessera Repairs", "Granite & Oak Contractors"]
ROOMS = ["upstairs bathroom", "kitchen", "laundry room", "en-suite bathroom"]
CAUSES = ["a pinhole leak in a copper supply line", "a failed compression fitting", "a cracked PEX connector",
          "a split solder joint on the hot-water feed"]
HIDDEN = ["inside the wall cavity behind the {room} units", "under the {room} floor, between the joists",
          "in the ceiling void above the {room}"]
EXPOSED = {
    "upstairs bathroom": ["at the exposed shut-off valve below the basin", "at the exposed trap under the basin"],
    "en-suite bathroom": ["at the exposed shut-off valve below the basin", "at the exposed trap under the basin"],
    "kitchen": ["under the kitchen sink, inside the base cabinet", "at the exposed hose connection behind the "
                "dishwasher"],
    "laundry room": ["at the exposed hose connection behind the washing machine", "under the utility sink, inside "
                     "the base cabinet"],
}


@dataclass(frozen=True)
class Narrative:
    """Every random choice that does not change the decision, drawn once per case so twins differ in one fact."""
    seed: int
    style: str
    date_style: str
    insured: str
    address: str
    claim: str
    policy_no: str
    adjuster: str
    plumber: str
    builder: str
    room: str
    cause: str
    hidden: str
    exposed: str
    first_issued_years: int
    prior_claim_year: int
    contents_claim: int
    midterm_change: bool
    draft_endorsement: str | None
    backup_endorsement: bool
    question_variant: int


def sample_policy(rng: random.Random) -> Policy:
    old = rng.choice([5000, 7500, 10000, 15000, 20000])
    new = rng.choice([x for x in (5000, 7500, 10000, 12500, 15000, 20000, 25000) if x != old])
    return Policy(insurer=rng.choice(INSURERS), form=f"{rng.choice(['HX', 'HO', 'HPF'])}-{rng.randint(3, 29)}",
                  currency=rng.choice(["USD", "EUR", "GBP"]), deductible=rng.choice([500, 750, 1000, 1500, 2500]),
                  seepage_days=rng.choice([14, 21, 30]), vacancy_days=rng.choice([30, 60, 90]),
                  notice_days=rng.choice([30, 45, 60]), sublimit_old=old, sublimit_new=new,
                  endorsement=f"HE-{rng.randint(1100, 2999)}", effective=date(2026, 1, 1))


def sample_narrative(rng: random.Random, seed: int) -> Narrative:
    room = rng.choice(ROOMS)
    return Narrative(
        seed=seed, style=rng.choice(list(SPLIT_BY_STYLE)), date_style=rng.choice(DATE_STYLES),
        insured=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
        address=f"{rng.randint(2, 98)} {rng.choice(STREETS)}, {rng.choice(TOWNS)}",
        claim=f"{rng.choice(['HX', 'WC', 'CL'])}-2026-{rng.randint(100000, 999999)}",
        policy_no=f"{rng.choice(['HP', 'HO', 'PH'])}{rng.randint(1000000, 9999999)}",
        adjuster=f"{rng.choice(FIRST)} {rng.choice(LAST)}", plumber=rng.choice(PLUMBERS),
        builder=rng.choice(BUILDERS), room=room, cause=rng.choice(CAUSES),
        hidden=rng.choice(HIDDEN).format(room=room), exposed=rng.choice(EXPOSED[room]),
        first_issued_years=rng.randint(2, 9), prior_claim_year=rng.randint(2019, 2024),
        contents_claim=rng.randint(8, 90) * 100, midterm_change=rng.random() < 0.5,
        draft_endorsement=f"HE-{rng.randint(3000, 3999)}" if rng.random() < 0.4 else None,
        backup_endorsement=rng.random() < 0.6, question_variant=rng.randrange(3))


PATHS = (("late", 0.15), ("vacant", 0.15), ("seepage_deny", 0.2), ("sublimit", 0.3), ("full", 0.2))


def sample_case(rng: random.Random, p: Policy) -> tuple[Policy, Case]:
    path = rng.choices([name for name, _ in PATHS], weights=[w for _, w in PATHS])[0]
    discovery = date(2026, rng.randint(2, 10), rng.randint(1, 28))
    renewal = add_days(discovery, -rng.randint(3, 360))
    # Endorsement date: either before the current term (new sublimit) or after the renewal, often before the loss,
    # which is the trap: the loss happens after the effective date but the term started before it.
    if rng.random() < 0.5:
        effective = add_days(renewal, -rng.randint(0, 240))
    else:
        effective = add_days(renewal, rng.randint(1, max(1, days_between(renewal, discovery) + 45)))
    p = replace(p, effective=effective)

    late = path == "late"
    if late:
        lag = p.notice_days + rng.randint(1, 40)
    else:                                                       # ~15% exactly on the last allowed day
        lag = p.notice_days if rng.random() < 0.15 else rng.randint(0, p.notice_days - 1)
    reported = add_days(discovery, lag)

    if path == "vacant":
        away, furnished = p.vacancy_days + rng.randint(1, 120), False
    else:
        kind = rng.choice(["lived_in", "unoccupied_long", "vacant_short"])
        if kind == "lived_in":
            away, furnished = 0, True
        elif kind == "unoccupied_long":
            away, furnished = p.vacancy_days + rng.randint(1, 90), True
        else:
            away, furnished = rng.randint(max(1, p.vacancy_days - 25), p.vacancy_days), False

    long_leak = path in ("seepage_deny", "sublimit") or (path in ("late", "vacant") and rng.random() < 0.5)
    if long_leak:
        lo = p.seepage_days + rng.choice([0, rng.randint(1, 30)])
        leak = (lo, lo + rng.randint(3, 21))
    elif rng.random() < 0.5:
        leak = (0, 0)                                           # sudden burst, found within hours
    else:
        hi = rng.randint(2, p.seepage_days - 1)
        leak = (max(1, hi - rng.randint(1, 6)), hi)

    concealed = True if path == "sublimit" else rng.random() < (0.5 if path == "seepage_deny" else 0.6)
    floor = p.deductible + (max(p.sublimit_old, p.sublimit_new) + 500 if long_leak else 1000)
    estimate = floor + rng.randint(0, 300) * 100
    c = Case(discovery=discovery, reported=reported, leak_days=leak, concealed=concealed, noticed_on=None,
             away_days=away, furnished=furnished, renewal=renewal, estimate=estimate)
    if path == "seepage_deny" and concealed or path not in ("sublimit", "seepage_deny") and rng.random() < 0.25:
        window = sign_window(p, c)
        if window:
            c = replace(c, noticed_on=add_days(window[0], rng.randint(0, days_between(*window))))
        elif path == "seepage_deny":
            c = replace(c, concealed=False)
    return p, c


def sign_window(p: Policy, c: Case) -> tuple[date, date] | None:
    """Dates on which an earlier sign of this leak could have been noticed: after the leak began, at least 3 days
    before discovery, and late enough that notice counted from the sign would still be timely."""
    if c.leak_days[0] < 5:
        return None
    lo = max(add_days(c.discovery, 1 - c.leak_days[0]), add_days(c.reported, -p.notice_days))
    hi = add_days(c.discovery, -3)
    return (lo, hi) if lo <= hi else None


# ------------------------------------------------------------------ rendering

def render_policy(p: Policy, n: Narrative, discovery: date) -> tuple[str, dict]:
    rng = random.Random(n.seed * 7 + 1)
    d = lambda x: fmt_date(x, n.date_style)
    m = lambda x: fmt_money(x, p.currency)
    head = rng.choice(["SECTION", "PART", "ARTICLE"])
    excl_no, cond_no, sett_no = 4, 5, 6
    exclusions = [
        ("Flood and surface water", "Flood, surface water, waves, tidal water or overflow of a body of water, "
         "whether or not driven by wind."),
        ("Earth movement", "Earthquake, landslide, mudflow, subsidence or sinkhole, however caused."),
        ("Wear and tear", "We do not pay to repair or replace the part of a system or appliance that wore out, "
         "rusted or failed. Water damage resulting from that failure is not excluded by this item."),
        ("Power failure", "Failure of power or other utility service if the failure takes place off the "
         "residence premises."),
        ("Intentional loss", "Any loss arising out of an act an insured commits or conspires to commit with the "
         "intent to cause a loss."),
    ]
    if 5 <= discovery.month <= 9:          # summer only, so a sudden winter leak never gives a second defence
        exclusions.append(("Freezing", "Freezing of a plumbing, heating or air-conditioning system while the dwelling "
                           "is vacant, unless heat was maintained or the water supply was shut off and the system "
                           "drained. This item applies only to loss caused by freezing."))
    rng.shuffle(exclusions)
    seep_pos = rng.randint(1, len(exclusions) + 1)
    vac_pos = rng.choice([i for i in range(1, len(exclusions) + 3) if i != seep_pos])
    order, seep_ref, vac_ref = [], "", ""
    filler = iter(exclusions)
    for i in range(1, len(exclusions) + 3):
        ref = f"{excl_no}.{i}"
        if i == seep_pos:
            seep_ref = ref
            order.append(f"{ref} Repeated seepage or leakage. Loss caused by repeated seepage or leakage of water or "
                         f"steam from within a plumbing, heating, air-conditioning or sprinkler system or a household "
                         f"appliance.\n{ref}.1 Exception: concealed water. Exclusion {ref} does not apply to "
                         f"loss caused by concealed water. Coverage restored by this exception is limited to the "
                         f"Concealed Water Sublimit in {head} {sett_no}.")
        elif i == vac_pos:
            vac_ref = ref
            order.append(f"{ref} Vacancy. Loss caused by water, including leakage of any kind, if on the date of "
                         f"discovery the dwelling has been vacant for more than {p.vacancy_days} consecutive days. "
                         f"A dwelling that is only unoccupied is not vacant.")
        else:
            title, text = next(filler)
            order.append(f"{ref} {title}. {text}")

    definitions = [
        ("Concealed water", "water hidden within the walls, floors or ceilings of the dwelling, provided that no "
         "insured knew of it, and no visible sign of it had been noticed by any insured, before the date of "
         "discovery."),
        ("Repeated seepage or leakage", f"continuous or repeated seepage or leakage of water over a period of "
         f"{p.seepage_days} days or more."),
        ("Vacant", "the dwelling has no one living in it and does not contain enough furniture and personal "
         "property for customary occupancy."),
        ("Unoccupied", "the dwelling has no one living in it but still contains enough furniture and personal "
         "property for customary occupancy."),
        ("Date of discovery", "the date on which an insured or a contractor first identified the damage as water "
         "damage."),
        ("Policy term", "the period shown in the Declarations. Each renewal starts a new policy term. A mid-term "
         "change to the policy does not start a new policy term."),
        ("Dwelling", "the house on the residence premises shown in the Declarations, including attached structures."),
        ("Occurrence", "one event, or continuous or repeated exposure to substantially the same conditions."),
        ("Replacement cost", "the cost to repair or replace with material of like kind and quality, without "
         "deduction for depreciation."),
        ("Business", "a trade, profession or occupation engaged in on a full-time, part-time or occasional basis."),
        ("Surface water", "water on the surface of the ground, including runoff from rain or melting snow."),
    ]
    rng.shuffle(definitions)
    defs = "\n".join(f"1.{i} \"{t}\" means {txt}" for i, (t, txt) in enumerate(definitions, 1))

    endorsements = [f"{p.endorsement} Concealed Water Sublimit Amendment. This endorsement applies to policy terms "
                    f"issued or renewed with an effective date on or after {d(p.effective)}. For those terms the "
                    f"Concealed Water Sublimit in {head} {sett_no} is {m(p.sublimit_new)} per occurrence. All other "
                    f"provisions of the policy are unchanged."]
    if n.backup_endorsement:
        endorsements.append(f"HE-{1000 + n.seed % 97} Water Backup and Sump Overflow. Adds cover for water that backs "
                            f"up through sewers or drains or overflows from a sump, up to {m(10000)} per occurrence. "
                            f"It does not apply to water from a plumbing supply line.")
    rng.shuffle(endorsements)

    text = f"""{p.insurer.upper()} HOMEOWNERS POLICY, FORM {p.form}

{head} 1 DEFINITIONS
{defs}

{head} 2 PROPERTY COVERAGES
2.1 Coverage A (Dwelling) covers the dwelling against direct physical loss unless the loss is excluded in {head} {excl_no}.
2.2 Coverage B (Other structures), Coverage C (Personal property) and Coverage D (Loss of use) are subject to the limits in the Declarations. Personal property is adjusted separately from the dwelling.

{head} 3 DEDUCTIBLE
3.1 An all-perils deductible of {m(p.deductible)} applies to each occurrence under Coverage A.
3.2 A separate windstorm and hail deductible of 2% of the Coverage A limit applies only to loss caused by windstorm or hail.

{head} {excl_no} EXCLUSIONS
We do not cover loss caused by any of the following.
{chr(10).join(order)}

{head} {cond_no} CONDITIONS
{cond_no}.1 Notice of loss. Coverage does not apply to a loss that is reported to us more than {p.notice_days} days after the date of discovery.
{cond_no}.2 Duties after loss. You must protect the property from further damage and keep records of repair costs.
{cond_no}.3 Proof of loss. You must send us a signed proof of loss within 60 days after we request it.
{cond_no}.4 Mortgage clause. Covered loss to the dwelling is payable to you and the mortgagee as interests appear.

{head} {sett_no} LOSS SETTLEMENT
{sett_no}.1 Covered loss to the dwelling is settled at replacement cost.
{sett_no}.2 We first subtract the deductible from the amount of the loss and then apply any limit or sublimit that applies.
{sett_no}.3 Concealed Water Sublimit: {m(p.sublimit_old)} per occurrence, for coverage restored by Exception {seep_ref}.1.
{sett_no}.4 Fungi and mould sublimit: {m(5000)} per policy term.

ENDORSEMENTS ISSUED WITH FORM {p.form}
{chr(10).join(endorsements)}"""
    return text, {"seep": seep_ref, "vac": vac_ref, "head": head}


def render_case(p: Policy, c: Case, n: Narrative) -> str:
    """The claim file in one of five narrative styles; facts in Case.missing are left out."""
    d = lambda x: fmt_date(x, n.date_style)
    m = lambda x: fmt_money(x, p.currency)
    who = n.insured
    term_end = add_days(c.renewal, 365)
    facts = []        # (date or None, third-person sentence, first-person sentence or None)

    if "renewal" in c.missing:
        facts.append((None, "The Declarations page for the current policy term has not been received from the agency.",
                      None))
    else:
        facts.append((c.renewal, f"Current policy term: {d(c.renewal)} to {d(term_end)} (renewal). The policy was "
                      f"first issued {n.first_issued_years} years earlier.", None))
    if n.midterm_change and "renewal" not in c.missing:
        change = add_days(c.renewal, max(1, days_between(c.renewal, c.discovery) // 2))
        facts.append((change, f"On {d(change)} a mid-term change added a home-office rider to the policy.", None))

    if c.away_days:
        start = add_days(c.discovery, -c.away_days)
        if "furnished" in c.missing:
            facts.append((start, f"Nobody lived in the house from {d(start)} until the damage was found.",
                          f"Nobody was living in the house from {d(start)} until we found the damage."))
        elif c.furnished:
            facts.append((start, f"{who} was away from {d(start)} until the damage was found, caring for a relative. "
                          f"Furniture, clothing and a stocked pantry stayed in the house.",
                          f"I was away from {d(start)} caring for my mother. Everything stayed in the house: "
                          f"furniture, clothes, food in the pantry."))
        else:
            facts.append((start, f"The house was emptied ahead of a planned sale on {d(start)}: all furniture went "
                          f"into storage and nobody lived there from that date.",
                          f"We emptied the house on {d(start)} because we were selling it. All the furniture went "
                          f"into storage and nobody lived there after that."))
    else:
        facts.append((None, f"{who} and the household lived in the house throughout.",
                      "We were living in the house the whole time."))

    if "noticed_on" in c.missing:
        facts.append((None, "We have not yet asked the insured about any earlier signs of water.", None))
    elif c.noticed_on:
        facts.append((c.noticed_on, f"{who} said that around {d(c.noticed_on)} a small water stain appeared in the "
                      f"{n.room}, which {who.split()[0]} took for an old mark.",
                      f"Around {d(c.noticed_on)} I did see a small water stain in the {n.room}, but I thought it "
                      f"was an old mark."))
    elif c.concealed:
        facts.append((None, f"{who} saw no stain, damp patch or other sign of water before the damage was found.",
                      "Before that day I never saw a stain or a damp patch anywhere."))

    location = n.hidden if c.concealed else n.exposed
    facts.append((c.discovery, f"On {d(c.discovery)} {who} found water damage in the {n.room}. {n.plumber} traced it "
                  f"to {n.cause} {location}. The main water valve was shut off the same day.",
                  f"On {d(c.discovery)} I found the damage in the {n.room} and turned the water off at the main."))
    if "leak_days" in c.missing:
        facts.append((None, f"{n.plumber} has not yet given an opinion on how long the water had been leaking.", None))
    elif c.leak_days == (0, 0):
        facts.append((None, f"{n.plumber}: the leak started suddenly and water ran for a few hours before it was "
                      f"found.", None))
    else:
        lo, hi = c.leak_days
        span = (f"about {lo // 7} to {hi // 7} weeks" if lo % 7 == 0 and hi % 7 == 0 and lo >= 14
                else f"between {lo} and {hi} days")
        facts.append((None, f"{n.plumber}: corrosion and staining at the joint show the leak had been active for "
                      f"{span} before it was found.", None))

    facts.append((c.reported, f"The loss was reported to {p.insurer} on {d(c.reported)}.",
                  f"I phoned it in on {d(c.reported)}."))
    facts.append((None, f"Building repair estimate from {n.builder}: {m(c.estimate)} (drywall, framing, flooring and "
                  f"decoration). A contents claim of {m(n.contents_claim)} is handled separately.", None))
    facts.append((None, f"A hail claim in {n.prior_claim_year} was paid in full and is closed.", None))
    if n.draft_endorsement:
        facts.append((None, f"Underwriting has circulated draft endorsement {n.draft_endorsement}, which would extend "
                      f"the repeated seepage period by 14 days. It has not been approved and is not part of any "
                      f"policy.", None))

    header = (f"Claim {n.claim} | Policy {n.policy_no} | Insured: {who} | Risk address: {n.address}\n"
              f"Peril: water damage, building portion. Adjuster: {n.adjuster}.")
    rng = random.Random(n.seed * 7 + 2)
    if n.style == "memo":
        body = " ".join(s for _, s, _ in facts)
        return f"ADJUSTER FILE NOTE\n{header}\n\n{body}"
    if n.style == "record":
        return f"CLAIM RECORD\n{header}\n" + "\n".join(f"- {s}" for _, s, _ in facts)
    if n.style == "email":
        half = len(facts) // 2
        a = " ".join(s for _, s, _ in facts[:half])
        b = " ".join(s for _, s, _ in facts[half:])
        return (f"From: {n.adjuster}\nTo: Coverage review\nSubject: {n.claim}, building water damage\n\n{header}\n\n"
                f"{a}\n\n--- reply from the field adjuster ---\n{b}")
    if n.style == "timeline":
        dated = sorted((x for x in facts if x[0]), key=lambda x: x[0])
        undated = [x for x in facts if not x[0]]
        rows = [f"{x[0].isoformat()}: {x[1]}" for x in dated] + [f"Note: {x[1]}" for x in undated]
        return f"CLAIM TIMELINE\n{header}\n" + "\n".join(rows)
    # transcript: the insured's own statements in the first person, the rest as the adjuster's summary
    qa = [x for x in facts if x[2]]
    rest = [x for x in facts if not x[2]]
    rng.shuffle(rest)
    lines = "\n".join(f"Q: Please describe what happened.\nA: {x[2]}" if i == 0 else f"Q: Anything else?\nA: {x[2]}"
                      for i, x in enumerate(qa))
    summary = " ".join(x[1] for x in rest)
    return f"RECORDED STATEMENT AND FILE SUMMARY\n{header}\n\nStatement of {who}:\n{lines}\n\nFile summary: {summary}"


# ------------------------------------------------------------------ items

def option_text(label: str, p: Policy) -> str:
    m = lambda x: fmt_money(x, p.currency)
    if label.startswith("pay_subject"):
        cap = int(label.split("_")[3])
        return f"Pay the repair estimate less the deductible, capped at a Concealed Water Sublimit of {m(cap)}."
    return {
        DENY_NOTICE: "Deny: the loss was reported later than the notice condition allows.",
        DENY_VACANCY: "Deny: the vacancy exclusion applies.",
        DENY_SEEPAGE: "Deny: the repeated seepage or leakage exclusion applies and no exception restores coverage.",
        PAY_FULL: "Pay the full repair estimate less the deductible, with no sublimit.",
        MISSING: "Do not settle yet: the file lacks a fact needed to decide the settlement.",
    }[label]


def likely_trap(p: Policy, c: Case, gold: str) -> str:
    """The wrong label a careless reading most likely gives."""
    if gold.startswith("pay_subject"):
        return pay_capped(p.sublimit_old if sublimit(p, c) == p.sublimit_new else p.sublimit_new)
    if gold == DENY_SEEPAGE:
        return pay_capped(sublimit(p, c))
    if gold == PAY_FULL:
        return pay_capped(sublimit(p, c))
    if gold == DENY_VACANCY:
        return DENY_SEEPAGE if is_seepage(p, c) and not is_restored(c) else PAY_FULL
    if gold == DENY_NOTICE:
        return pay_capped(sublimit(p, c)) if is_seepage(p, c) else PAY_FULL
    return pay_capped(p.sublimit_new)


def main_options(rng: random.Random, p: Policy, gold: str, trap: str, with_missing: bool) -> dict[str, str]:
    universe = [DENY_NOTICE, DENY_VACANCY, DENY_SEEPAGE, PAY_FULL, pay_capped(p.sublimit_old),
                pay_capped(p.sublimit_new)]
    chosen = {gold, trap}
    if with_missing:
        chosen.add(MISSING)
    rest = [x for x in universe if x not in chosen]
    rng.shuffle(rest)
    chosen.update(rest[:max(0, rng.randint(4, 6) - len(chosen))])
    return {label: option_text(label, p) for label in chosen}


QUESTIONS = [
    "Acting as the coverage reviewer, decide how the building water-damage portion of claim {claim} must be settled "
    "under the policy form and the endorsements issued with it.",
    "Under form {form} and its endorsements, what is the correct settlement of the building water damage in claim "
    "{claim}?",
    "Apply the policy as in force for the current term and select the settlement of the building portion of claim "
    "{claim}.",
]


def build_items(rng: random.Random, p: Policy, c: Case, n: Narrative, group: str, role: str) -> list[dict]:
    policy_text, _ = render_policy(p, n, c.discovery)
    state = f"{policy_text}\n\n{render_case(p, c, n)}"
    gold, amount = decide(p, c)
    rationale = explain(p, c, n)
    base = dict(COMMON, state=state, group=group, style=n.style, rationale=f"[{role}] {rationale}")
    q = QUESTIONS[n.question_variant].format(claim=n.claim, form=p.form)
    trap = likely_trap(p, c, gold)
    items = [choice_item(rng=rng, options=main_options(rng, p, gold, trap, bool(c.missing) or rng.random() < 0.5),
                         expected=gold, trap=trap, instructions=q, **base)]
    if c.missing:
        items.append(noul_item(
            instructions=f"Does the file for claim {n.claim} contain every fact needed to decide the settlement of the "
                         f"building water damage?",
            true_text="The file is sufficient to decide the settlement.",
            false_text="A fact needed to decide the settlement is missing from the file.",
            answer=gold != MISSING, **base))
        return items
    extra = rng.random()
    if extra < 0.25:
        candidates = {amount, c.estimate, c.estimate - p.deductible, p.sublimit_old, p.sublimit_new,
                      max(0, sublimit(p, c) - p.deductible), 0}
        others = sorted(x for x in candidates if x != amount)
        rng.shuffle(others)
        picked = [amount] + others[:rng.randint(3, 4)]
        opts = {f"amount_{x}": (f"Pay {fmt_money(x, p.currency)}." if x else "Pay nothing.") for x in picked}
        items.append(choice_item(rng=rng, options=opts, expected=f"amount_{amount}", trap=None,
                                 instructions=f"What amount must {p.insurer} pay for the building water damage in "
                                              f"claim {n.claim}?", **base))
    elif extra < 0.5:
        items.append(noul_item(
            instructions=f"Is the building water damage in claim {n.claim} covered, in whole or in part, under the "
                         f"policy as endorsed?",
            true_text="Some amount is payable for the building water damage.",
            false_text="Nothing is payable for the building water damage.", answer=amount > 0, **base))
    elif extra < 0.7:
        level = 0 if amount == 0 else (1 if gold.startswith("pay_subject") else 2)
        items.append(score_item(
            instructions=f"Rate the correct settlement of the building water damage in claim {n.claim}. Output the "
                         f"level index.",
            levels=["Denied: nothing is payable", "Paid, capped by the Concealed Water Sublimit",
                    "Paid in full less the deductible"], expected=level, **base))
    return items


MUTATIONS = ("renewal", "noticed_on", "concealed", "furnished", "leak_days", "reported", "away_days")


def mutate(rng: random.Random, p: Policy, c: Case, what: str) -> Case | None:
    if what == "renewal":
        return replace(c, renewal=add_days(p.effective, -rng.randint(1, 60)) if c.renewal >= p.effective
                       else add_days(p.effective, rng.randint(0, 20)))
    if what == "noticed_on":
        if c.noticed_on:
            return replace(c, noticed_on=None)
        window = sign_window(p, c)
        return replace(c, noticed_on=add_days(window[0], rng.randint(0, days_between(*window)))) if window else None
    if what == "concealed":
        return replace(c, concealed=not c.concealed)
    if what == "furnished":
        return replace(c, furnished=not c.furnished) if c.away_days else None
    if what == "leak_days":
        if is_seepage(p, c):
            short = (max(1, p.seepage_days - 9), p.seepage_days - 2)
            if c.noticed_on and days_between(c.noticed_on, c.discovery) >= short[0]:
                return None          # the earlier sign would predate the shorter leak
            return replace(c, leak_days=short)
        return replace(c, leak_days=(p.seepage_days + 4, p.seepage_days + 17))
    if what == "reported":
        lag = days_between(c.discovery, c.reported)
        new_lag = p.notice_days + rng.randint(1, 20) if lag <= p.notice_days else rng.randint(0, p.notice_days)
        return replace(c, reported=add_days(c.discovery, new_lag))
    if what == "away_days":
        if not c.away_days or c.furnished:
            return None
        return replace(c, away_days=p.vacancy_days + rng.randint(1, 60) if c.away_days <= p.vacancy_days
                       else rng.randint(1, p.vacancy_days))
    raise KeyError(what)


def valid(p: Policy, c: Case) -> bool:
    """One defensible answer, and a sublimit that binds whenever it is in play."""
    try:
        labels = {label for label, _ in outcomes(p, c)}
    except Ambiguous:
        return False
    if len(labels) > 1 and all(label.startswith("deny") for label in labels):
        return False          # denied either way, on a ground the file cannot fix: no single defensible label
    if c.estimate - p.deductible <= max(p.sublimit_old, p.sublimit_new) and (
            is_seepage(p, c) or "leak_days" in c.missing):
        return False
    if c.noticed_on and days_between(c.noticed_on, c.reported) > p.notice_days and \
            days_between(c.discovery, c.reported) <= p.notice_days:
        return False          # notice counted from the earlier sign would be late: two readings, two answers
    if c.noticed_on and "leak_days" in c.missing:
        return False          # the sign itself bounds the leak duration: two readings of a missing duration
    return True


def generate_case(seed: int) -> list[dict]:
    """All items of one case: the base item, a twin that flips the answer, and sometimes a missing-fact variant."""
    rng = random.Random(seed)
    for _ in range(50):
        p, c = sample_case(rng, sample_policy(rng))
        if valid(p, c):
            break
    else:
        return []
    n = sample_narrative(rng, seed)
    group = f"dg-{FAMILY}-{stable_id(GENERATOR, str(seed))}"
    items = build_items(rng, p, c, n, group, "base")

    gold = decide(p, c)[0]
    for what in rng.sample(MUTATIONS, len(MUTATIONS)):
        twin = mutate(rng, p, c, what)
        if twin is not None and valid(p, twin) and decide(p, twin)[0] != gold:
            items += build_items(rng, p, twin, n, group, f"twin:{what}")
            break

    if rng.random() < 0.35:
        for fact in rng.sample(FACTS, len(FACTS)):
            gap = replace(c, missing=frozenset({fact}))
            if valid(p, gap):
                items += build_items(rng, p, gap, n, group, f"missing:{fact}")
                break
    return items
