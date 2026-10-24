"""Multi-hop routing through a handbook (JevBench hard tier, family multi_hop), wording our own.

Shape taken from the public items: an on-call handbook with numbered sections (how an alert becomes a page, a
severity matrix by service tier, a service catalog mapping process names to services and teams, a team directory
with hand-overs, rotation hand-over times in a local zone, override windows), then one alert with a UTC timestamp
and a chat exchange that routes it wrongly. Resolving the alert takes five or six hops; the engine does them and the
rendered tables carry every fact the hops need.

Questions: who must be paged (choice over the rotation members plus "nobody, ticket only"), does the alert page
anyone (yes/no), and the adjusted severity as an ordered level (score). Twins change one fact that moves the answer:
the tier of the service, a hand-over time, or an override window.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta

from .common import SPLIT_BY_STYLE, Nar, choice_item, noul_item, render_file, score_item, stable_id

FAMILY = "multi_hop"
GENERATOR = "oncall_routing_v1"

RULES = ["ProcCrashLoop", "ErrorRate5xx", "DiskPressure", "CertExpiry7d", "ReplicaLag", "QueueBacklog", "LatencyP99"]
BASE_SEV = {"ProcCrashLoop": 3, "ErrorRate5xx": 2, "DiskPressure": 3, "CertExpiry7d": 3, "ReplicaLag": 2,
            "QueueBacklog": 3, "LatencyP99": 2}
# tier adjustments: tier-1 raises by one (min SEV1), tier-2 keeps, tier-3 lowers by one (max SEV4); CertExpiry fixed
SERVICES = [("edge-router", ["envoyd", "edge-sidecar"]), ("identity-api", ["authd"]), ("report-export", ["rptexp"]),
            ("card-vault", ["vaultd"]), ("storefront-web", ["sf-next"]), ("risk-scorer", ["riskd", "score-py"]),
            ("fx-feed", ["fxpoll"]), ("ledger-core", ["ledgerd", "ledger-migrate"]), ("ledger-reports", ["ledgerrpt"]),
            ("notify-hub", ["notifyd"]), ("payout-runner", ["payoutd"]), ("recon-batch", ["recond"]),
            ("search-index", ["idxd"]), ("session-cache", ["sess-redis"]), ("statement-gen", ["stmtgen"])]
TEAMS = ["Edge", "Identity", "Data-Finance", "Payments-Core", "Storefront", "Risk", "Messaging", "Money-Movement",
         "Discovery"]
PEOPLE = ["Ana", "Bjorn", "Chen", "Dmitri", "Esra", "Farah", "Gustav", "Hana", "Ilya", "Jonas", "Kaia", "Luca"]
ZONES = [("Europe/Berlin", 2, "CEST"), ("Europe/Warsaw", 2, "CEST"), ("Europe/London", 1, "BST"),
         ("America/New_York", -4, "EDT"), ("Asia/Singapore", 8, "SGT"), ("Australia/Sydney", 10, "AEST")]


@dataclass(frozen=True)
class World:
    services: tuple          # (service, processes tuple, team, tier)
    transfers: tuple         # (from_team, to_team): services of from_team are now paged to to_team
    zone: tuple              # (name, utc offset, abbr)
    handover_hour: int       # rotation hand-over, local time, daily
    rotation: dict           # team -> ((weekday_group_a members), (group_b members)): two rotation slots A/B
    overrides: tuple         # (team, start_utc, end_utc, person)
    alert_rule: str
    alert_process: str
    alert_utc: datetime


def resolve(w: World) -> dict:
    """The hops: process -> service -> tier -> severity -> page or ticket -> team -> transfer -> slot -> override."""
    svc = next(s for s in w.services if w.alert_process in s[1])
    service, _, team, tier = svc
    sev = BASE_SEV[w.alert_rule]
    if w.alert_rule != "CertExpiry7d":
        sev = max(1, sev - 1) if tier == 1 else min(4, sev + 1) if tier == 3 else sev
    pages = sev <= 2
    team_paged = dict(w.transfers).get(team, team)
    local = w.alert_utc + timedelta(hours=w.zone[1])
    # slot A covers from the hand-over on even calendar days, slot B on odd days (hand-over day = local date)
    shifted = local - timedelta(hours=w.handover_hour)          # the rotation day starts at the hand-over
    slot = "A" if shifted.date().day % 2 == 0 else "B"
    primary = w.rotation[team_paged][0 if slot == "A" else 1][0]
    for t, start, end, person in w.overrides:
        if t == team_paged and start <= w.alert_utc < end:
            primary = person
    return {"service": service, "tier": tier, "sev": sev, "pages": pages, "team": team, "team_paged": team_paged,
            "slot": slot, "primary": primary if pages else None, "local": local}


def sample(rng: random.Random) -> World | None:
    svcs = rng.sample(SERVICES, rng.randint(9, 12))
    teams = rng.sample(TEAMS, rng.randint(5, 6))
    services = tuple((name, tuple(procs), rng.choice(teams), rng.choice([1, 1, 2, 2, 3])) for name, procs in svcs)
    used_teams = sorted({s[2] for s in services})
    if len(used_teams) < 4:
        return None
    transfers = ()
    if rng.random() < 0.7:
        a, b = rng.sample(used_teams, 2)
        transfers = ((a, b),)
    people = rng.sample(PEOPLE, 2 * len(used_teams))
    rotation = {t: ((people[2 * i],), (people[2 * i + 1],)) for i, t in enumerate(used_teams)}
    zone = rng.choice(ZONES)
    handover = rng.choice([8, 9, 10, 18])
    day = date(2026, rng.randint(3, 10), rng.randint(2, 27))
    alert_utc = datetime(day.year, day.month, day.day, rng.randint(0, 23), rng.choice([5, 12, 30, 41, 55]))
    svc = rng.choice(services)
    rule = rng.choice(RULES)
    overrides = ()
    team_paged = dict(transfers).get(svc[2], svc[2])
    if rng.random() < 0.5:
        other = rng.choice([p for p in people if p not in rotation[team_paged][0] + rotation[team_paged][1]])
        start = alert_utc - timedelta(hours=rng.randint(1, 20)) if rng.random() < 0.6 else alert_utc + timedelta(hours=rng.randint(1, 8))
        end = start + timedelta(hours=rng.choice([8, 12, 24]))
        overrides = ((team_paged, start.replace(minute=0), end.replace(minute=0), other),)
    return World(services, transfers, zone, handover, rotation, overrides, rule, rng.choice(svc[1]), alert_utc)


def mutate(rng: random.Random, w: World) -> World | None:
    kind = rng.choice(["tier", "handover", "override", "time"])
    if kind == "tier":
        services = []
        for s in w.services:
            if w.alert_process in s[1]:
                services.append((s[0], s[1], s[2], rng.choice([t for t in (1, 2, 3) if t != s[3]])))
            else:
                services.append(s)
        return replace(w, services=tuple(services))
    if kind == "handover":
        return replace(w, handover_hour=rng.choice([h for h in (8, 9, 10, 18) if h != w.handover_hour]))
    if kind == "override" and w.overrides:
        t, start, end, person = w.overrides[0]
        shift = timedelta(hours=rng.choice([-30, -26, 26, 30]))
        return replace(w, overrides=((t, start + shift, end + shift, person),))
    return replace(w, alert_utc=w.alert_utc + timedelta(hours=rng.choice([-14, -10, 10, 14])))


def render(w: World, n: Nar, r: random.Random) -> tuple[str, dict]:
    company = r.choice(["Fernway Pay", "Harbor Ledger", "Kestrel Payments", "Nimbus Commerce", "Orbital Retail"])
    zone, off, abbr = w.zone
    sev_rows = []
    for rule in RULES:
        b = BASE_SEV[rule]
        if rule == "CertExpiry7d":
            adj = [f"stays SEV{b}"] * 3
        else:
            adj = [f"raise to SEV{max(1, b - 1)}", f"stays SEV{b}", f"lower to SEV{min(4, b + 1)}"]
        sev_rows.append(f"| {rule:14s} | SEV{b}          | {adj[0]:14s} | {adj[1]:12s} | {adj[2]:14s} |")
    catalog = "\n".join(f"| {s:16s} | {', '.join(p):26s} | {t:15s} | tier-{tier} |" for s, p, t, tier in sorted(w.services))
    teams = sorted({s[2] for s in w.services})
    directory = []
    for t in teams:
        moved = [a for a, b in w.transfers if a == t]
        gained = [a for a, b in w.transfers if b == t]
        note = (f"all of this team's services were handed over to {dict(w.transfers)[t]} on 1 {w.alert_utc:%B}; page "
                f"{dict(w.transfers)[t]}'s rotation for them" if moved else
                f"also covers the services handed over from {gained[0]}" if gained else "no changes")
        directory.append(f"| {t:15s} | {note} |")
    rota = "\n".join(f"| {t:15s} | {w.rotation[t][0][0]:8s} | {w.rotation[t][1][0]:8s} |" for t in teams)
    ov = "\n".join(f"| {t:15s} | {s + timedelta(hours=off):%Y-%m-%d %H:%M} | {e + timedelta(hours=off):%Y-%m-%d %H:%M} | {p} |"
                   for t, s, e, p in w.overrides) or "| (none)          |                  |                  |        |"
    res = resolve(w)
    # the chat routes it wrongly: a plausible slip (ignoring the transfer, the override, or the zone conversion)
    wrong = []
    if w.transfers and res["team"] != res["team_paged"]:
        wrong.append(w.rotation[res["team"]][0 if res["slot"] == "A" else 1][0])
    other_slot = w.rotation[res["team_paged"]][1 if res["slot"] == "A" else 0][0]
    wrong.append(other_slot)
    if res["pages"] is False:
        wrong.append(w.rotation[res["team_paged"]][0][0])
    chat_name = next((x for x in wrong if x != res["primary"]), other_slot)
    chat = (f'Chat, #ops: "{w.alert_process}... that is {res["service"]}, {res["team"]}. {chat_name} has the pager '
            f'this week, I think." — "Looks right to me."')
    text = (f"{company.upper()}: ON-CALL ROUTING HANDBOOK (SRE), revision {r.randint(9, 19)}\n\n"
            f"1. FROM ALERT TO PAGE\n"
            f"1.1 An alert names the failing PROCESS and host, not the service. Find the service through the catalog "
            f"(section 3); a service may run several processes.\n"
            f"1.2 The rule's base severity is adjusted by the service tier (section 2). Only SEV1 and SEV2 page a "
            f"person; SEV3 and SEV4 open a ticket for the owning team and page nobody, whatever the hour.\n"
            f"1.3 The page goes to the primary of the team that owns the service, after checking the directory "
            f"(section 4) for hand-overs: a handed-over service pages the receiving team's rotation.\n"
            f"1.4 Rotations hand over daily at {w.handover_hour:02d}:00 local time ({zone}). The rotation day that "
            f"starts at a hand-over on an even-numbered calendar day is covered by slot A, on an odd-numbered day by "
            f"slot B. Alert timestamps are UTC; {zone} is UTC{off:+d} ({abbr}) throughout {w.alert_utc:%B %Y}.\n"
            f"1.5 Overrides (section 6) replace the scheduled primary inside [start, end), given in local time.\n"
            f"1.6 Secondaries are never paged directly; the tool escalates after 10 minutes.\n\n"
            f"2. SEVERITY MATRIX\n"
            f"| Rule           | Base severity | tier-1 service | tier-2       | tier-3 service |\n"
            f"|----------------|---------------|----------------|--------------|----------------|\n" + "\n".join(sev_rows) +
            f"\n\n3. SERVICE CATALOG (production)\n"
            f"| Service          | Processes                  | Owning team     | Tier   |\n"
            f"|------------------|----------------------------|-----------------|--------|\n{catalog}\n\n"
            f"4. TEAM DIRECTORY\n| Team            | Note |\n|-----------------|------|\n" + "\n".join(directory) +
            f"\n\n5. ROTATIONS (primary per slot)\n| Team            | Slot A   | Slot B   |\n"
            f"|-----------------|----------|----------|\n{rota}\n\n"
            f"6. OVERRIDES (local time, {zone})\n| Team            | Start            | End              | Covers |\n"
            f"|-----------------|------------------|------------------|--------|\n{ov}\n\n"
            f"7. OTHER RULES\n7.1 Swaps must be logged within 7 days. 7.2 If the paging tool is down, call the number "
            f"on the rotation wiki page.\n\n"
            f"8. ALERT TO ROUTE\nAlertmanager, {w.alert_utc:%Y-%m-%dT%H:%M}:12Z\n  rule: {w.alert_rule}\n"
            f"  labels: process=\"{w.alert_process}\", host=\"{r.choice(['pay-eu', 'core-us', 'edge-ap'])}-{r.randint(2, 9)}\", env=\"prod\"\n")
    state = text + "\n" + render_file(n.style, "Channel log", "Shortly after the alert", [(w.alert_utc.date(), chat)])
    return state, res


def items(rng: random.Random, w: World, n: Nar, base: dict) -> list[dict]:
    r = n.rng(21)
    state, res = render(w, n, r)
    teams = sorted({s[2] for s in w.services})
    names = sorted({p for t in teams for slot in w.rotation[t] for p in slot} | {p for _, _, _, p in w.overrides})
    cands = [res["primary"]] if res["primary"] else []
    pool = [x for x in names if x not in cands]
    r2 = n.rng(22)
    r2.shuffle(pool)
    cands += pool[:4 if res["primary"] else 5]
    options = {c.lower(): f"Page {c}." for c in cands}
    options["nobody_ticket_only"] = "Nobody is paged; a ticket is opened for the owning team."
    gold = res["primary"].lower() if res["primary"] else "nobody_ticket_only"
    chat_name = state.split("I think.")[0].split(". ")[-1].split(" has the pager")[0].strip()
    trap = chat_name.lower() if chat_name.lower() in options and chat_name.lower() != gold else None
    why = (f"{w.alert_process} -> {res['service']} (tier-{res['tier']}, team {res['team']}); severity SEV{res['sev']} "
           f"({'pages' if res['pages'] else 'ticket only'}); paged team {res['team_paged']}; local time "
           f"{res['local']:%Y-%m-%d %H:%M}, slot {res['slot']}; primary {res['primary']}.")
    common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
    return [choice_item(rng=rng, options=options, expected=gold, trap=trap,
                        instructions="Following the handbook, who must be paged for this alert, if anyone?", **common),
            noul_item(instructions="Under the handbook, does this alert page a person at all?",
                      true_text="The adjusted severity is SEV1 or SEV2, so a primary is paged.",
                      false_text="The adjusted severity is SEV3 or SEV4: ticket only.", answer=res["pages"], **common),
            score_item(instructions="What is the alert's severity after the tier adjustment? Output the level index "
                                    "(0 = SEV1, most severe; 3 = SEV4).",
                       levels=["SEV1", "SEV2", "SEV3", "SEV4"], expected=res["sev"] - 1, **common)]


def generate_case(seed: int) -> list[dict]:
    rng = random.Random(seed)
    w = None
    for _ in range(20):
        w = sample(rng)
        if w is not None:
            break
    if w is None:
        return []
    n = Nar(seed=seed, style=rng.choice(list(SPLIT_BY_STYLE)), ds="iso")
    group = f"dg-{FAMILY}-{stable_id(GENERATOR, str(seed))}"
    base = {"family": FAMILY, "generator": GENERATOR, "use_case": "model_routing", "topic": "support_ops",
            "group": group, "style": n.style}
    out = items(rng, w, n, dict(base, rationale="[base:oncall]"))
    gold = resolve(w)["primary"]
    for _ in range(8):
        t = mutate(rng, w)
        if t is not None and resolve(t)["primary"] != gold:
            out += items(rng, t, n, dict(base, rationale="[twin:oncall]"))
            break
    return out
