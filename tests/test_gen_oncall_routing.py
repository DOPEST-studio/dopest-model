"""oncall_routing: every answer is recomputed from the RENDERED HANDBOOK (tables parsed with regexes) with a separate
implementation of the hops, never with the generator's engine."""
import re
from datetime import datetime, timedelta

import pytest

from dopest_model.gen import oncall_routing as oc


def parse(state: str) -> dict:
    off = int(re.search(r"is UTC([+-]\d+) \(", state).group(1))
    handover = int(re.search(r"hand over daily at (\d{2}):00 local time", state).group(1))
    sev = {}
    for rule, base, t1, t2, t3 in re.findall(r"^\| (\w+)\s+\| SEV(\d)\s+\| (\w+ ?\w* ?\w*)\s+\| (\w+ ?\w*)\s+\| (\w+ ?\w* ?\w*)\s+\|$", state, re.M):
        sev[rule] = {1: t1, 2: t2, 3: t3, "base": int(base)}
    catalog = {}
    for service, procs, team, tier in re.findall(r"^\| ([a-z-]+)\s+\| ([a-z0-9, -]+?)\s+\| ([A-Za-z-]+)\s+\| tier-(\d) \|$", state, re.M):
        for p in procs.split(", "):
            catalog[p.strip()] = (service, team, int(tier))
    transfers = dict(re.findall(r"^\| ([A-Za-z-]+)\s+\| all of this team's services were handed over to ([A-Za-z-]+) on", state, re.M))
    rotation = {t: (a, b) for t, a, b in re.findall(r"^\| ([A-Za-z-]+)\s+\| (\w+)\s+\| (\w+)\s+\|$", state, re.M)}
    overrides = [(t, datetime.strptime(s, "%Y-%m-%d %H:%M"), datetime.strptime(e, "%Y-%m-%d %H:%M"), p)
                 for t, s, e, p in re.findall(r"^\| ([A-Za-z-]+)\s+\| (\d{4}-\d\d-\d\d \d\d:\d\d) \| (\d{4}-\d\d-\d\d \d\d:\d\d) \| (\w+) \|$", state, re.M)]
    ts = re.search(r"Alertmanager, (\d{4}-\d\d-\d\dT\d\d:\d\d):12Z", state).group(1)
    alert_utc = datetime.strptime(ts, "%Y-%m-%dT%H:%M")
    rule = re.search(r"rule: (\w+)", state).group(1)
    process = re.search(r'process="([^"]+)"', state).group(1)
    return dict(off=off, handover=handover, sev=sev, catalog=catalog, transfers=transfers, rotation=rotation,
                overrides=overrides, alert_utc=alert_utc, rule=rule, process=process)


def route(h: dict):
    service, team, tier = h["catalog"][h["process"]]
    cell = h["sev"][h["rule"]][tier]
    severity = int(re.search(r"SEV(\d)", cell).group(1))
    if severity > 2:
        return severity, None
    team = h["transfers"].get(team, team)
    local = h["alert_utc"] + timedelta(hours=h["off"])
    day = (local - timedelta(hours=h["handover"])).day
    primary = h["rotation"][team][0 if day % 2 == 0 else 1]
    for t, s, e, p in h["overrides"]:
        if t == team and s <= local < e:
            primary = p
    return severity, primary


@pytest.mark.parametrize("items", [oc.generate_case(seed) for seed in range(200)],
                         ids=lambda its: its[0]["group"] if its else "empty")
def test_routing_recomputed_from_the_rendered_handbook(items):
    assert items
    for it in items:
        h = parse(it["state"])
        severity, primary = route(h)
        q = it["question"]["type"]
        if q == "score":
            assert it["expected"] == severity - 1
        elif q == "noul":
            assert (it["expected"] == "yes") == (severity <= 2)
        else:
            assert it["expected"] == (primary.lower() if primary else "nobody_ticket_only")
            if it["trap_answer"]:
                assert it["trap_answer"] != it["expected"] and it["trap_answer"] in it["labels"]


def test_twin_flips_the_paged_person_and_shares_the_group():
    flipped = 0
    for seed in range(120):
        items = oc.generate_case(seed)
        choice = [it for it in items if it["question"]["type"] == "choice"]
        if len(choice) == 2:
            assert choice[0]["group"] == choice[1]["group"] and choice[0]["expected"] != choice[1]["expected"]
            flipped += 1
    assert flipped >= 60
