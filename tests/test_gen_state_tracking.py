"""state_tracking: the final state is replayed from the RENDERED LOG with a separate implementation of the OPS-4
rules; probe and summary answers are checked against that replay."""
import re
from datetime import datetime

import pytest

from dopest_model.gen import state_tracking as st

LINE = re.compile(r"^(\d\d \w{3} \d\d:\d\d)  (.+)$", re.M)


def parse_event(text: str):
    if m := re.match(r"customer asks to add (\d+) unit", text): return "add", int(m.group(1))
    if m := re.match(r"customer asks to remove (\d+) unit", text): return "remove", int(m.group(1))
    if m := re.match(r"customer asks to change the delivery address to '([^']+)'", text): return "address", m.group(1)
    if text.startswith("payment of"): return "pay", None
    if text.startswith("warehouse marks"): return "pack", None
    if text.startswith("carrier picks"): return "ship", None
    if text.startswith("carrier confirms"): return "deliver", None
    if text.startswith("customer asks to cancel"): return "cancel", None
    if m := re.match(r"finance requests a refund of (\d+) EUR", text): return "refund", int(m.group(1))
    raise AssertionError(text)


def replay(events):
    status, units, address, refund = "created", 0, "home", 0
    applied = []
    for kind, arg in events:
        ok = False
        if kind == "pay" and status == "created": status, ok = "paid", True
        elif kind == "add" and status in ("created", "paid"): units, ok = units + arg, True
        elif kind == "remove" and status in ("created", "paid") and units - arg >= 1: units, ok = units - arg, True
        elif kind == "address" and status in ("created", "paid"): address, ok = arg, True
        elif kind == "pack" and status == "paid": status, ok = "packed", True
        elif kind == "ship" and status == "packed": status, ok = "shipped", True
        elif kind == "deliver" and status == "shipped": status, ok = "delivered", True
        elif kind == "cancel" and status in ("created", "paid", "packed"): status, ok = "cancelled", True
        elif kind == "refund" and status in ("cancelled", "delivered") and refund == 0: refund, ok = arg, True
        applied.append(ok)
    return status, units, address, refund, applied


@pytest.mark.parametrize("items", [st.generate_case(seed) for seed in range(200)],
                         ids=lambda its: its[0]["group"] if its else "empty")
def test_state_recomputed_from_the_rendered_log(items):
    assert items
    for it in items:
        state = it["state"]
        log = state.split("Request log (chronological, local time):\n", 1)[1].split("\n\n", 1)[0]
        rows = LINE.findall(log)
        events = [parse_event(text) for _, text in rows]
        status, units, address, refund, applied = replay(events)
        q = it["question"]
        if q["type"] == "choice":
            assert it["expected"] == status
        elif q["type"] == "score":
            assert it["expected"] == min(3, units - 1)
        elif "logged at" in q["instructions"]:
            stamp = re.search(r"logged at (\d\d \w{3} \d\d:\d\d)", q["instructions"]).group(1)
            idx = [s for s, _ in rows].index(stamp)
            assert (it["expected"] == "yes") == applied[idx]
        else:
            note = re.search(r"ended up '(\w+)' with (\d+) unit\(s\), address '([^']+)'(?:, refund (\d+) EUR)?", state)
            n_status, n_units, n_addr, n_refund = note.group(1), int(note.group(2)), note.group(3), int(note.group(4) or 0)
            assert (it["expected"] == "yes") == ((n_status, n_units, n_addr, n_refund) == (status, units, address, refund))
