"""State tracking: the final state of an object after a log of requests, some of which the rules reject.

JevBench v1.5 added a state-tracking family to every question type (at least 12 % of the sealed set); its public
items are not in the repository, so this generator follows only the idea: an entity (an order) with a status
machine and a few fields, a timestamped log of 8-14 requests, rules that reject some of them, one reversal, and a
colleague's summary that counts a rejected request as applied. The engine replays the log; the labels are:
  choice   the final status
  score    the number of units in the order at the end (bucketed)
  noul     whether a given request took effect, or whether the summary is right
Twins change one request so that the final status or the unit count moves.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .common import SPLIT_BY_STYLE, Nar, choice_item, noul_item, render_file, score_item, stable_id

FAMILY = "state_tracking"
GENERATOR = "state_tracking_v1"
STATUSES = ["created", "paid", "packed", "shipped", "delivered", "cancelled"]
STATUS_TEXT = {"created": "Created, awaiting payment.", "paid": "Paid, not yet packed.", "packed": "Packed, not yet "
               "shipped.", "shipped": "Shipped, in transit.", "delivered": "Delivered.", "cancelled": "Cancelled."}


@dataclass(frozen=True)
class Order:
    status: str
    units: int
    address: str
    refund: int


@dataclass(frozen=True)
class World:
    events: tuple            # (kind, arg) in log order
    start: datetime
    unit_price: int


def apply(order: Order, kind: str, arg) -> tuple[Order, bool, str]:
    """One request against the rules; returns (new order, applied?, reason)."""
    s = order.status
    if kind == "pay":
        return (replace(order, status="paid"), True, "") if s == "created" else (order, False, f"payment only from 'created' (was '{s}')")
    if kind == "add":
        if s in ("created", "paid"):
            return replace(order, units=order.units + arg), True, ""
        return order, False, f"items can change only before packing (was '{s}')"
    if kind == "remove":
        if s in ("created", "paid") and order.units - arg >= 1:
            return replace(order, units=order.units - arg), True, ""
        return order, False, ("items can change only before packing" if s not in ("created", "paid") else "an order keeps at least one unit")
    if kind == "address":
        return (replace(order, address=arg), True, "") if s in ("created", "paid") else (order, False, f"address locked once packed (was '{s}')")
    if kind == "pack":
        return (replace(order, status="packed"), True, "") if s == "paid" else (order, False, f"packing needs 'paid' (was '{s}')")
    if kind == "ship":
        return (replace(order, status="shipped"), True, "") if s == "packed" else (order, False, f"shipping needs 'packed' (was '{s}')")
    if kind == "deliver":
        return (replace(order, status="delivered"), True, "") if s == "shipped" else (order, False, f"delivery needs 'shipped' (was '{s}')")
    if kind == "cancel":
        if s in ("created", "paid", "packed"):
            return replace(order, status="cancelled"), True, ""
        return order, False, f"no cancellation after shipping (was '{s}')"
    if kind == "refund":
        if s in ("cancelled", "delivered") and order.refund == 0:
            return replace(order, refund=arg), True, ""
        return order, False, ("refund only for cancelled or delivered orders" if s not in ("cancelled", "delivered") else "a refund was already issued")
    raise ValueError(kind)


def replay(w: World) -> tuple[Order, list[tuple[str, object, bool, str]]]:
    order = Order("created", 0, "home", 0)
    trail = []
    for kind, arg in w.events:
        order, ok, why = apply(order, kind, arg)
        trail.append((kind, arg, ok, why))
    return order, trail


def sample(rng: random.Random) -> World | None:
    units0 = rng.randint(1, 2)
    events = [("add", units0)]
    # a happy path with a few rejected requests sprinkled in and one reversal (remove after add)
    path = rng.choice([["pay", "pack", "ship", "deliver"], ["pay", "pack", "ship", "deliver"], ["pay", "pack", "ship"],
                       ["pay", "pack", "ship"], ["pay", "pack"], ["pay"], ["pay", "pack", "cancel"], ["pay", "cancel"],
                       ["pay", "pack", "ship", "deliver", "refund"], ["pay", "cancel", "refund"]])
    extras = [("add", rng.randint(1, 2)), ("remove", 1), ("address", rng.choice(["office", "pickup point", "neighbour"])),
              ("refund", rng.randint(20, 90)), ("add", 1), ("address", "office"), ("pay", None), ("pack", None),
              ("deliver", None)]
    if rng.random() < 0.35:
        extras.append(("cancel", None))
    rng.shuffle(extras)
    k = rng.randint(4, 7)
    slots = sorted(rng.random() for _ in range(k))
    merged, ei = [], 0
    for i, step in enumerate(path):
        while ei < k and slots[ei] < (i + 1) / (len(path) + 1):
            merged.append(extras[ei]); ei += 1
        merged.append((step, rng.randint(20, 90) if step == "refund" else None))
    merged += extras[ei:k]
    events += merged
    w = World(tuple(events), datetime(2026, rng.randint(2, 11), rng.randint(1, 24), rng.randint(7, 17), rng.choice([0, 15, 30, 45])),
              rng.randint(9, 60))
    _, trail = replay(w)
    if sum(not ok for _, _, ok, _ in trail) < 2 or sum(ok for _, _, ok, _ in trail) < 4:
        return None
    return w


def mutate(rng: random.Random, w: World) -> World | None:
    """Drop or add one request so that the final status or the unit count moves."""
    events = list(w.events)
    i = rng.randrange(1, len(events))
    if rng.random() < 0.5:
        del events[i]
    else:
        events.insert(i, rng.choice([("cancel", None), ("add", rng.randint(1, 3)), ("ship", None), ("remove", 1)]))
    return replace(w, events=tuple(events))


KIND_TEXT = {"pay": "payment of {amt} captured", "add": "customer asks to add {arg} unit(s)",
             "remove": "customer asks to remove {arg} unit(s)", "address": "customer asks to change the delivery address to '{arg}'",
             "pack": "warehouse marks the order packed", "ship": "carrier picks the order up", "deliver": "carrier confirms delivery",
             "cancel": "customer asks to cancel", "refund": "finance requests a refund of {arg} EUR"}


def items(rng: random.Random, w: World, n: Nar, base: dict) -> list[dict]:
    r = n.rng(31)
    shop = r.choice(["Lindqvist Home", "Bramble & Co", "Northwind Outdoor", "Casa Verde", "Orbit Electronics"])
    order_no = f"{r.choice(['ORD', 'SO', 'WEB'])}-{r.randint(10000, 99999)}"
    final, trail = replay(w)
    t = w.start
    facts, lines, stamps = [], [], []
    for (kind, arg, ok, why) in trail:
        t += timedelta(minutes=r.choice([7, 15, 40, 65, 130, 300, 720]))
        amt = w.unit_price * sum(a for k_, a in w.events[:1] if k_ == "add")
        text = KIND_TEXT[kind].format(arg=arg, amt=amt)
        lines.append(f"{t:%d %b %H:%M}  {text}")
        stamps.append(t)
        facts.append((t.date(), text))
    rules = (f"{shop.upper()}: ORDER HANDLING RULES (OPS-4) AND REQUEST LOG FOR {order_no}\n\n"
             f"Statuses move only forward along created → paid → packed → shipped → delivered, one step per request.\n"
             f"OPS-4.1 A request that the current status does not allow is REJECTED and changes nothing; the log "
             f"records every request, applied or not.\n"
             f"OPS-4.2 Units can be added or removed only while the order is 'created' or 'paid'; an order never drops "
             f"below one unit.\n"
             f"OPS-4.3 The delivery address can change only while the order is 'created' or 'paid'.\n"
             f"OPS-4.4 Cancellation is possible until the order is shipped; after shipping it is refused.\n"
             f"OPS-4.5 A refund can be issued once, and only for a cancelled or delivered order.\n"
             f"OPS-4.6 Payment is accepted only for a 'created' order; packing needs 'paid'; shipping needs 'packed'; "
             f"delivery needs 'shipped'.\n\nRequest log (chronological, local time):\n" + "\n".join(lines))
    # the colleague's summary is right 40% of the time; otherwise it treats one rejected request as applied
    rejected = [(k, a) for k, a, ok, _ in trail if not ok]
    summary_right = r.random() < 0.4
    wrong_kind, wrong_arg = (None, None) if summary_right else r.choice(rejected)
    naive = Order("created", 0, "home", 0)
    for (kind, arg, ok, _) in trail:
        if ok:
            naive = apply(naive, kind, arg)[0]
        elif (kind, arg) == (wrong_kind, wrong_arg):
            naive = _force(naive, kind, arg)
    summary = (f"Handover note from the morning shift: \"{order_no} ended up '{naive.status}' with {naive.units} unit(s), "
               f"address '{naive.address}'{', refund ' + str(naive.refund) + ' EUR' if naive.refund else ''}.\"")
    state = rules + "\n\n" + render_file(n.style, "Shift handover", f"Order {order_no}", [(t.date(), summary)])
    why = (f"Applied: {[k for k, _, ok, _ in trail if ok]}; rejected: {[(k, w_) for k, _, ok, w_ in trail if not ok]}. "
           f"Final: {final}. Summary {'right' if summary_right else 'counted ' + repr(wrong_kind) + ' as applied'}.")
    common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
    opts = {s: STATUS_TEXT[s] for s in STATUSES}
    trap = naive.status if naive.status != final.status else None
    out = [choice_item(rng=rng, options=opts, expected=final.status, trap=trap,
                       instructions=f"Applying OPS-4 to the log, what is the status of {order_no} at the end?", **common),
           score_item(instructions=f"How many units does {order_no} contain at the end? Output the level index.",
                      levels=["1 unit", "2 units", "3 units", "4 or more units"], expected=min(3, final.units - 1), **common)]
    pi = r.choice([i for i, x in enumerate(trail) if x[0] in ("address", "cancel", "refund", "remove", "add")] or [0])
    probe = trail[pi]
    out.append(noul_item(instructions=f"Did the request logged at {stamps[pi]:%d %b %H:%M} take effect under OPS-4?",
                         true_text="The status at that moment allowed it, so it was applied.",
                         false_text="The rules rejected it; the order did not change.", answer=probe[2], **common))
    out.append(noul_item(instructions="Is the handover note's description of the order's final state correct?",
                         true_text="Status, units and address in the note all match the state after the log.",
                         false_text="The note counts a rejected request as applied, so at least one detail is wrong.",
                         answer=naive == final, **common))
    return out


def _force(order: Order, kind: str, arg) -> Order:
    """What a careless reader does with a rejected request: applies it anyway."""
    if kind == "pay": return replace(order, status="paid")
    if kind == "add": return replace(order, units=order.units + arg)
    if kind == "remove": return replace(order, units=max(0, order.units - arg))
    if kind == "address": return replace(order, address=arg)
    if kind == "pack": return replace(order, status="packed")
    if kind == "ship": return replace(order, status="shipped")
    if kind == "deliver": return replace(order, status="delivered")
    if kind == "cancel": return replace(order, status="cancelled")
    return replace(order, refund=arg)


def generate_case(seed: int) -> list[dict]:
    rng = random.Random(seed)
    w = None
    for _ in range(30):
        w = sample(rng)
        if w is not None:
            break
    if w is None:
        return []
    n = Nar(seed=seed, style=rng.choice(list(SPLIT_BY_STYLE)), ds="iso")
    group = f"dg-{FAMILY}-{stable_id(GENERATOR, str(seed))}"
    base = {"family": FAMILY, "generator": GENERATOR, "use_case": "customer_support", "topic": "support_ops",
            "group": group, "style": n.style}
    out = items(rng, w, n, dict(base, rationale="[base:order]"))
    final, _ = replay(w)
    for _ in range(10):
        t = mutate(rng, w)
        if t is None:
            continue
        f2, trail2 = replay(t)
        if (f2.status, min(3, f2.units - 1)) != (final.status, min(3, final.units - 1)) and sum(not ok for *_, ok, _ in trail2) >= 1:
            out += items(rng, t, n, dict(base, rationale="[twin:order]"))
            break
    return out
