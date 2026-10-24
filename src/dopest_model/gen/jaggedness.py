"""Tasks that are trivial for code and treacherous for one forward pass (family jaggedness, the 'jagged frontier').

Nine small worlds, each with an exact engine and a one-fact twin:
  invoice_sum     exact sum of quantity x unit price lines; a stated total to check; size of the discrepancy
  weekday         weekday of a date from a stated reference weekday (month ends, leap days, year ends)
  letter_count    occurrences of a letter in a name (case-insensitive)
  palindrome      does a token read the same backwards
  digit_present   digit characters versus numbers written as words
  list_category   how many entries of a list belong to a category (closed word lists in this module)
  decimal_extreme largest or smallest of decimals whose digit counts mislead (3.9 > 3.11)
  unit_compare    which weight is heavier across pounds and kilograms with a stated factor
  return_window   how late a request is against a window of N days
Surface forms are this project's own; they do not copy the OpenDecisions templates, so the OpenDecisions
jaggedness score stays an out-of-template check rather than a memorised one.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal

from .common import DATE_STYLES, SPLIT_BY_STYLE, Nar, choice_item, fmt_date, noul_item, score_item, stable_id

FAMILY = "jaggedness"
GENERATOR = "jaggedness_v1"
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

PRODUCTS = ["safety gloves", "desk lamps", "label rolls", "toner cartridges", "cable ties", "HDMI adapters",
            "LED panels", "extension cords", "printer paper", "whiteboard markers", "USB hubs", "packing tape"]
NAMES = ["Hollingsworth", "Bannerman", "Abernathy", "Mississauga", "Pennington", "Kowalczyk", "Lindqvist",
         "Strathmore", "Annapolis", "Barrington", "Cassiopeia", "Delacroix", "Montgomery", "Rappaport", "Tennessee",
         "Ellingham", "Garraway", "Vanterpool", "Quarrington", "Sorrento", "Hallingdal", "Brennerhof"]
PALINDROMES = ["sagas", "level", "rotor", "civic", "refer", "kayak", "stats", "tenet", "radar", "madam", "solos",
               "redder", "repaper", "deified", "racecar", "minim", "noon", "peep"]
CATEGORIES = {
    "birds": ["kestrel", "puffin", "falcon", "osprey", "pelican", "finch", "heron", "wren", "plover", "curlew"],
    "metals": ["cobalt", "titanium", "nickel", "platinum", "zinc", "tungsten", "copper", "lead", "tin", "iron"],
    "fruits": ["apricot", "guava", "pear", "lime", "fig", "quince", "mango", "plum", "lychee", "papaya"],
    "musical instruments": ["banjo", "clarinet", "violin", "harp", "cello", "oboe", "trumpet", "sitar", "lute",
                            "bassoon"],
    "trees": ["alder", "rowan", "hazel", "larch", "beech", "aspen", "cedar", "linden", "hornbeam", "yew"],
}
NUMBER_WORDS = ["eighteen", "four", "twelve", "seven", "forty", "nine", "three", "sixty", "two", "eleven"]


def money(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"


def money_label(cents: int) -> str:
    return f"usd_{cents // 100}_{cents % 100:02d}"


# ------------------------------------------------------------------ the nine worlds

@dataclass(frozen=True)
class Invoice:
    lines: tuple              # (product, qty, unit cents)
    stated: int               # stated total in cents
    ask: str                  # "choice" | "noul" | "score"

    def total(self) -> int:
        return sum(q * u for _, q, u in self.lines)


class InvoiceSum:
    name, tier, topic = "invoice_sum", "easy", "math"
    BANDS = ["No discrepancy (equal to the cent)", "Discrepancy under $1.00", "Discrepancy from $1.00 to $50.00",
             "Discrepancy above $50.00"]

    @staticmethod
    def band(w: Invoice) -> int:
        gap = abs(w.stated - w.total())
        return 0 if gap == 0 else 1 if gap < 100 else 2 if gap <= 5000 else 3

    def solve(self, w: Invoice):
        if w.ask == "choice":
            return money_label(w.total())
        if w.ask == "noul":
            return w.stated == w.total()
        return self.band(w)

    def sample(self, rng: random.Random) -> Invoice:
        lines = tuple((p, rng.randint(2, 40), rng.randint(150, 9500))
                      for p in rng.sample(PRODUCTS, rng.randint(3, 5)))
        w = Invoice(lines, 0, rng.choice(["choice", "noul", "noul", "score"]))
        gap = rng.choice([0, 0, rng.randint(1, 99), rng.randint(100, 5000), rng.randint(5001, 20000)])
        return replace(w, stated=w.total() + rng.choice([-1, 1]) * gap)

    def mutate(self, rng: random.Random, w: Invoice) -> Invoice:
        if w.ask == "choice":
            p, q, u = w.lines[0]
            return replace(w, lines=((p, q + 1, u),) + w.lines[1:])
        if w.ask == "noul":
            return replace(w, stated=w.total() if w.stated != w.total() else w.total() + rng.choice([1, 9, 10, 90]))
        target = rng.choice([b for b in range(4) if b != self.band(w)])
        gap = [0, 37, 1250, 9900][target]
        return replace(w, stated=w.total() + gap)

    def items(self, rng, w: Invoice, n: Nar, base: dict) -> list[dict]:
        r = n.rng(21)
        vendor = r.choice(["Juniper Logistics", "Calloway Mobility", "Ambergate Supply", "Tessaro Health"])
        inv = f"INV-{r.randint(10000, 99999)}"
        body = "\n".join(f"{p}: {q} x {money(u)}" for p, q, u in w.lines)
        state = f"Invoice {inv} from {vendor}\n{body}"
        if w.ask != "choice":
            state += f"\nStated total: {money(w.stated)}"
        why = f"Sum of lines = {money(w.total())}" + (f"; stated {money(w.stated)}." if w.ask != "choice" else ".")
        common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
        if w.ask == "choice":
            t = w.total()
            p, q, u = w.lines[0]
            wrong = [t - u, t + u, t + 100, t - 10, t + 9]          # one unit too many or few, a dollar, a dime
            picked = [t] + [x for x in dict.fromkeys(wrong) if x != t and x > 0][:3]
            return [choice_item(rng=rng, options={money_label(x): f"The lines sum to {money(x)}." for x in picked},
                                expected=money_label(t), trap=money_label(picked[1]),
                                instructions="What is the exact sum of the lines (quantity times unit price)?", **common)]
        if w.ask == "noul":
            return [noul_item(instructions="Is the stated total exactly equal to the sum of the lines "
                                           "(quantity times unit price)?",
                              true_text="The stated total equals the sum to the cent.",
                              false_text="The stated total differs from the sum.", answer=self.solve(w), **common)]
        return [score_item(instructions="How large is the gap between the stated total and the exact sum of the "
                                        "lines? Output the level index.", levels=self.BANDS, expected=self.band(w),
                           **common)]


@dataclass(frozen=True)
class Weekday:
    ref: date
    target: date


class WeekdayOf:
    name, tier, topic = "weekday", "standard", "everyday_language"

    def solve(self, w: Weekday) -> str:
        return WEEKDAYS[w.target.weekday()]

    def sample(self, rng: random.Random) -> Weekday:
        ref = date(rng.choice([2026, 2027, 2028]), rng.randint(1, 12), rng.randint(1, 28))
        return Weekday(ref, ref + timedelta(days=rng.choice([-1, 1]) * rng.randint(9, 75)))

    def mutate(self, rng: random.Random, w: Weekday) -> Weekday:
        return replace(w, target=w.target + timedelta(days=rng.choice([1, 2, 3, 4, 5, 6])))

    def items(self, rng, w: Weekday, n: Nar, base: dict) -> list[dict]:
        r = n.rng(22)
        event = r.choice(["The delivery", "The site visit", "The board meeting", "The audit", "The handover"])
        state = (f"{fmt_date(w.ref, n.ds)} is a {WEEKDAYS[w.ref.weekday()].capitalize()}. "
                 f"{event} is booked for {fmt_date(w.target, n.ds)}.")
        gold = self.solve(w)
        opts = [gold] + [d for d in rng.sample(WEEKDAYS, 7) if d != gold][:rng.randint(3, 5)]
        common = dict(base, state=state, rationale=f"{base['rationale']} {w.target} is a {gold}.")
        return [choice_item(rng=rng, options={d: d.capitalize() for d in opts}, expected=gold, trap=None,
                            instructions=f"On which day of the week does {event[0].lower() + event[1:]} fall?",
                            **common)]


@dataclass(frozen=True)
class Letters:
    word: str
    letter: str
    k: int                    # the count the yes/no question asks about
    ask: str                  # "noul" | "score"


class LetterCount:
    name, tier, topic = "letter_count", "easy", "everyday_language"

    @staticmethod
    def count(w: Letters) -> int:
        return w.word.lower().count(w.letter)

    def solve(self, w: Letters):
        return self.count(w) == w.k if w.ask == "noul" else min(4, self.count(w))

    def sample(self, rng: random.Random) -> Letters:
        word = rng.choice(NAMES)
        letter = rng.choice(sorted(set(word.lower())) + ["z", "q", "x"])
        true = word.lower().count(letter)
        return Letters(word, letter, max(1, rng.choice([true, true, true - 1, true + 1])),
                       rng.choice(["noul", "score"]))

    def mutate(self, rng: random.Random, w: Letters) -> Letters | None:
        """Ask about another letter of the same name, one whose answer differs."""
        flipped = [c for c in "abcdefghijklmnopqrstuvwxyz" if self.solve(replace(w, letter=c)) != self.solve(w)]
        return replace(w, letter=rng.choice(flipped)) if flipped else None

    def items(self, rng, w: Letters, n: Nar, base: dict) -> list[dict]:
        r = n.rng(23)
        state = f"{r.choice(['Supplier name', 'Street name', 'Project code name', 'Customer surname'])}: {w.word}"
        why = f"'{w.letter}' occurs {self.count(w)} times in {w.word}."
        common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
        if w.ask == "noul":
            times = "exactly once" if w.k == 1 else f"exactly {w.k} times"
            return [noul_item(instructions=f"Does the letter '{w.letter}' occur {times} in the name after the colon "
                                           f"(upper and lower case counted together)?",
                              true_text=f"It occurs {times}.", false_text=f"It does not occur {times}.",
                              answer=self.solve(w), **common)]
        return [score_item(instructions=f"How many times does the letter '{w.letter}' occur in the name after the "
                                        f"colon? Upper and lower case count together. Output the level index.",
                           levels=["It does not occur", "Once", "Twice", "Three times", "Four or more times"],
                           expected=self.solve(w), **common)]


@dataclass(frozen=True)
class Token:
    token: str


class Palindrome:
    name, tier, topic = "palindrome", "easy", "everyday_language"

    def solve(self, w: Token) -> bool:
        return w.token == w.token[::-1]

    def sample(self, rng: random.Random) -> Token:
        p = rng.choice(PALINDROMES)
        if rng.random() < 0.5:
            return Token(p)
        i = rng.randrange(len(p))
        return Token(p[:i] + rng.choice([c for c in "aeioulnrst" if c != p[i]]) + p[i + 1:])

    def mutate(self, rng: random.Random, w: Token) -> Token:
        t = w.token
        if self.solve(w):
            i = rng.randrange(len(t) // 2)
            return Token(t[:i] + rng.choice([c for c in "aeioulnrst" if c != t[i]]) + t[i + 1:])
        half = t[: (len(t) + 1) // 2]
        return Token(half + half[: len(t) // 2][::-1])

    def items(self, rng, w: Token, n: Nar, base: dict) -> list[dict]:
        common = dict(base, state=f"Token: {w.token}", rationale=f"{base['rationale']} Reversed: {w.token[::-1]}.")
        return [noul_item(instructions="Does the token read the same forwards and backwards, character by character?",
                          true_text="It reads the same backwards.", false_text="It does not read the same backwards.",
                          answer=self.solve(w), **common)]


@dataclass(frozen=True)
class Message:
    number: str               # always written as a word
    slot: str
    code: str                 # the only place a digit can appear


class DigitPresent:
    name, tier, topic = "digit_present", "easy", "everyday_language"
    SLOT = ["locker", "room", "gate", "bay", "desk", "row"]
    WORD_CODES = ["B-twelve", "east wing", "north annex", "level four", "the mezzanine"]

    def solve(self, w: Message) -> bool:
        return any(c.isdigit() for c in w.code)

    def digit_code(self, rng: random.Random) -> str:
        return rng.choice([f"{rng.randint(1, 9)}{rng.choice('ABCD')}", f"{rng.choice('ABCD')}{rng.randint(2, 99)}"])

    def sample(self, rng: random.Random) -> Message:
        code = self.digit_code(rng) if rng.random() < 0.5 else rng.choice(self.WORD_CODES)
        return Message(rng.choice(NUMBER_WORDS), rng.choice(self.SLOT), code)

    def mutate(self, rng: random.Random, w: Message) -> Message:
        return replace(w, code=rng.choice(self.WORD_CODES) if self.solve(w) else self.digit_code(rng))

    def items(self, rng, w: Message, n: Nar, base: dict) -> list[dict]:
        text = f"Please confirm the {w.number} seats for the offsite in {w.slot} {w.code}."
        common = dict(base, state=f"Message: {text}",
                      rationale=f"{base['rationale']} Digits present: {self.solve(w)}.")
        return [noul_item(instructions="Does the message contain at least one digit character (0-9)?",
                          true_text="It contains a digit character.", false_text="It contains no digit character.",
                          answer=self.solve(w), **common)]


@dataclass(frozen=True)
class Listing:
    items: tuple
    category: str
    k: int
    ask: str


class ListCategory:
    name, tier, topic = "list_category", "standard", "everyday_language"

    @staticmethod
    def count(w: Listing) -> int:
        return sum(x in CATEGORIES[w.category] for x in w.items)

    def solve(self, w: Listing):
        c = self.count(w)
        return c >= w.k if w.ask == "noul" else 0 if c == 0 else 1 if c <= 2 else 2 if c <= 4 else 3

    def sample(self, rng: random.Random) -> Listing:
        cat = rng.choice(list(CATEGORIES))
        inside = rng.sample(CATEGORIES[cat], rng.randint(0, 6))
        pool = [x for c, xs in CATEGORIES.items() if c != cat for x in xs]
        items = inside + rng.sample(pool, rng.randint(8, 11) - len(inside))
        rng.shuffle(items)
        c = len(inside)
        return Listing(tuple(items), cat, max(1, c + rng.choice([0, 0, 1, -1])), rng.choice(["noul", "score"]))

    def mutate(self, rng: random.Random, w: Listing) -> Listing | None:
        items = list(w.items)
        ins = [i for i, x in enumerate(items) if x in CATEGORIES[w.category]]
        outs = [i for i, x in enumerate(items) if x not in CATEGORIES[w.category]]
        for _ in range(3):
            if (rng.random() < 0.5 and ins) or not outs:
                i = rng.choice(ins)
                pool = [x for c, xs in CATEGORIES.items() if c != w.category for x in xs if x not in items]
            else:
                i = rng.choice(outs)
                pool = [x for x in CATEGORIES[w.category] if x not in items]
            items[i] = rng.choice(pool)
            t = replace(w, items=tuple(items))
            if self.solve(t) != self.solve(w):
                return t
        return None

    def items(self, rng, w: Listing, n: Nar, base: dict) -> list[dict]:
        state = "List: " + ", ".join(w.items)
        common = dict(base, state=state, rationale=f"{base['rationale']} There are {self.count(w)} {w.category}.")
        if w.ask == "noul":
            return [noul_item(instructions=f"Does the list contain at least {w.k} {w.category}?",
                              true_text=f"It contains at least {w.k} {w.category}.",
                              false_text=f"It contains fewer than {w.k} {w.category}.", answer=self.solve(w), **common)]
        return [score_item(instructions=f"How many entries in the list are {w.category}? Output the level index.",
                           levels=[f"No {w.category}", f"1-2 {w.category}", f"3-4 {w.category}",
                                   f"5 or more {w.category}"], expected=self.solve(w), **common)]


@dataclass(frozen=True)
class Decimals:
    values: tuple             # (name, Decimal)
    largest: bool


class DecimalExtreme:
    name, tier, topic = "decimal_extreme", "easy", "math"
    NAMES = ["glade", "atlas", "beacon", "fjord", "summit", "harbour", "meadow", "quarry"]

    def solve(self, w: Decimals) -> str:
        pick = max if w.largest else min
        return pick(w.values, key=lambda v: v[1])[0]

    def sample(self, rng: random.Random) -> Decimals | None:
        whole = rng.randint(1, 9)
        vals = {Decimal(f"{whole}.{rng.choice([rng.randint(1, 9), rng.randint(10, 99)])}") for _ in range(6)}
        vals = sorted(vals)[:rng.randint(4, 5)]
        if len(vals) < 4:
            return None
        names = rng.sample(self.NAMES, len(vals))
        return Decimals(tuple(zip(names, vals)), rng.random() < 0.5)

    def mutate(self, rng: random.Random, w: Decimals) -> Decimals | None:
        gold = self.solve(w)
        vals = [v for _, v in w.values]
        others = [(nm, v) for nm, v in w.values if nm != gold]
        nm, v = rng.choice(others)
        step = Decimal("0.01")
        new = (max(vals) + step) if w.largest else (min(vals) - step)
        if new <= 0:
            return None
        return replace(w, values=tuple((a, new if a == nm else b) for a, b in w.values))

    def items(self, rng, w: Decimals, n: Nar, base: dict) -> list[dict]:
        r = n.rng(24)
        what = r.choice(["Average delivery rating per courier", "Mean response score per team",
                         "Average review score per supplier"])
        state = f"{what}:\n" + "\n".join(f"{nm}: {v}" for nm, v in w.values)
        word = "highest" if w.largest else "lowest"
        common = dict(base, state=state, rationale=f"{base['rationale']} {self.solve(w)} is the {word} as a decimal.")
        return [choice_item(rng=rng, options={nm: f"{nm} has the {word} value." for nm, _ in w.values},
                            expected=self.solve(w), trap=None,
                            instructions=f"Which entry has the {word} value, read as decimal numbers?", **common)]


@dataclass(frozen=True)
class Weights:
    a_lb: Decimal
    b_kg: Decimal


class UnitCompare:
    name, tier, topic = "unit_compare", "standard", "math"
    LB = Decimal("0.4536")

    def solve(self, w: Weights) -> bool:
        return w.a_lb * self.LB > w.b_kg

    def sample(self, rng: random.Random) -> Weights | None:
        a = Decimal(rng.randint(150, 2500)) / 100
        b = (a * self.LB * (1 + Decimal(rng.choice([-1, 1]) * rng.randint(2, 15)) / 100)).quantize(Decimal("0.01"))
        return Weights(a, b) if b > 0 and a * self.LB != b else None

    def mutate(self, rng: random.Random, w: Weights) -> Weights:
        kg = w.a_lb * self.LB
        delta = kg * Decimal(rng.randint(2, 15)) / 100
        return replace(w, b_kg=(kg + delta if self.solve(w) else kg - delta).quantize(Decimal("0.01")))

    def items(self, rng, w: Weights, n: Nar, base: dict) -> list[dict]:
        state = f"Parcel weights: crate_c: {w.a_lb} lb; crate_b: {w.b_kg} kg (1 lb = 0.4536 kg)"
        why = f"crate_c {w.a_lb * self.LB:.3f} kg vs crate_b {w.b_kg} kg."
        common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
        return [noul_item(instructions="Is crate_c heavier than crate_b?", true_text="crate_c is heavier.",
                          false_text="crate_c is not heavier.", answer=self.solve(w), **common)]


@dataclass(frozen=True)
class Window:
    delivered: date
    requested: date
    days: int


class ReturnWindow:
    name, tier, topic = "return_window", "standard", "support_ops"
    LEVELS = ["Within the window", "1-2 days late", "3-7 days late", "More than 7 days late"]

    def solve(self, w: Window) -> int:
        late = (w.requested - w.delivered).days - w.days
        return 0 if late <= 0 else 1 if late <= 2 else 2 if late <= 7 else 3

    def sample(self, rng: random.Random) -> Window:
        delivered = date(rng.choice([2027, 2028]), rng.randint(1, 12), rng.randint(1, 28))
        days = rng.choice([14, 30, 45, 60])
        return Window(delivered, delivered + timedelta(days=days + rng.choice([-5, -1, 0, 1, 2, 3, 6, 9, 15])), days)

    def mutate(self, rng: random.Random, w: Window) -> Window:
        target = rng.choice([lv for lv in range(4) if lv != self.solve(w)])
        late = [0, 1, 5, 12][target]
        return replace(w, requested=w.delivered + timedelta(days=w.days + late))

    def items(self, rng, w: Window, n: Nar, base: dict) -> list[dict]:
        r = n.rng(25)
        order = f"ORD-{r.randint(10000, 99999)}"
        state = (f"Order {order} was delivered on {fmt_date(w.delivered, n.ds)}. The customer asked to return it on "
                 f"{fmt_date(w.requested, n.ds)}. Returns are accepted if requested no more than {w.days} days after "
                 f"delivery.")
        why = f"{(w.requested - w.delivered).days} days after delivery; window {w.days}."
        common = dict(base, state=state, rationale=f"{base['rationale']} {why}")
        return [score_item(instructions="How late was the return request relative to the return window? Output the "
                                        "level index.", levels=self.LEVELS, expected=self.solve(w), **common)]


# ------------------------------------------------------------------ driver

TASKS = [InvoiceSum(), WeekdayOf(), LetterCount(), Palindrome(), DigitPresent(), ListCategory(), DecimalExtreme(),
         UnitCompare(), ReturnWindow()]


def generate_case(seed: int) -> list[dict]:
    rng = random.Random(seed)
    task = TASKS[seed % len(TASKS)]
    w = None
    for _ in range(50):
        w = task.sample(rng)
        if w is not None:
            break
    if w is None:
        return []
    n = Nar(seed=seed, style=rng.choice(list(SPLIT_BY_STYLE)), ds=rng.choice(DATE_STYLES))
    group = f"dg-{FAMILY}-{stable_id(GENERATOR, str(seed))}"
    base = {"family": FAMILY, "generator": GENERATOR, "use_case": "other", "topic": task.topic, "group": group,
            "style": n.style, "tier": task.tier}
    items = task.items(rng, w, n, dict(base, rationale=f"[base:{task.name}]"))
    for _ in range(5):
        t = task.mutate(rng, w)
        if t is not None and task.solve(t) != task.solve(w):
            items += task.items(rng, t, n, dict(base, rationale=f"[twin:{task.name}]"))
            break
    return items
