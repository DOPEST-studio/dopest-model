"""Backend-agnostic training data helpers, shared by the MLX trainer (train.py) and the PyTorch one
(torch_backend.py), so both see identical examples and targets."""
from __future__ import annotations

import json
import random


def shuffled(item: dict, rng: random.Random) -> dict:
    """The item with its choice options in a new random order."""
    q = item["question"]
    if q["type"] != "choice" or not isinstance(q["criteria"], dict):
        return item
    pairs = list(q["criteria"].items())
    rng.shuffle(pairs)
    return dict(item, question=dict(q, criteria=dict(pairs)))


def target(item: dict, ids: list[str], soft: bool) -> list[float]:
    """Target distribution over the item's options in prompt order."""
    def key(i: str) -> str:
        return {"true": "yes", "false": "no"}.get(i, i) if item["question"]["type"] == "noul" else i
    tp = item.get("target_probs")
    if soft and tp:
        return [float(tp.get(key(i), 0.0)) for i in ids]
    gold = str(item["expected"])
    return [1.0 if key(i) == gold else 0.0 for i in ids]


def read_items(paths: list[str], split: str | None) -> list[dict]:
    items = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            items += [it for it in map(json.loads, f) if split is None or it.get("split") == split]
    return items


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warm-up from 1% to 100%, then cosine decay to 0 (the MLX trainer's schedule)."""
    if step < warmup:
        return 0.01 + 0.99 * step / warmup
    import math
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
