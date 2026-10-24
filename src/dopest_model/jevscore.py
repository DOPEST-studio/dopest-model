# Vendored unchanged from jgeuter/OpenDecisions score.py (MIT License).
"""Score a model's predictions on OpenDecisions: Intelligence, Calibration and Capability.

    python score.py predictions.jsonl                  # data file: opendecisions.jsonl next to this script
    python score.py predictions.jsonl --data path/to/opendecisions.jsonl --json

predictions.jsonl has one line per item: {"id": "<item id>", "probs": {"<label>": probability, ...}}
  choice  one probability per id in the item's "labels"
  yes/no  {"yes": p, "no": 1 - p}  (or only {"yes": p})
  score   one probability per level "0", "1", ... (the item's "labels")
Probabilities are used as given and must sum to 1 over the item's labels (within 0.01, so rounded outputs are fine).
An item without a prediction counts as wrong (this is how we count inputs that are too long for a model).

The metric is JevBench v1.6's Capability, with one change: a yes/no answer counts as "yes" when P(yes) >= 0.5 and
as "no" otherwise. JevBench v1.6 instead counts an answer with 0.2 < P(yes) < 0.8 as an abstention (wrong).

  Capability   = (Intelligence + Calibration) / 2
  Intelligence = mean over the three question types of a tier-weighted, chance-corrected competence
                 (tier weights easy 0.1, standard 0.2, judge 0.3, hard 0.4; each type's value clipped at 0),
                 clipped to [0, 100]
                 choice: 100 (acc - c) / (1 - c), c = mean of 1 / (number of options)
                 yes/no: 100 (acc - 0.5) / 0.5
                 score:  100 (1 - mean nMAE / mean nMAE of the mid-scale guess), nMAE = |E[level] - gold| / (K - 1)
  Calibration  = mean over the three types of
                 choice: 100 (1 - ECE / 0.5), top-label ECE on hard-tier items
                 yes/no: 100 (1 - ECE / 0.5), ECE of P(yes)
                 score:  mean of 100 (1 - top-level ECE / 0.5) and 100 (1 - mean nRPS)
                 (ECE: 10 equal-width bins; each ECE term clipped at 0; only items with a prediction)

Also reported, not part of Capability: on the 129 calibration items the state fixes the true probability of each
option (the item's "target_probs"). There the scorer reports the mean total variation distance between the predicted
and the true probabilities, TVD = 1/2 sum over labels |p - target| (0 = exact, 1 = disjoint; a missing prediction
counts as 1).

This file reimplements the JevBench v1.6 scorer from its published method and reproduces the published v1.6.1
results; JevBench's own v1.6 scorer is not public. MIT License.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

TIER_WEIGHTS = {"easy": 0.1, "standard": 0.2, "judge": 0.3, "hard": 0.4}
TYPES = ("choice", "noul", "score")


# ------------------------------------------------------------------ per item

def ece(pairs: list[tuple[float, float]], bins: int = 10) -> float | None:
    """Expected calibration error of (forecast, outcome) pairs; equal-width bins, empty bins absent."""
    if not pairs:
        return None
    sums = defaultdict(lambda: [0, 0.0, 0.0])
    for p, y in pairs:
        p = min(max(float(p), 0.0), 1.0)
        b = sums[min(int(p * bins), bins - 1)]
        b[0] += 1
        b[1] += p
        b[2] += float(y)
    return sum(n / len(pairs) * abs(o / n - f / n) for n, f, o in sums.values())


def nrps(probs: list[float], gold: int) -> float:
    """Normalized ranked probability score (0 is perfect)."""
    k, cum, total = len(probs), 0.0, 0.0
    for level in range(k - 1):
        cum += probs[level]
        total += (cum - (1.0 if gold <= level else 0.0)) ** 2
    return total / (k - 1)


def item_outcome(task: dict, probs: dict | None) -> dict:
    """Outcome of one item; probs is None for a missing prediction (wrong, no forecast)."""
    qtype, labels = task["question"]["type"], task["labels"]
    out = {"id": task["id"], "type": qtype, "tier": task["tier"], "k": len(labels), "valid": probs is not None}
    if qtype == "choice":
        out["chance"] = 1.0 / len(labels)
        if probs is None:
            out.update(correct=False, conf=None)
        else:
            best = max(sorted(labels), key=lambda lab: probs[lab])        # ties: the smallest label
            out.update(correct=best == task["expected"], conf=probs[best])
    elif qtype == "noul":
        y = 1.0 if task["expected"] == "yes" else 0.0
        if probs is None:
            out.update(correct=False, p_yes=None, y=y)
        else:
            p = probs["yes"]
            out.update(correct=(p >= 0.5) == (y == 1.0), p_yes=p, y=y)   # the more likely side; no abstention
    else:
        k, gold = len(labels), int(task["expected"])
        out["nmae_chance"] = sum(abs(level - gold) for level in range(k)) / k / (k - 1)   # random level
        out["nmae_chance_mid"] = abs((k - 1) / 2 - gold) / (k - 1)                         # mid-scale guess
        if probs is None:
            out.update(nmae=max(gold, k - 1 - gold) / (k - 1), conf=None, correct=False)   # the worst level
        else:
            vector = [probs[str(level)] for level in range(k)]
            ev = sum(level * p for level, p in enumerate(vector))
            best = max(range(k), key=lambda level: (vector[level], -level))
            out.update(nmae=abs(ev - gold) / (k - 1), conf=vector[best], correct=best == gold, nrps=nrps(vector, gold))
    target = task.get("target_probs")
    if target:                                            # calibration items: distance to the true probabilities
        out["tvd"] = 1.0 if probs is None else 0.5 * sum(abs(probs[lab] - float(target.get(lab, 0.0))) for lab in labels)
    return out


# ------------------------------------------------------------------ aggregation

def _weighted(values: dict, weights: dict) -> float | None:
    present = {k: v for k, v in values.items() if v is not None}
    total = sum(weights[k] for k in present)
    return sum(weights[k] * v for k, v in present.items()) / total if total else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def competence(outs: list[dict]) -> float | None:
    """Chance-corrected competence of items of one type and tier (0 = chance, 100 = perfect)."""
    if not outs:
        return None
    qtype = outs[0]["type"]
    if qtype == "choice":
        acc, chance = _mean([o["correct"] for o in outs]), _mean([o["chance"] for o in outs])
        return 100 * (acc - chance) / (1 - chance)
    if qtype == "noul":
        return 100 * (_mean([o["correct"] for o in outs]) - 0.5) / 0.5
    chance = _mean([o["nmae_chance_mid"] for o in outs])
    if chance <= 0:                                       # every gold at mid-scale: the random-level baseline
        chance = _mean([o["nmae_chance"] for o in outs])
    return 100 * (1 - _mean([o["nmae"] for o in outs]) / chance)


def intelligence(outs: list[dict]) -> tuple[float | None, dict]:
    per_type = {}
    for qtype in TYPES:
        tiers = {t: competence([o for o in outs if o["type"] == qtype and o["tier"] == t]) for t in TIER_WEIGHTS}
        cc = _weighted(tiers, TIER_WEIGHTS)
        per_type[qtype] = None if cc is None else max(0.0, cc)
    total = _weighted(per_type, {t: 1 / 3 for t in TYPES})
    return (None if total is None else min(100.0, max(0.0, total))), per_type


def _ece_score(e: float | None) -> float | None:
    return None if e is None else max(0.0, 100 * (1 - e / 0.5))


def calibration(outs: list[dict]) -> tuple[float | None, dict]:
    parts = {}
    choice = [o for o in outs if o["type"] == "choice" and o["valid"]]
    if choice:
        parts["choice"] = _ece_score(ece([(o["conf"], float(o["correct"])) for o in choice if o["tier"] == "hard"]))
    noul = [o for o in outs if o["type"] == "noul" and o["valid"]]
    if noul:
        parts["noul"] = _ece_score(ece([(o["p_yes"], o["y"]) for o in noul]))
    score = [o for o in outs if o["type"] == "score" and o["valid"]]
    if score:
        top = _ece_score(ece([(o["conf"], float(o["correct"])) for o in score]))
        parts["score"] = (top + 100 * (1 - _mean([o["nrps"] for o in score]))) / 2
    return _weighted(parts, {t: 1 / 3 for t in TYPES}), parts


def summary(outs: list[dict]) -> dict:
    i, cc = intelligence(outs)
    c, cal = calibration(outs)
    accuracy = {t: _mean([o["correct"] for o in outs if o["type"] == t]) * 100 for t in TYPES
                if any(o["type"] == t for o in outs)}
    tvd = [o["tvd"] for o in outs if "tvd" in o]
    return {"capability": None if i is None or c is None else (i + c) / 2, "intelligence": i, "calibration": c,
            "competence_per_type": cc, "calibration_per_type": cal, "accuracy_per_type": accuracy,
            "items": len(outs), "predicted": sum(o["valid"] for o in outs),
            "calibration_items_tvd": _mean(tvd) if tvd else None, "calibration_items": len(tvd)}


# ------------------------------------------------------------------ input

def normalized(task: dict, probs: dict) -> dict:
    labels = task["labels"]
    if task["question"]["type"] == "noul" and "no" not in probs:
        probs = {"yes": probs["yes"], "no": 1.0 - probs["yes"]}
    values = {lab: float(probs.get(lab, 0.0)) for lab in labels}
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError(f"{task['id']}: probabilities must be finite and non-negative")
    if abs(sum(values.values()) - 1.0) > 0.01:
        raise ValueError(f"{task['id']}: probabilities over the labels {labels} sum to {sum(values.values()):.4f}, not 1")
    return values


def score(tasks: list[dict], predictions: list[dict]) -> dict:
    by_id = {t["id"]: t for t in tasks}
    probs = {}
    for p in predictions:
        if p["id"] not in by_id:
            raise ValueError(f"unknown item id {p['id']!r}")
        if p["id"] in probs:
            raise ValueError(f"item {p['id']!r} predicted twice")
        probs[p["id"]] = normalized(by_id[p["id"]], p["probs"])
    return summary([item_outcome(t, probs.get(t["id"])) for t in tasks])


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions")
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "opendecisions.jsonl"))
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()
    result = score(read_jsonl(args.data), read_jsonl(args.predictions))
    if args.json:
        print(json.dumps(result, indent=1))
        return
    print(f"Capability {result['capability']:.1f}  (Intelligence {result['intelligence']:.1f}, "
          f"Calibration {result['calibration']:.1f});  {result['predicted']} of {result['items']} items predicted")
    for t in TYPES:
        if t in result["accuracy_per_type"]:
            print(f"  {t:7s} accuracy {result['accuracy_per_type'][t]:5.1f}%  competence "
                  f"{result['competence_per_type'][t]:5.1f}  calibration {result['calibration_per_type'][t]:5.1f}")
    if result["calibration_items_tvd"] is not None:
        print(f"Distance to the true probabilities on the {result['calibration_items']} calibration items "
              f"(mean TVD, 0 = exact): {result['calibration_items_tvd']:.3f}")


if __name__ == "__main__":
    main()
