"""Score letter logits with JevBench's Capability, by default under the board's rules (METHOD v1.5 §3.1).

    uv run python -m dopest_model.evaluate runs/g4-12b/opendecisions.jsonl \
        --calib-logits runs/g4-12b/jevbench_public.jsonl --calib-data data/dev/jevbench_public.jsonl
    uv run python -m dopest_model.evaluate runs/g4-12b/opendecisions.jsonl --calib runs/g4-12b/calibration.json
    uv run python -m dopest_model.evaluate runs/g4-12b/opendecisions.jsonl --fit-cv      # estimate only

Official rules on top of the vendored OpenDecisions scorer (jevscore.py): a yes/no answer with 0.2 < P(yes) < 0.8
is an abstention and counts as wrong; the chance baseline of a score item is the uniform random level, not the
mid-scale guess. --od-rules restores the OpenDecisions variant.

Calibration is an artifact, never fitted on the set being scored: per-type temperatures (NLL fit) plus the yes/no
commit floor, saved as JSON and applied identically here and when serving. --fit-cv (2-fold on the scored set) is
an optimistic estimate kept for quick looks.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np

from . import jevscore
from .calibrate import NOUL_FLOOR, commit_floor, fit_temperature, softmax
from .prompt import probs_by_label

TYPES = ("choice", "noul", "score")
DEFAULT_DATA = "data/raw/opendecisions/opendecisions.jsonl"   # no import of predict.py: it needs MLX


def gold_index(task: dict, ids: list[str]) -> int:
    expected = task["expected"]
    if task["question"]["type"] == "noul":
        expected = "true" if expected == "yes" else "false"
    return ids.index(str(expected))


def fit_per_type(rows: list[dict], tasks: dict) -> dict[str, float]:
    temps = {}
    for qtype in TYPES:
        sel = [r for r in rows if r["type"] == qtype]
        temps[qtype] = fit_temperature([np.array(r["logits"]) for r in sel],
                                       [gold_index(tasks[r["id"]], r["ids"]) for r in sel]) if sel else 1.0
    return temps


def fit_calibration(rows: list[dict], tasks: dict, floor: float = NOUL_FLOOR, source: str = "") -> dict:
    """Calibration artifact: temperatures per type and the yes/no commit floor, with its provenance."""
    counts = collections.Counter(r["type"] for r in rows)
    return {"temperatures": fit_per_type(rows, tasks), "noul_floor": floor,
            "fitted_on": {"source": source, "items": dict(counts)}}


def load_calibration(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_calibration(calib: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(calib, indent=1), encoding="utf-8")


def as_calibration(calib_or_temps: dict) -> dict:
    """Accept a bare temperature dict (legacy) or a calibration artifact."""
    if "temperatures" in calib_or_temps:
        return calib_or_temps
    return {"temperatures": dict(calib_or_temps), "noul_floor": None}


def predictions(rows: list[dict], calib: dict) -> list[dict]:
    calib = as_calibration(calib)
    temps, floor = calib["temperatures"], calib.get("noul_floor")
    out = []
    for r in rows:
        probs = probs_by_label(r["type"], r["ids"], softmax(np.array(r["logits"]), temps[r["type"]]).tolist())
        if r["type"] == "noul" and floor:
            probs = {"yes": commit_floor(probs["yes"], floor)}
            probs["no"] = 1.0 - probs["yes"]
        out.append({"id": r["id"], "probs": probs})
    return out


def official_outcome(task: dict, probs: dict | None) -> dict:
    """jevscore's item outcome under the board's rules (METHOD v1.5 §3.1)."""
    o = jevscore.item_outcome(task, probs)
    if task["question"]["type"] == "noul" and probs is not None and 0.2 < probs["yes"] < 0.8:
        o["correct"] = False                                   # abstention
    if task["question"]["type"] == "score":
        o["nmae_chance_mid"] = o["nmae_chance"]                # chance = uniform random level
    return o


def outcomes(rows: list[dict], tasks_list: list[dict], calib: dict, official: bool = True) -> list[dict]:
    """One outcome per task, in task order; a task without a row counts as unanswered (wrong)."""
    by_id = {t["id"]: t for t in tasks_list}
    probs = {}
    for p in predictions([r for r in rows if r["id"] in by_id], calib):
        probs[p["id"]] = jevscore.normalized(by_id[p["id"]], p["probs"])
    make = official_outcome if official else jevscore.item_outcome
    return [make(t, probs.get(t["id"])) for t in tasks_list]


def crossfit_outcomes(rows: list[dict], tasks_list: list[dict], official: bool = True,
                      floor: float | None = NOUL_FLOOR) -> list[dict]:
    """Estimate: temperatures fitted on one half of the scored set, applied to the other half, both ways."""
    tasks = {t["id"]: t for t in tasks_list}
    rows = sorted((r for r in rows if r["id"] in tasks), key=lambda r: r["id"])
    random.Random(0).shuffle(rows)
    halves = [rows[::2], rows[1::2]]
    outs = {}
    for fit_rows, eval_rows in (halves, halves[::-1]):
        calib = {"temperatures": fit_per_type(fit_rows, tasks), "noul_floor": floor}
        eval_tasks = [tasks[r["id"]] for r in eval_rows]
        for t, o in zip(eval_tasks, outcomes(eval_rows, eval_tasks, calib, official)):
            outs[t["id"]] = o
    make = official_outcome if official else jevscore.item_outcome
    return [outs.get(t["id"]) or make(t, None) for t in tasks_list]


def score_rows(rows: list[dict], tasks_list: list[dict], calib: dict | None = None, official: bool = True) -> dict:
    """Capability summary; calib=None gives the 2-fold cross-fitted estimate."""
    outs = outcomes(rows, tasks_list, calib, official) if calib is not None else \
        crossfit_outcomes(rows, tasks_list, official)
    return jevscore.summary(outs)


def paired_bootstrap(outs_by_run: dict[str, list[dict]], n_boot: int = 400, seed: int = 0) -> dict:
    """Paired, stratified (type x tier) bootstrap of Capability differences against the first run.

    Returns {run: (difference, ci_low, ci_high)} for every run but the first, plus "_point": full-set values.
    The same resample is used for all runs, as on the leaderboard."""
    names = list(outs_by_run)
    ref = outs_by_run[names[0]]
    strata = collections.defaultdict(list)
    for i, o in enumerate(ref):
        strata[(o["type"], o["tier"])].append(i)
    rng = random.Random(seed)
    point = {n: jevscore.summary(outs_by_run[n])["capability"] for n in names}
    diffs = {n: [] for n in names[1:]}
    for _ in range(n_boot):
        sample = [rng.choice(idx) for idx in strata.values() for _ in idx]
        base = jevscore.summary([ref[i] for i in sample])["capability"]
        for n in names[1:]:
            diffs[n].append(jevscore.summary([outs_by_run[n][i] for i in sample])["capability"] - base)
    result = {"_point": point}
    for n, d in diffs.items():
        d.sort()
        result[n] = (point[n] - point[names[0]], d[int(0.025 * n_boot)], d[max(0, int(0.975 * n_boot) - 1)])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logits")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--calib", help="calibration JSON (temperatures per type, yes/no floor)")
    parser.add_argument("--calib-logits", help="fit the calibration on these logits ...")
    parser.add_argument("--calib-data", help="... of this data set, then save it next to the scored logits")
    parser.add_argument("--temperatures", help="choice,yesno,score: fixed temperatures with the default floor")
    parser.add_argument("--fit-cv", action="store_true", help="2-fold cross-fitted estimate (optimistic)")
    parser.add_argument("--od-rules", action="store_true", help="OpenDecisions variant: no abstention band")
    args = parser.parse_args()

    tasks_list = jevscore.read_jsonl(args.data)
    rows = jevscore.read_jsonl(args.logits)
    official = not args.od_rules
    if args.calib:
        calib = load_calibration(args.calib)
    elif args.calib_logits:
        calib_tasks = {t["id"]: t for t in jevscore.read_jsonl(args.calib_data)}
        calib = fit_calibration(jevscore.read_jsonl(args.calib_logits), calib_tasks, source=args.calib_data)
        out = Path(args.logits).with_name(f"calibration_from_{Path(args.calib_data).stem}.json")
        save_calibration(calib, out)
        print(f"calibration {calib['temperatures']} floor {calib['noul_floor']} -> {out}")
    elif args.temperatures:
        calib = {"temperatures": dict(zip(TYPES, (float(x) for x in args.temperatures.split(",")))),
                 "noul_floor": NOUL_FLOOR}
    elif args.fit_cv:
        calib = None
    else:
        raise SystemExit("give --calib, --calib-logits with --calib-data, --temperatures or --fit-cv")

    result = score_rows(rows, tasks_list, calib, official)
    label = "official rules" if official else "OpenDecisions rules"
    print(f"Capability {result['capability']:.1f}  (Intelligence {result['intelligence']:.1f}, "
          f"Calibration {result['calibration']:.1f}; {label}{', cross-fitted estimate' if calib is None else ''});  "
          f"{result['predicted']} of {result['items']} items predicted")
    for t in TYPES:
        if t in result["accuracy_per_type"]:
            print(f"  {t:7s} accuracy {result['accuracy_per_type'][t]:5.1f}%  competence "
                  f"{result['competence_per_type'][t]:5.1f}  calibration {result['calibration_per_type'][t]:5.1f}")
    Path(args.logits.replace(".jsonl", ".score.json")).write_text(
        json.dumps({"calibration": calib, "official": official, **result}), encoding="utf-8")


if __name__ == "__main__":
    main()
