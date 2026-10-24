"""Compare runs on one evaluation set under the board's rules, with paired bootstrap intervals.

    uv run python -m dopest_model.compare --data data/raw/opendecisions/opendecisions.jsonl \
        --calib data/dev/jevbench_public.jsonl runs/g4-12b/opendecisions.jsonl runs/pilot-12b-ab-v1/opendecisions.jsonl

--calib names the calibration DATA set; each run's logits on it are taken from the run's directory under the same
file name (runs/<run>/jevbench_public.jsonl), so temperatures never come from the scored set. Without --calib the
table shows the 2-fold cross-fitted estimate and says so. Groups: question type, tier, family, sub-generator.
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from . import jevscore
from .evaluate import (TYPES, crossfit_outcomes, fit_calibration, gold_index, outcomes, paired_bootstrap,
                       predictions)


def group_keys(task: dict) -> list[str]:
    keys = ["ALL", f"type:{task['question']['type']}", f"tier:{task['tier']}", f"family:{task['family']}"]
    tag = ((task.get("provenance") or {}).get("rationale") or "").split("]")[0]
    if tag.startswith("[") and ":" in tag:
        keys.append(f"sub:{tag.split(':')[1]}")
    return keys


def accuracy(rows: list[dict], tasks: dict) -> dict[str, list[int]]:
    """Argmax accuracy by group (temperature-free)."""
    acc = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        t = tasks[r["id"]]
        ok = max(range(len(r["ids"])), key=lambda i: r["logits"][i]) == gold_index(t, r["ids"])
        for k in group_keys(t):
            acc[k][0] += ok
            acc[k][1] += 1
    return acc


def rows_for(path: str, tasks: dict) -> list[dict]:
    """A run's rows restricted to the evaluation set (one per item), so a cleaned set can reuse old logits."""
    seen, out = set(), []
    for r in jevscore.read_jsonl(path):
        if r["id"] in tasks and r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--calib", help="calibration data set; per-run logits: <run dir>/<its file name>")
    parser.add_argument("--od-rules", action="store_true", help="OpenDecisions variant (no abstention band)")
    parser.add_argument("--bootstrap", type=int, default=400, help="paired bootstrap resamples (0 = off)")
    parser.add_argument("--min-n", type=int, default=10, help="hide groups smaller than this")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    official = not args.od_rules

    tasks = {}
    for t in jevscore.read_jsonl(args.data):
        tasks.setdefault(t["id"], t)
    tasks_list = list(tasks.values())
    runs = {Path(p).parent.name: rows_for(p, tasks) for p in args.runs}
    names = list(runs)
    width = 22

    calibs = {}
    if args.calib:
        calib_tasks = {t["id"]: t for t in jevscore.read_jsonl(args.calib)}
        for p in args.runs:
            logits = Path(p).parent / Path(args.calib).name
            if not logits.exists():
                raise SystemExit(f"{logits} missing: score this run on the calibration set first")
            calibs[Path(p).parent.name] = fit_calibration(rows_for(str(logits), calib_tasks), calib_tasks,
                                                          source=args.calib)
    print(f"rules: {'official (abstention band, uniform score chance)' if official else 'OpenDecisions'}; "
          f"calibration: {'from ' + args.calib if args.calib else '2-fold cross-fitted on the scored set (estimate)'}")
    print(f"{'':30s}" + "".join(f"{n[:width - 2]:>{width}s}" for n in names))
    one = {"temperatures": {t: 1.0 for t in TYPES}, "noul_floor": None}
    res1 = [jevscore.summary(outcomes(runs[n], tasks_list, one, official)) for n in names]
    print(f"{'Capability, T = 1, no floor':30s}" + "".join(f"{r['capability']:>{width}.1f}" for r in res1))
    outs = {n: outcomes(runs[n], tasks_list, calibs[n], official) if calibs else
            crossfit_outcomes(runs[n], tasks_list, official) for n in names}
    res = [jevscore.summary(outs[n]) for n in names]
    print(f"{'Capability, calibrated':30s}" + "".join(f"{r['capability']:>{width}.1f}" for r in res))
    print(f"{'  intelligence':30s}" + "".join(f"{r['intelligence']:>{width}.1f}" for r in res))
    print(f"{'  calibration':30s}" + "".join(f"{r['calibration']:>{width}.1f}" for r in res))
    for t in TYPES:
        print(f"{'  competence ' + t:30s}" + "".join(f"{r['competence_per_type'].get(t, float('nan')):>{width}.1f}"
                                                   for r in res))
    if calibs:
        print(f"{'  temperatures c/y/s':30s}" + "".join(
            f"{'/'.join(f'{calibs[n]['temperatures'][t]:.2f}' for t in TYPES):>{width}s}" for n in names))
    if args.bootstrap and len(names) > 1:
        bs = paired_bootstrap(outs, args.bootstrap)
        print(f"{'  vs first, 95% CI':30s}" + f"{'':>{width}s}" + "".join(
            f"{f'{bs[n][0]:+.1f} [{bs[n][1]:+.1f}, {bs[n][2]:+.1f}]':>{width}s}" for n in names[1:]))

    accs = [accuracy(runs[n], tasks) for n in names]
    keys = sorted({k for a in accs for k in a}, key=lambda k: (k != "ALL", k.split(":")[0], k))
    print(f"\n{'argmax accuracy':30s}" + "".join(f"{n[:width - 2]:>{width}s}" for n in names) + f"{'change':>10s}")
    for k in keys:
        n_items = accs[0].get(k, [0, 0])[1]
        if n_items < args.min_n:
            continue
        vals = [a[k][0] / a[k][1] if k in a and a[k][1] else float("nan") for a in accs]
        print(f"{k[:29]:30s}" + "".join(f"{v:>{width - 8}.1%} n={n_items:<5d}" for v in vals)
              + f"{(vals[-1] - vals[0]) * 100:>+9.1f}")


if __name__ == "__main__":
    main()
