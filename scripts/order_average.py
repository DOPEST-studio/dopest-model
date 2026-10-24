"""Does averaging two option orders help? Official-rules Capability of single-order vs averaged probabilities.

    uv run python scripts/order_average.py --data data/raw/opendecisions/opendecisions.jsonl \
        --calib runs/pilot-31b-ab-bidir-full/calibration.json \
        --forward runs/pilot-31b-ab-bidir-full/opendecisions.jsonl --reverse runs/pilot-31b-ab-bidir-full-rev/opendecisions.jsonl

Choice questions only (training shuffled choice options; yes/no and score keep their fixed order). The two rows of
an item carry the same option ids in different letter positions; probabilities are averaged by id after the
temperature, then scored exactly as the server would answer.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dopest_model import jevscore                                   # noqa: E402
from dopest_model.evaluate import official_outcome, paired_bootstrap, predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--forward", required=True)
    parser.add_argument("--reverse", required=True)
    parser.add_argument("--bootstrap", type=int, default=400)
    args = parser.parse_args()
    tasks = list({t["id"]: t for t in map(json.loads, open(args.data, encoding="utf-8"))}.values())
    by_id = {t["id"]: t for t in tasks}
    calib = json.load(open(args.calib, encoding="utf-8"))
    fwd = {p["id"]: p["probs"] for p in predictions([r for r in map(json.loads, open(args.forward)) if r["id"] in by_id], calib)}
    rev = {p["id"]: p["probs"] for p in predictions([r for r in map(json.loads, open(args.reverse)) if r["id"] in by_id], calib)}
    single, averaged = [], []
    flips = 0
    for t in tasks:
        a = fwd.get(t["id"])
        b = rev.get(t["id"], a)
        single.append(official_outcome(t, jevscore.normalized(t, a) if a else None))
        if a and t["question"]["type"] == "choice":
            m = {k: (a[k] + b.get(k, a[k])) / 2 for k in a}
            flips += max(a, key=a.get) != max(b, key=b.get)
            averaged.append(official_outcome(t, jevscore.normalized(t, m)))
        else:
            averaged.append(single[-1])
    s1, s2 = jevscore.summary(single), jevscore.summary(averaged)
    n_choice = sum(t["question"]["type"] == "choice" for t in tasks)
    print(f"choice items where the two orders disagree on the answer: {flips}/{n_choice}")
    for label, s in (("single order", s1), ("two orders averaged", s2)):
        print(f"{label:22s} Capability {s['capability']:.1f}  I {s['intelligence']:.1f}  C {s['calibration']:.1f}  "
              f"choice competence {s['competence_per_type']['choice']:.1f}  choice calibration {s['calibration_per_type']['choice']:.1f}")
    bs = paired_bootstrap({"single": single, "averaged": averaged}, args.bootstrap)
    d, lo, hi = bs["averaged"]
    print(f"averaged minus single: {d:+.1f} [{lo:+.1f}, {hi:+.1f}] (95% CI, {args.bootstrap} resamples)")


if __name__ == "__main__":
    main()
