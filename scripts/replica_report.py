"""Report of a model's logits on the replica: Capability with the shipped calibration and with a calibration cross-fitted
on the replica itself (two halves by id), competence per tier, type and family, abstentions, and the reviewer's blind
accuracy per family next to the model's.

    uv run python scripts/replica_report.py --data data/dev/replica_v1.jsonl \
        --logits runs/pilot-31b-ab-bidir-full/replica_v1.jsonl --calib release/calibration.json \
        --report data/dev/replica_v1.report.json
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dopest_model import jevscore                                              # noqa: E402
from dopest_model.evaluate import fit_per_type, official_outcome, predictions  # noqa: E402


def half(task_id: str) -> int:
    return int(hashlib.sha1(task_id.encode()).hexdigest(), 16) % 2


def outcomes(tasks, rows, calib):
    P = {p["id"]: p["probs"] for p in predictions(rows, calib)}
    return [official_outcome(t, jevscore.normalized(t, P[t["id"]])) for t in tasks], P


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--logits", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--report", default=None, help="the select step's report (reviewer blind accuracy)")
    args = parser.parse_args()
    tasks = list({t["id"]: t for t in map(json.loads, open(args.data, encoding="utf-8"))}.values())
    by = {t["id"]: t for t in tasks}
    rows = [r for r in map(json.loads, open(args.logits, encoding="utf-8")) if r["id"] in by]
    calib = json.load(open(args.calib, encoding="utf-8"))
    r1 = lambda d: {k: (round(v, 1) if isinstance(v, (int, float)) else v) for k, v in d.items()}    # noqa: E731

    outs, P = outcomes(tasks, rows, calib)
    s = jevscore.summary(outs)
    print(f"kalibracja z release:        Capability {s['capability']:.1f}  I {s['intelligence']:.1f}  C {s['calibration']:.1f}")
    print(f"  competence per type {r1(s['competence_per_type'])} | calibration per type {r1(s['calibration_per_type'])}")

    # cross-fit: temperatures from one half, scored on the other, pooled
    cross = []
    temps = {}
    for h in (0, 1):
        fit_rows = [r for r in rows if half(r["id"]) != h]
        c = {"temperatures": fit_per_type(fit_rows, by), "noul_floor": calib.get("noul_floor", 0.801)}
        temps[h] = c["temperatures"]
        o, _ = outcomes([t for t in tasks if half(t["id"]) == h], [r for r in rows if half(r["id"]) == h], c)
        cross += o
    s2 = jevscore.summary(cross)
    print(f"kalibracja krzyżowa z repliki: Capability {s2['capability']:.1f}  I {s2['intelligence']:.1f}  C {s2['calibration']:.1f}")
    print("  temperatury z połówek:", {h: {k: round(v, 2) for k, v in t.items()} for h, t in temps.items()},
          "| release:", {k: round(v, 2) for k, v in calib["temperatures"].items()})

    acc = collections.defaultdict(lambda: [0, 0])
    abst = [0, 0]
    for t in tasks:
        p = P[t["id"]]
        qt = t["question"]["type"]
        if qt == "noul":
            yes = p.get("true", p.get("yes"))
            pred = "yes" if yes >= 0.5 else "no"
            abst[1] += 1
            abst[0] += 0.2 < yes < 0.8
        else:
            pred = max(p, key=p.get)
        hit = pred == t["expected"]
        for key in (("type", qt), ("tier", t["tier"]), ("family", t["family"]), ("author", t["provenance"]["author_model"])):
            acc[key][0] += hit
            acc[key][1] += 1
    for group in ("type", "tier", "author"):
        print(f"trafność per {group}:", {k: f"{a / n:.2f} ({n})" for (g, k), (a, n) in sorted(acc.items()) if g == group})
    print(f"abstencje noul: {abst[0]}/{abst[1]}")
    blind = {}
    if args.report and Path(args.report).exists():
        rep = json.load(open(args.report, encoding="utf-8"))
        blind = {f: c.get("blind_accuracy") for f, c in rep.get("families", {}).items()}
    print(f"\n{'rodzina':24s} {'n':>4s} {'model':>6s} {'recenzent (ślepo)':>18s}")
    for (g, fam), (a, n) in sorted(acc.items(), key=lambda kv: kv[0][1]):
        if g == "family":
            b = blind.get(fam)
            print(f"{fam:24s} {n:4d} {a / n:6.2f} {('%.2f' % b) if b is not None else '-':>18s}")


if __name__ == "__main__":
    main()
