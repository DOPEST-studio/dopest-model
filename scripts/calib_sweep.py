"""Temperature sweep per question type against the board's own objective, not the likelihood: for every candidate
temperature, the type's chance-corrected competence (score uses the expected level) and its calibration part (choice:
ECE on hard items; yes/no: ECE of P(yes) with the commit floor; score: top-level ECE and nRPS), on several held-out
sets of one run. The temperature that maximises (competence + calibration) / 2, averaged over the sets with equal
weight, is the one to ship.

    uv run python scripts/calib_sweep.py --run runs/pilot-31b-ab-bidir-full \
        --sets data/raw/opendecisions/opendecisions.jsonl=opendecisions.jsonl,data/dev/jevbench_public.jsonl=jevbench_public.jsonl,...
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dopest_model import jevscore                                              # noqa: E402
from dopest_model.evaluate import official_outcome, predictions                # noqa: E402

GRID = [0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0]


def type_scores(tasks: list[dict], rows: list[dict], qtype: str, temp: float, floor: float) -> tuple[float, float]:
    calib = {"temperatures": {"choice": temp, "noul": temp, "score": temp}, "noul_floor": floor}
    P = {p["id"]: p["probs"] for p in predictions(rows, calib)}
    outs = [official_outcome(t, jevscore.normalized(t, P[t["id"]])) for t in tasks if t["question"]["type"] == qtype]
    cc = jevscore.competence(outs)
    cal = jevscore.calibration(outs)[1].get(qtype)
    return cc, cal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--sets", required=True, help="<data>=<logits file under the run>, comma-separated")
    parser.add_argument("--floor", type=float, default=0.801)
    parser.add_argument("--grid", default=",".join(map(str, GRID)))
    args = parser.parse_args()
    run = Path(args.run)
    grid = [float(x) for x in args.grid.split(",")]
    sets = []
    for part in args.sets.split(","):
        data, rel = part.split("=", 1)
        tasks = list({t["id"]: t for t in map(json.loads, open(data, encoding="utf-8"))}.values())
        by = {t["id"]: t for t in tasks}
        rows = [r for r in map(json.loads, open(run / rel, encoding="utf-8")) if r["id"] in by]
        have = {r["id"] for r in rows}
        sets.append((Path(data).stem, [t for t in tasks if t["id"] in have], rows))
    best = {}
    for qtype in ("choice", "noul", "score"):
        print(f"\n== {qtype}: (competence, calibration) per set, objective = mean over sets of (cc + cal) / 2")
        header = f"{'T':>5s} " + " ".join(f"{name[:14]:>14s}" for name, tasks, _ in sets if any(t['question']['type'] == qtype for t in tasks)) + f" {'objective':>9s}"
        print(header)
        table = []
        for temp in grid:
            cells, objs = [], []
            for name, tasks, rows in sets:
                if not any(t["question"]["type"] == qtype for t in tasks):
                    continue
                cc, cal = type_scores(tasks, rows, qtype, temp, args.floor)
                if cc is None or cal is None:
                    cells.append(f"{'-':>14s}")
                    continue
                cells.append(f"{cc:6.1f}/{cal:6.1f}")
                objs.append((cc + cal) / 2)
            obj = statistics.mean(objs) if objs else float("nan")
            table.append((temp, obj))
            print(f"{temp:5.2f} " + " ".join(cells) + f" {obj:9.2f}")
        best[qtype] = max(table, key=lambda x: x[1])[0]
    print("\nbest temperatures by the board objective:", best)


if __name__ == "__main__":
    main()
