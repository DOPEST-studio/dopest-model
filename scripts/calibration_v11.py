"""Calibration artifact of a run from several sources: one (data, logits) pair per question type, so the score
temperature can come from the sets that have many score items while choice and yes/no keep the public JevBench items.

    uv run python scripts/calibration_v11.py --run runs/v11-31b \
        --choice data/dev/jevbench_public.jsonl=jevbench_public.jsonl \
        --noul data/dev/jevbench_public.jsonl=jevbench_public.jsonl \
        --score data/dev/replica_v2.jsonl=replica_v2.jsonl,data/dev/gen_score_test.jsonl=gen_score_test.jsonl \
        --out release/calibration.json

Each source is <data file>=<logits file under the run directory>; several sources of one type are pooled. The yes/no
commit floor stays 0.801. The artifact records where every temperature came from.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dopest_model.calibrate import NOUL_FLOOR, fit_temperature   # noqa: E402
from dopest_model.evaluate import gold_index                      # noqa: E402


def pooled(run: Path, qtype: str, sources: str) -> tuple[float, dict]:
    logits, golds, used = [], [], {}
    for part in sources.split(","):
        data, rel = part.split("=", 1)
        tasks = {t["id"]: t for t in map(json.loads, open(data, encoding="utf-8")) if t["question"]["type"] == qtype}
        n = 0
        for r in map(json.loads, open(run / rel, encoding="utf-8")):
            t = tasks.get(r["id"])
            if t is None or r["type"] != qtype:
                continue
            logits.append(np.array(r["logits"]))
            golds.append(gold_index(t, r["ids"]))
            n += 1
        used[data] = n
    if not logits:
        raise SystemExit(f"no {qtype} items in {sources}")
    return float(fit_temperature(logits, golds)), used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--choice", required=True)
    parser.add_argument("--noul", required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--floor", type=float, default=NOUL_FLOOR)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    temps, fitted = {}, {}
    for qtype in ("choice", "noul", "score"):
        temps[qtype], fitted[qtype] = pooled(run, qtype, getattr(args, qtype))
    calib = {"temperatures": temps, "noul_floor": args.floor,
             "fitted_on": {"run": run.name, "items": fitted,
                           "note": "temperatures fitted per type on the listed sets (pooled when several); "
                                   "the yes/no commit floor is a fixed rule"}}
    Path(args.out).write_text(json.dumps(calib, indent=2) + "\n")
    print(json.dumps(calib, indent=1))


if __name__ == "__main__":
    main()
