"""Does reasoning at uncertainty help? For one set: the one-pass calibrated probabilities p1 of every item, the reasoned
letter logits p2 of the gated choice items, and the official scoring of the mix for several gates and weights.

    uv run python scripts/reason_report.py --data data/dev/jevbench_public.jsonl \
        --logits runs/pilot-31b-ab-bidir-full/jevbench_public.jsonl --reason runs/pilot-31b-ab-bidir-full/reason_jevbench_public.jsonl \
        --calib release/calibration.json [--tier hard]

p2 gets its own temperature, chosen on a grid by the mixed Capability of the gated items (reported, so overfitting is
visible: with few gated items prefer the temperature that is stable across sets).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dopest_model import jevscore                                              # noqa: E402
from dopest_model.evaluate import official_outcome, predictions                # noqa: E402
from dopest_model.reason import IN_USD_PER_M, OUT_USD_PER_M, blend, unsure    # noqa: E402

GATES = (0.6, 0.7, 0.75, 0.8, 0.9)
WEIGHTS = (0.5, 0.75, 1.0)
T2_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def seconds_per_item(row: dict) -> float:
    """Wall time of one second pass; older rows (batched HF generate) carry the batch time instead."""
    return row["seconds"] if "seconds" in row else row["batch_gen_s"] / row["batch"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--logits", required=True)
    parser.add_argument("--reason", default=None, help="rows of the second passes (modal_app.py::reason_eval)")
    parser.add_argument("--calib", required=True)
    parser.add_argument("--tier", default=None, help="restrict the set to one tier (the gate lists were built that way)")
    parser.add_argument("--max-prompt-tokens", type=int, default=None,
                        help="cost-aware gate: items whose second-pass prompt is longer than this keep the one-pass answer")
    parser.add_argument("--write-gate", default=None, metavar="PATH",
                        help=f"write the ids of the choice items whose one-pass top probability is below {max(GATES)}, one "
                             f"per line (the candidates of every gate in the grid; modal_app.py::reason_eval reads them)")
    parser.add_argument("--weights", default=",".join(map(str, WEIGHTS)), help="mixing weights to try, comma-separated")
    parser.add_argument("--cut-weight", type=float, default=None,
                        help="mixing weight for rationales that did not reach the marker (default: the same weight)")
    args = parser.parse_args()
    weights = tuple(float(w) for w in args.weights.split(","))
    if not args.reason and not args.write_gate:
        parser.error("give --reason rows to score, or --write-gate to list the candidates")
    tasks = list({t["id"]: t for t in map(json.loads, open(args.data, encoding="utf-8"))}.values())
    if args.tier:
        tasks = [t for t in tasks if t["tier"] == args.tier]
    by = {t["id"]: t for t in tasks}
    calib = json.load(open(args.calib, encoding="utf-8"))
    rows = [r for r in map(json.loads, open(args.logits, encoding="utf-8")) if r["id"] in by]
    P1 = {p["id"]: p["probs"] for p in predictions(rows, calib)}
    tasks = [t for t in tasks if t["id"] in P1]                    # items without one-pass logits cannot be scored
    if args.write_gate:
        ids = [t["id"] for t in tasks if t["question"]["type"] == "choice" and unsure(P1[t["id"]].values(), max(GATES))]
        Path(args.write_gate).write_text("".join(i + "\n" for i in ids), encoding="utf-8")
        print(f"{len(ids)} gate candidates -> {args.write_gate}")
        if not args.reason:
            return
    R2 = {r["id"]: r for r in map(json.loads, open(args.reason, encoding="utf-8")) if r["id"] in P1}
    if args.max_prompt_tokens is not None:
        R2 = {k: r for k, r in R2.items() if r["prompt_tokens"] <= args.max_prompt_tokens}
    reached = sum(r["reached_marker"] for r in R2.values())
    secs = sorted(seconds_per_item(r) for r in R2.values())
    print(f"items {len(tasks)} ({sum(t['question']['type'] == 'choice' for t in tasks)} choice); rationales {len(R2)}, "
          f"marker reached {reached}, mean new tokens {statistics.mean(r['new_tokens'] for r in R2.values()):.0f}, "
          f"seconds per item p50 {secs[len(secs) // 2]:.2f} max {secs[-1]:.2f}")

    def outcomes(gate: float, weight: float, t2: float) -> list[dict]:
        outs = []
        for t in tasks:
            p = P1[t["id"]]
            r = R2.get(t["id"])
            if t["question"]["type"] == "choice" and r is not None and unsure(p.values(), gate):
                w = weight if r["reached_marker"] or args.cut_weight is None else args.cut_weight
                p = dict(zip(r["ids"], blend([p[k] for k in r["ids"]], r["logits"], w, t2)))
            outs.append(official_outcome(t, jevscore.normalized(t, p)))
        return outs

    base = jevscore.summary(outcomes(0.0, 0.0, 1.0))
    print(f"one pass:            Capability {base['capability']:.2f}  I {base['intelligence']:.2f}  C {base['calibration']:.2f}  "
          f"choice CC {base['competence_per_type'].get('choice', float('nan')):.1f} / cal {base['calibration_per_type'].get('choice') or float('nan'):.1f}")
    best = None
    print(f"{'gate':>5s} {'w':>5s} {'T2':>4s} {'gated':>6s} {'Cap':>7s} {'I':>7s} {'C':>7s} {'choice CC':>10s} {'choice cal':>11s}")
    for gate in GATES:
        chosen = [R2[t["id"]] for t in tasks
                  if t["question"]["type"] == "choice" and t["id"] in R2 and unsure(P1[t["id"]].values(), gate)]
        n_gated = len(chosen)
        # The board bills the base model's list prices on the measured tokens of every pass, so the second pass costs
        # its whole prompt again plus the generated tokens (the board's cap is 0.0646 $/1000 decisions and the one-pass
        # server measured 0.049 $/1000 there).
        extra = sum((r["prompt_tokens"] + r.get("marker_tokens", 0)) * IN_USD_PER_M + r["new_tokens"] * OUT_USD_PER_M
                    for r in chosen) / 1e6 / len(tasks) * 1000
        print(f"gate {gate}: {n_gated} reasoned = {100 * n_gated / len(tasks):.1f}% of decisions, extra cost {extra:.4f} $/1000 decisions")
        for weight in weights:
            for t2 in T2_GRID:
                s = jevscore.summary(outcomes(gate, weight, t2))
                cap = s["capability"]
                if best is None or cap > best[0]:
                    best = (cap, gate, weight, t2)
                print(f"{gate:5.2f} {weight:5.2f} {t2:4.1f} {n_gated:6d} {cap:7.2f} {s['intelligence']:7.2f} {s['calibration']:7.2f} "
                      f"{s['competence_per_type'].get('choice', float('nan')):10.1f} {s['calibration_per_type'].get('choice') or float('nan'):11.1f}")
    print(f"\nbest: Capability {best[0]:.2f} (+{best[0] - base['capability']:.2f}) at gate {best[1]}, weight {best[2]}, T2 {best[3]}")


if __name__ == "__main__":
    main()
