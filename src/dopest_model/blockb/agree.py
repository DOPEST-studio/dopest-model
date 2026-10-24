"""Keep block B items that the blind verifiers confirm, and report the yield.

    uv run python -m dopest_model.blockb.agree --items data/blockb/pilot_raw.jsonl \
        --votes data/blockb/pilot_votes.jsonl --out data/blockb/pilot_kept.jsonl

Keep rule (adapted from OpenDecisions, where the author is checked by a panel): the teacher's label stays if both
verifiers pick it, or if one does and the verifiers' mean probability on it is at least 0.5. For ordered levels
(score), it stays when the verifiers' mean expected level is within 0.5 of the teacher's level. The label is never
switched to the verifiers' choice; an item they reject is dropped. The verifiers' mean distribution is stored as
`soft_probs` for the soft-label ablation.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

VERIFIERS = ("qwen", "gptoss")


def gold_key(item: dict) -> str:
    exp = str(item["expected"])
    if item["question"]["type"] == "noul":
        return "true" if exp == "yes" else "false"
    return exp


def decide(item: dict, votes: dict) -> tuple[bool, str, dict | None]:
    gold = gold_key(item)
    dists = [votes.get(v, {}).get("probs") for v in VERIFIERS]
    dists = [d for d in dists if d]
    if len(dists) < len(VERIFIERS):
        return False, "missing_vote", None
    mean = {k: sum(d.get(k, 0.0) for d in dists) / len(dists) for k in dists[0]}
    if item["question"]["type"] == "score":
        # ordered levels (scored by distance): keep when the panel's expected level rounds to the author's level
        close = abs(sum(int(k) * p for k, p in mean.items()) - int(gold)) <= 0.5
        return close, ("score_close" if close else "score_far"), mean
    picks = [max(d, key=d.get) for d in dists]
    agree = sum(p == gold for p in picks)
    if agree == len(dists):
        return True, "unanimous", mean
    if agree >= 1 and mean.get(gold, 0.0) >= 0.5:
        return True, "majority", mean
    return False, ("both_disagree" if agree == 0 else "weak_majority"), mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", nargs="+", required=True, help="one or more item files (duplicates by id dropped)")
    parser.add_argument("--votes", nargs="+", required=True, help="one or more vote files, merged by item id")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    items = list({it["id"]: it for path in args.items for it in map(json.loads, open(path, encoding="utf-8"))}.values())
    votes = collections.defaultdict(dict)
    for path in args.votes:
        for v in map(json.loads, open(path, encoding="utf-8")):
            votes[v["id"]].update({k: x for k, x in v.items() if k != "id"})
    outcome = collections.Counter()
    by_group = collections.defaultdict(lambda: [0, 0])
    kept = []
    for it in items:
        ok, why, mean = decide(it, votes.get(it["id"], {}))
        outcome[why] += 1
        key = f"{it['family']}/{it['question']['type']}"
        by_group[key][0] += ok
        by_group[key][1] += 1
        by_group[f"tier:{it['tier']}"][0] += ok
        by_group[f"tier:{it['tier']}"][1] += 1
        if ok:
            soft = {("yes" if k == "true" else "no" if k == "false" else k): round(p, 4) for k, p in mean.items()}
            kept.append(dict(it, label_source="teacher+panel", soft_probs=soft,
                             provenance=dict(it["provenance"], panel=why)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)}/{len(items)} ({len(kept) / max(1, len(items)):.0%}) -> {args.out}")
    print("outcomes:", dict(outcome))
    for k, (a, n) in sorted(by_group.items()):
        print(f"  {k:28s} {a:4d}/{n:<4d} {a / n:5.0%}")


if __name__ == "__main__":
    main()
