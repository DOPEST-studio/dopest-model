"""Assemble the replica from the stage outputs.

    uv run python -m dopest_model.replica.assemble collect --rows data/replica/author_gptoss.jsonl,data/replica/author_qwen.jsonl \
        --out data/replica/items.jsonl
    uv run python -m dopest_model.replica.assemble select --items data/replica/items.jsonl \
        --verdicts data/replica/verdict_qwen.jsonl,data/replica/verdict_gptoss.jsonl \
        --blind data/replica/blind_qwen.jsonl,data/replica/blind_gptoss.jsonl --out data/dev/replica_v1.jsonl

collect: the valid items of the author rows, one per brief. select: the items the reviewer accepted, drawn per
(type, tier, family) cell up to the sealed quota, with a report (acceptance, the reviewer's blind accuracy, gold balance).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from .spec import quotas


def read(paths: str) -> list[dict]:
    rows = []
    for p in paths.split(","):
        if p and Path(p).exists():
            rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows


def collect(rows: list[dict]) -> list[dict]:
    seen, items = set(), []
    for r in rows:
        it = r.get("item") if r.get("ok") else None
        if it and it["id"] not in seen:
            seen.add(it["id"])
            items.append(it)
    return items


def select(items: list[dict], verdicts: list[dict], blind: list[dict], seed: int = 0) -> tuple[list[dict], dict]:
    """Accepted items per cell up to the quota (random within the cell), and the report."""
    verdict = {v["id"]: v for v in verdicts if v.get("ok")}
    blind_by = {b["id"]: b for b in blind}
    rng = random.Random(seed)
    by_cell = collections.defaultdict(list)
    for it in items:
        if verdict.get(it["id"], {}).get("verdict") == "accept":
            by_cell[(it["question"]["type"], it["tier"], it["family"])].append(it)
    chosen, short = [], {}
    for cell, n in sorted(quotas().items()):
        pool = by_cell.get(cell, [])
        rng.shuffle(pool)
        chosen += pool[:n]
        if len(pool) < n:
            short["/".join(cell)] = f"{len(pool)}/{n}"
    fam = collections.defaultdict(lambda: collections.Counter())
    for it in items:
        c = fam[it["family"]]
        c["valid"] += 1
        v = verdict.get(it["id"], {}).get("verdict")
        c["reviewed"] += v is not None
        c["accepted"] += v == "accept"
        b = blind_by.get(it["id"])
        if v == "accept" and b and b.get("ok"):
            c["blind_n"] += 1
            c["blind_hit"] += b["answer"] == it["expected"]
    for it in chosen:
        fam[it["family"]]["chosen"] += 1
    report = {"chosen": len(chosen), "wanted": sum(quotas().values()), "short_cells": short,
              "families": {f: dict(c, blind_accuracy=round(c["blind_hit"] / c["blind_n"], 3) if c["blind_n"] else None)
                           for f, c in sorted(fam.items())},
              "gold_balance": {
                  "noul_yes_share": _share([it["expected"] == "yes" for it in chosen if it["question"]["type"] == "noul"]),
                  "choice_gold_position": dict(collections.Counter(
                      it["labels"].index(it["expected"]) for it in chosen if it["question"]["type"] == "choice")),
                  "score_levels": dict(collections.Counter(
                      it["expected"] for it in chosen if it["question"]["type"] == "score"))}}
    return chosen, report


def _share(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 3) if flags else None


TRAIN_KEEP_REJECTS = ("leak", "format")       # a leaky item is only easier; a wrong gold would teach errors


def select_training(items: list[dict], verdicts: list[dict]) -> tuple[list[dict], dict]:
    """Items for a training round: accepted ones, plus rejects whose only fault is a leak or the format. Items with
    no verdict, a wrong or disputed gold, outside knowledge or an unclassified reason are left out."""
    verdict = {v["id"]: v for v in verdicts if v.get("ok")}
    kept, why = [], collections.Counter()
    for it in items:
        v = verdict.get(it["id"])
        if v is None:
            why["no verdict"] += 1
            continue
        if v["verdict"] == "accept" or v.get("reason_class") in TRAIN_KEEP_REJECTS:
            tag = "accept" if v["verdict"] == "accept" else f"reject:{v['reason_class']}"
            kept.append(dict(it, split="train", provenance=dict(it["provenance"], review=tag)))
            why[tag] += 1
        else:
            why[f"dropped:{v.get('reason_class') or 'unclassified'}"] += 1
    return kept, dict(why)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("collect", "select", "train"))
    parser.add_argument("--rows", default="", help="collect: author row files, comma-separated")
    parser.add_argument("--items", default="data/replica/items.jsonl")
    parser.add_argument("--verdicts", default="")
    parser.add_argument("--blind", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.step == "collect":
        items = collect(read(args.rows))
        with open(out, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        cells = collections.Counter(f"{it['question']['type']}/{it['tier']}" for it in items)
        print(f"{len(items)} valid items -> {out}; by type/tier: {dict(sorted(cells.items()))}")
    elif args.step == "train":
        kept, why = select_training(read(args.items), read(args.verdicts))
        with open(out, "w", encoding="utf-8") as f:
            for it in kept:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"{len(kept)} training items -> {out}; {why}")
    else:
        items = read(args.items)
        chosen, report = select(items, read(args.verdicts), read(args.blind), args.seed)
        with open(out, "w", encoding="utf-8") as f:
            for it in chosen:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        with open(out.with_suffix(".report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"{report['chosen']}/{report['wanted']} items -> {out}; short cells: {len(report['short_cells'])}; "
              f"report -> {out.with_suffix('.report.json')}")


if __name__ == "__main__":
    main()
