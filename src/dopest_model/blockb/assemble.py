"""Assemble the block B training set from verified items.

    uv run python -m dopest_model.blockb.assemble --kept data/blockb/pilot_kept.jsonl \
        data/blockb/batch12_kept.jsonl data/blockb/final_kept.jsonl --out data/blockb/blockb_v1.jsonl

Applies the wording fixes in specs.REWORDED (templates that matched OpenDecisions verbatim), drops duplicate ids,
keeps a twin only when its parent is in the set (so pairs stay whole and share a split), and prints the make-up of
the set: families, languages, question types, tiers, splits, JSON states, yes/no balance and twins.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .specs import REWORDED


def reworded(item: dict) -> dict:
    text = item["question"]["instructions"].strip()
    if text not in REWORDED:
        return item
    return dict(item, question=dict(item["question"], instructions=REWORDED[text]))


def assemble(items: list[dict]) -> list[dict]:
    by_id = {it["id"]: reworded(it) for it in items}
    return [it for it in by_id.values()
            if it["provenance"].get("twin_of") is None or it["provenance"]["twin_of"] in by_id]


def describe(items: list[dict]) -> None:
    def share(counter: collections.Counter) -> str:
        n = sum(counter.values())
        return ", ".join(f"{k} {v} ({v / n:.0%})" for k, v in counter.most_common())
    print(f"{len(items)} items")
    print("family:", share(collections.Counter(it["family"] for it in items)))
    print("type:", share(collections.Counter(it["question"]["type"] for it in items)))
    print("tier:", share(collections.Counter(it["tier"] for it in items)))
    print("split:", share(collections.Counter(it["split"] for it in items)))
    print("language (top 10):", share(collections.Counter(it["language"] for it in items)).split(", ")[:10])
    print("JSON states:", sum(isinstance(it["state"], dict) for it in items))
    print("yes/no:", dict(collections.Counter(it["expected"] for it in items if it["question"]["type"] == "noul")))
    print("twins:", dict(collections.Counter(it["provenance"].get("twin_kind") for it in items
                                             if it["provenance"].get("twin_of"))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kept", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    items = assemble([json.loads(l) for path in args.kept for l in open(path, encoding="utf-8") if l.strip()])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    describe(items)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
