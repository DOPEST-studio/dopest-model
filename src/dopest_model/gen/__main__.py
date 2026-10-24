"""Generate code-labelled decision items and print their composition.

    uv run python -m dopest_model.gen water_claim --cases 1000 --seed 0 --out data/gen/water_claim.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from . import GENERATORS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generator", choices=sorted(GENERATORS))
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    generate = GENERATORS[args.generator]
    items, seen = [], set()
    for i in range(args.cases):
        for it in generate(args.seed * 1_000_003 + i):
            if it["id"] not in seen:                  # identical items (small vocabularies) would leak across splits
                seen.add(it["id"])
                items.append(it)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    c = collections.Counter
    print(f"{len(items)} items from {args.cases} cases -> {out}")
    print("split:", dict(c(it["split"] for it in items)))
    print("type:", dict(c(it["question"]["type"] for it in items)))
    tags = c(it["provenance"]["rationale"].split("]")[0][1:] for it in items)
    print("role:", dict(c(t.split(":")[0] for t in tags.elements())))
    if any(":" in t and not t.startswith(("twin", "missing")) for t in tags):
        print("task:", dict(c(t.split(":")[1] for t in tags.elements() if t.startswith("base"))))
    print("gold (choice, top 12):", dict(c(str(it["expected"]).split("_sublimit")[0] for it in items
                                           if it["question"]["type"] == "choice").most_common(12)))
    print("gold (score):", dict(sorted(c(it["expected"] for it in items if it["question"]["type"] == "score").items())))
    print("gold (yes/no):", dict(c(it["expected"] for it in items if it["question"]["type"] == "noul")))
    pos = c(it["labels"].index(it["expected"]) for it in items if it["question"]["type"] == "choice")
    print("gold position (choice):", dict(sorted(pos.items())))
    lengths = sorted(len(it["state"]) for it in items)
    print(f"state chars: p10 {lengths[len(lengths) // 10]}, p50 {lengths[len(lengths) // 2]}, max {lengths[-1]}")


if __name__ == "__main__":
    main()
