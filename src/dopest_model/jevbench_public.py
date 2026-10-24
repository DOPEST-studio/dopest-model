"""Convert the public JevBench v1.2 splits (github.com/fstandhartinger/jevbench, 231 items) to the OpenDecisions
item format, so predict.py and evaluate.py run on them unchanged. Dev set only: never train on it.

    uv run python -m dopest_model.jevbench_public --out data/dev/jevbench_public.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SRC = Path("data/raw/jevbench/datasets/public")
TIER_BY_SPLIT = {"easy": "easy", "original": "standard", "hard": "hard"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/dev/jevbench_public.jsonl")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for split, tier in TIER_BY_SPLIT.items():
            for line in open(SRC / f"{split}.jsonl", encoding="utf-8"):
                if not line.strip():
                    continue
                item = json.loads(line)
                item["tier"] = tier
                item["split"] = split
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} items to {out}")


if __name__ == "__main__":
    main()
