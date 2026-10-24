"""Convert the JevBench-240 questions (huggingface.co/datasets/JevBench/jevbench: the metamorphic-coherence benchmark
of the JevBench org, Apache-2.0; 240 synthetic cases, 2,240 typed questions) to the item format evaluate.py reads.
Dev set only, never train on it.

    uv run python -m dopest_model.jb240 --out data/dev/jb240.jsonl

Independent of Benchmark Heaven's JevBench and of our training data: a different authoring route, 12 domains, short
states (median 1,000 characters). Questions with an unknown gold answer (302) are dropped. Every question carries the
tier "standard": the source has no tiers, so the scorer weights all of them alike.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO = "JevBench/jevbench"
FILE = "data/questions.jsonl"
SOURCE = "JevBench-240 (JevBench org on Hugging Face), synthetic case with gold"


def convert(row: dict) -> dict | None:
    """One source question as an item, or None when it has no usable gold."""
    gold = row.get("gold")
    if gold in (None, "", "unknown"):
        return None
    qtype = row.get("type")
    if qtype == "noul":
        criteria, labels = None, ["no", "yes"]
        expected = "yes" if str(gold).lower() == "true" else "no"
    elif qtype == "choice":
        criteria = {o["id"]: o["description"] for o in row["options"]}
        labels, expected = list(criteria), str(gold)
    elif qtype == "score":
        criteria = list(row["levels"])
        labels, expected = [str(i) for i in range(len(criteria))], str(int(gold))
    else:
        return None
    if expected not in labels:
        return None
    return {"id": f"jb240-{row['case_id']}-{row['question_key']}", "family": row["domain"], "tier": "standard",
            "state": row["state"], "question": {"type": qtype, "instructions": row["instructions"], "criteria": criteria},
            "labels": labels, "expected": expected, "group": None,
            "provenance": {"source": SOURCE, "license": "Apache-2.0"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/dev/jb240.jsonl")
    parser.add_argument("--src", default=None, help="local questions.jsonl (default: download from the Hub)")
    args = parser.parse_args()
    src = args.src
    if src is None:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(REPO, FILE, repo_type="dataset", local_dir="data/raw/jb240")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            if not line.strip():
                continue
            item = convert(json.loads(line))
            if item is None:
                counts["dropped"] += 1
                continue
            counts[item["question"]["type"]] += 1
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {sum(v for k, v in counts.items() if k != 'dropped')} items to {out}: {dict(counts)}")


if __name__ == "__main__":
    main()
