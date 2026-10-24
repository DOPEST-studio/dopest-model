"""Word 8-gram overlap of generated items with the evaluation sets (OpenDecisions, JevBench public).

    uv run python scripts/contamination.py data/blockb/batch12_kept.jsonl [more.jsonl ...]

For every item: the share of its state's 8-grams (and question's) that occur anywhere in an evaluation item.
Prints the distribution and the worst items; exits non-zero when an item crosses --limit.
"""
import argparse
import json
import re
import sys

EVAL = ["data/raw/opendecisions/opendecisions.jsonl", "data/dev/jevbench_public.jsonl"]
WORD = re.compile(r"\w+", re.UNICODE)


def text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def grams(s: str, n: int = 8) -> set:
    w = WORD.findall(s.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--limit", type=float, default=0.5, help="flag items with a larger overlap share")
    args = parser.parse_args()
    ref = set()
    for path in EVAL:
        for r in map(json.loads, open(path, encoding="utf-8")):
            ref |= grams(text(r["state"]) + " " + r["question"]["instructions"])
    scores = []
    for path in args.files:
        for r in map(json.loads, open(path, encoding="utf-8")):
            g = grams(text(r["state"]) + " " + r["question"]["instructions"])
            scores.append((len(g & ref) / len(g) if g else 0.0, r["id"], text(r["state"])[:120]))
    scores.sort(reverse=True)
    n = len(scores)
    for cut in (0.05, 0.1, 0.2, 0.5):
        print(f"items with > {cut:.0%} of 8-grams in the eval sets: {sum(s > cut for s, _, _ in scores)} / {n}")
    print("worst:")
    for s, i, t in scores[:5]:
        print(f"  {s:.2f} {i} {t!r}")
    sys.exit(1 if scores and scores[0][0] > args.limit else 0)


if __name__ == "__main__":
    main()
