"""Zero-shot letter logits on OpenDecisions; resumable, logs prefill throughput.

    uv run python -m dopest_model.predict --model mlx-community/gemma-4-12B-it-bf16 --out runs/g4-12b/logits.jsonl

Writes raw (soft-capped, untempered) letter logits so temperatures can be fitted and applied later
without another forward pass. Items above --max-tokens are skipped and count as wrong when scored.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from .prompt import messages
from .readout import LetterReader

DEFAULT_DATA = "data/raw/opendecisions/opendecisions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None, help="directory with adapters.safetensors and adapter_config.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()][:args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}

    reader = LetterReader(args.model, adapter_path=args.adapter)
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)

    written = skipped = total_tokens = 0
    started = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as out:
        for n, item in enumerate(todo, 1):
            msgs, ids = messages(item)
            tokens = reader.prompt_tokens(msgs)
            if len(tokens) > args.max_tokens:
                skipped += 1
                continue
            t0 = time.perf_counter()
            logits = reader.letter_logits(tokens, len(ids))
            out.write(json.dumps({"id": item["id"], "type": item["question"]["type"], "ids": ids,
                                  "logits": [round(float(x), 5) for x in logits], "n_tokens": len(tokens),
                                  "seconds": round(time.perf_counter() - t0, 4)}) + "\n")
            written += 1
            total_tokens += len(tokens)
            if n % 100 == 0:
                out.flush()
                mx.clear_cache()
                rate = total_tokens / (time.perf_counter() - started)
                print(f"{n}/{len(todo)}  {rate:,.0f} prompt tok/s", flush=True)

    elapsed = time.perf_counter() - started
    print(f"wrote {written}, skipped {skipped} (> {args.max_tokens} tokens); "
          f"{total_tokens:,} tokens in {elapsed:,.0f} s = {total_tokens / max(elapsed, 1e-9):,.0f} tok/s")


if __name__ == "__main__":
    main()
