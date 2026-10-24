"""LoRA decision training on the letter readout (MLX, Apple silicon).

    uv run python -m dopest_model.train --model mlx-community/gemma-4-12B-it-bf16 \
        --data data/gen/water_claim.jsonl data/gen/temporal_numeric.jsonl --out runs/pilot-12b

Loss: cross-entropy over the item's option letters only, on the same soft-capped readout that evaluation uses.
Targets are one-hot on the gold option, or the item's target_probs with --soft. Choice options are shuffled per
example (and again every epoch), so the gold position carries no signal; yes/no and score keep their order, as at
evaluation. Adapters are saved in the mlx-lm format: predict.py --adapter <out> loads them unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers

from .data import read_items, shuffled, target
from .prompt import messages
from .readout import letter_logits, letter_token_ids, prompt_tokens, text_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--split", default="train", help="item split to train on ('' for all items)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None, help="at most this many items (after shuffling)")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora-layers", type=int, default=16,
                        help="LoRA on the last N layers only: backward stops there (Mac: ~3x faster than all 48)")
    parser.add_argument("--scale", type=float, default=2.0, help="LoRA scale (alpha / rank; Jeff: 64 / 32)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=20, help="optimizer steps of linear warm-up")
    parser.add_argument("--accum", type=int, default=16, help="examples per optimizer step")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--soft", action="store_true", help="train on target_probs where items have them")
    parser.add_argument("--train-temp", type=float, default=4.0,
                        help="divide the letter logits by this in the loss. Zero-shot letter logits are far too "
                             "sharp (fitted T ~4 for Gemma 4 12B); at T = 1 the cheapest way down the loss is to "
                             "shrink every logit, and LoRA did that at the cost of discrimination (pilot v1)")
    parser.add_argument("--save-every", type=int, default=100, help="optimizer steps between checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-gb", type=float, default=4.0,
                        help="MLX buffer-cache limit: every new prompt length allocates new buffers, and an unbounded "
                             "cache grew to 100+ GB and pushed the Mac into swap")
    args = parser.parse_args()

    mx.set_cache_limit(int(args.cache_gb * 1024 ** 3))
    rng = random.Random(args.seed)
    mx.random.seed(args.seed)
    items = read_items(args.data, args.split or None)
    rng.shuffle(items)
    items = items[:args.limit]

    model, tokenizer = load(args.model)
    model.freeze()
    lora = {"rank": args.rank, "scale": args.scale, "dropout": 0.0}
    n_layers = min(args.lora_layers, len(model.layers))
    linear_to_lora_layers(model, n_layers, lora)
    lm = text_model(model)
    letters = letter_token_ids(tokenizer)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"{len(items)} items; LoRA rank {args.rank} on the last {n_layers} of {len(model.layers)} layers: "
          f"{n_train / 1e6:.1f}M trainable", flush=True)

    total = max(1, math.ceil(len(items) * args.epochs / args.accum))
    warm = min(args.warmup, max(1, total // 10))
    schedule = optim.join_schedules([optim.linear_schedule(args.lr * 0.01, args.lr, warm),
                                     optim.cosine_decay(args.lr, max(1, total - warm), 0.0)], [warm])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.0)

    def loss_fn(model, tokens, letter_ids, tgt):
        z = letter_logits(lm, tokens, letter_ids)[0].astype(mx.float32)
        return -(tgt * nn.log_softmax(z / args.train_temp)).sum(), z

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "num_layers": n_layers, "lora_parameters": lora,
         "base_model": args.model, "train_args": vars(args)}, indent=1))
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")

    def save(tag: str) -> None:
        mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
        print(f"saved adapters ({tag})", flush=True)

    step, seen, tokens_seen, grads = 0, 0, 0, None
    window = {"loss": 0.0, "hit": 0, "n": 0}
    started = time.perf_counter()
    order = []
    while step < total:
        if not order:
            order = list(items)
            rng.shuffle(order)
        item = shuffled(order.pop(), rng)
        msgs, ids = messages(item)
        tokens = prompt_tokens(tokenizer, msgs)
        if len(tokens) > args.max_tokens:
            continue
        tgt = mx.array(target(item, ids, args.soft))
        (loss, z), g = loss_and_grad(model, mx.array([tokens]), mx.array(letters[:len(ids)]), tgt)
        grads = g if grads is None else tree_map(mx.add, grads, g)
        mx.eval(loss, z, grads)
        seen += 1
        window["loss"] += loss.item()
        window["hit"] += int(mx.argmax(z).item() == int(mx.argmax(tgt).item()))
        window["n"] += 1
        tokens_seen += len(tokens)
        if seen % args.accum:
            continue
        optimizer.update(model, tree_map(lambda x: x / args.accum, grads))
        mx.eval(model.trainable_parameters(), optimizer.state)
        grads = None
        step += 1
        if step % 5 == 0 or step == total:
            el = time.perf_counter() - started
            row = {"step": step, "of": total, "loss": round(window["loss"] / window["n"], 4),
                   "acc": round(window["hit"] / window["n"], 3), "lr": float(optimizer.learning_rate),
                   "tok_s": round(tokens_seen / el), "elapsed_min": round(el / 60, 1),
                   "eta_min": round(el / step * (total - step) / 60, 1)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            window = {"loss": 0.0, "hit": 0, "n": 0}
        if step % args.save_every == 0:
            save(f"step {step}")
            mx.clear_cache()
    save("final")
    print(f"done: {step} steps, {seen} examples, {(time.perf_counter() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
