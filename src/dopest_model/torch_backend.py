"""PyTorch / CUDA backend (Modal or any NVIDIA GPU) with the same letter readout, loss, option shuffling and
learning-rate schedule as the MLX code, so runs on both backends are comparable. Predictions are written in the
predict.py format, so evaluate.py and compare.py read them unchanged.

Model loading follows the OpenDecisions reference script for Gemma 4 (AutoModelForCausalLM, bf16, one forward pass
with logits_to_keep=1; Gemma's final-logit soft-capping happens inside the forward).
"""
from __future__ import annotations

import itertools
import json
import math
import random
import re
import time
from pathlib import Path

import torch

from .data import lr_factor, read_items, shuffled, target
from .prompt import LETTERS, messages


BIDIR_MODES = ("all", "full", "sliding")


def load(model_id: str, adapter: str | None = None, attn: str | None = None, bidir: str | None = None):
    """attn: transformers attention implementation ('sdpa' default, 'flex_attention' for block-sparse packed
    masks: FlashAttention 2/3 cannot be used, Gemma 4's global layers have head_dim 512).
    bidir: let every token of a prompt attend to the whole prompt (see segment_masks) in these layer types:
    'all', 'full' (global attention layers) or 'sliding'; None keeps the causal masks. An adapter trained with a
    mode must be run with the same mode."""
    if bidir is not None and bidir not in BIDIR_MODES:
        raise ValueError(f"bidir must be one of {BIDIR_MODES} or None, got {bidir!r}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kw = {"dtype": torch.bfloat16, "device_map": "cuda"} | ({"attn_implementation": attn} if attn else {})
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except (ValueError, KeyError):           # architectures without a causal-LM auto mapping
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.dopest_bidir = bidir
    return model, tokenizer


def _text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


def segment_masks(model, pos: torch.Tensor, mode: str) -> dict:
    """Attention masks for a packed sequence (positions restart at 0 per prompt) in which every token sees its whole
    prompt, forwards and backwards, and nothing of the other prompts. mode says which layer types get it ('all',
    'full', 'sliding'); the others keep their causal mask. Built with transformers' own utilities (block_sequence_ids,
    blockwise_overlay), so the result matches the attention implementation: a BlockMask for flex_attention, a 4D mask
    for sdpa. Torchcast Decision 4B and Perplexity Decider read decisions this way ("noncausal" readout)."""
    from transformers.masking_utils import (blockwise_overlay, create_causal_mask, create_sliding_window_causal_mask,
                                            sliding_window_overlay)
    cfg = _text_config(model)
    seg = (pos[0] == 0).long().cumsum(0)[None] - 1                 # prompt index of every token: 0, 1, 2, ...
    dummy = torch.zeros(pos.shape[0], pos.shape[1], 1, dtype=torch.bfloat16, device=pos.device)
    kw = dict(config=cfg, inputs_embeds=dummy, attention_mask=None, past_key_values=None, position_ids=pos)
    full = create_causal_mask(**kw, block_sequence_ids=seg) if mode in ("all", "full") else create_causal_mask(**kw)
    if mode in ("all", "sliding"):                                 # as Gemma 4 does for image blocks
        sliding = create_causal_mask(**kw, or_mask_function=blockwise_overlay(seg),
                                     and_mask_function=sliding_window_overlay(cfg.sliding_window))
    else:
        sliding = create_sliding_window_causal_mask(**kw)
    return {"full_attention": full, "sliding_attention": sliding}


def letter_token_ids(tokenizer) -> list[int]:
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"the letter {letter!r} is not a single token for this tokenizer")
        ids.append(encoded[0])
    return ids


def prompt_tokens(tokenizer, msgs: list[dict]) -> list[int]:
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(prompt, add_special_tokens=False)


def letter_logits(model, tokens: list[int], letter_ids: list[int]) -> torch.Tensor:
    """Soft-capped next-token logits of the given letters at the last position: (k,) float32."""
    if getattr(model, "dopest_bidir", None):
        return packed_letter_logits(model, [tokens], [letter_ids])[0]
    x = torch.tensor([tokens], device="cuda")
    out = model(input_ids=x, use_cache=False, logits_to_keep=1)
    return out.logits[0, -1].float()[letter_ids]


def packed_letter_logits(model, seqs: list[list[int]], letter_ids: list[list[int]]) -> list[torch.Tensor]:
    """Letter logits of several prompts from ONE forward pass over their concatenation.

    Positions restart at 0 for every prompt. With no attention mask and no cache, transformers' masking utilities
    detect the restarts (find_packed_sequence_indices) and build block-diagonal causal masks, sliding-window layers
    included, so each prompt attends only to itself exactly as in an unpacked pass; RoPE sees the same positions too.
    Logits are computed only at each prompt's last token (logits_to_keep as an index tensor).
    """
    lengths = [len(s) for s in seqs]
    ids = torch.tensor([list(itertools.chain.from_iterable(seqs))], device="cuda")
    pos = torch.cat([torch.arange(n, device="cuda") for n in lengths])[None]
    ends = torch.tensor(list(itertools.accumulate(lengths)), device="cuda") - 1
    mode = getattr(model, "dopest_bidir", None)
    masks = segment_masks(model, pos, mode) if mode else None          # None: transformers builds the causal masks
    out = model(input_ids=ids, position_ids=pos, attention_mask=masks, use_cache=False, logits_to_keep=ends)
    logits = out.logits[0].float()
    return [logits[i, ids_] for i, ids_ in enumerate(letter_ids)]


def packs(lengths: list[int], max_tokens: int) -> list[list[int]]:
    """Greedy packing in the given order: lists of indices whose lengths sum to at most max_tokens (a prompt
    longer than max_tokens gets a pack of its own)."""
    out, cur, size = [], [], 0
    for i, n in enumerate(lengths):
        if cur and size + n > max_tokens:
            out.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    if cur:
        out.append(cur)
    return out


def predict(model, tokenizer, items: list[dict], out_path: Path, max_tokens: int = 8192, pack_tokens: int = 0,
            log=print) -> None:
    """Letter logits for every item, one forward per item or (pack_tokens > 0) per pack of items."""
    letters = letter_token_ids(tokenizer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared = []
    for item in items:
        msgs, ids = messages(item)
        tokens = prompt_tokens(tokenizer, msgs)
        if len(tokens) <= max_tokens:
            prepared.append((item, ids, tokens))
    groups = packs([len(t) for _, _, t in prepared], pack_tokens) if pack_tokens else \
        [[i] for i in range(len(prepared))]
    started, total, done = time.perf_counter(), 0, 0
    with open(out_path, "w", encoding="utf-8") as out, torch.inference_mode():
        for group in groups:
            t0 = time.perf_counter()
            batch = [prepared[i] for i in group]
            if len(batch) == 1:
                zs = [letter_logits(model, batch[0][2], letters[:len(batch[0][1])])]
            else:
                zs = packed_letter_logits(model, [t for _, _, t in batch], [letters[:len(ids)] for _, ids, _ in batch])
            seconds = (time.perf_counter() - t0) / len(batch)
            for (item, ids, tokens), z in zip(batch, zs):
                out.write(json.dumps({"id": item["id"], "type": item["question"]["type"], "ids": ids,
                                      "logits": [round(v, 5) for v in z.cpu().tolist()], "n_tokens": len(tokens),
                                      "seconds": round(seconds, 4)}) + "\n")
                total += len(tokens)
            done += len(batch)
            if done // 500 != (done - len(batch)) // 500:
                log(f"{out_path.name}: {done}/{len(prepared)}, {total / (time.perf_counter() - started):,.0f} tok/s")
    log(f"{out_path.name}: done, {total:,} tokens, {total / (time.perf_counter() - started):,.0f} tok/s")


def lora_targets(model, n_layers: int) -> list[str]:
    """Linear modules inside the last n decoder layers of the language model (vision and audio towers excluded)."""
    by_layer = {}
    for name, module in model.named_modules():
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if m and isinstance(module, torch.nn.Linear) and not re.search(r"vision|audio", name):
            by_layer.setdefault(int(m.group(1)), []).append(name)
    keep = sorted(by_layer)[-n_layers:]
    return [name for i in keep for name in by_layer[i]]


def train(model_id: str, data: list[str], out_dir: Path, *, split: str | None = "train", limit: int | None = None,
          epochs: float = 1.0, rank: int = 32, alpha: float = 64.0, lora_layers: int = 16, lr: float = 5e-5,
          warmup: int = 20, accum: int = 16, max_tokens: int = 4096, soft: bool = False, train_temp: float = 4.0,
          seed: int = 0, save_every: int = 100, log=print, on_checkpoint=None, max_minutes: float | None = None,
          progress_every: int = 25, pack_tokens: int = 0, step_tokens: int = 16384,
          attn: str | None = None, grad_clip: float | None = None, bidir: str | None = None) -> Path:
    """LoRA training on the letter readout; returns the adapter directory (peft format).

    pack_tokens > 0 packs consecutive examples into sequences of at most that many tokens (one forward per pack,
    see packed_letter_logits); an optimizer step then covers about step_tokens tokens and the loss is the mean over
    the step's examples (accum is ignored). pack_tokens = 0 keeps one example per forward and accum examples per step.
    grad_clip caps the global gradient norm of a packed step (None: no clipping, as in the first pilots); LoRA on all
    48 layers diverged without it at lr 5e-5 (pilot-12b-ab-l48, step 170). bidir: see load().

    on_checkpoint() runs after every adapter save and every progress_every steps (Modal: commit the Volume, so a
    detached run's progress survives and can be fetched mid-run). max_minutes stops training early, saving the
    adapter, so a paid run always leaves time for evaluation inside its budget.
    """
    if pack_tokens:
        return _train_packed(model_id, data, out_dir, split=split, limit=limit, epochs=epochs, rank=rank,
                             alpha=alpha, lora_layers=lora_layers, lr=lr, warmup=warmup, max_tokens=max_tokens,
                             soft=soft, train_temp=train_temp, seed=seed, save_every=save_every, log=log,
                             on_checkpoint=on_checkpoint, max_minutes=max_minutes, progress_every=progress_every,
                             pack_tokens=pack_tokens, step_tokens=step_tokens, attn=attn, grad_clip=grad_clip,
                             bidir=bidir)
    from peft import LoraConfig, get_peft_model

    rng = random.Random(seed)
    torch.manual_seed(seed)
    items = read_items(data, split)
    rng.shuffle(items)
    items = items[:limit]

    model, tokenizer = load(model_id, bidir=bidir)
    for p in model.parameters():
        p.requires_grad_(False)
    targets = lora_targets(model, lora_layers)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                                             target_modules=targets))
    model.dopest_bidir = bidir
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    letters = letter_token_ids(tokenizer)
    total = max(1, math.ceil(len(items) * epochs / accum))
    warm = min(warmup, max(1, total // 10))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warm, total))
    log(f"{len(items)} items; LoRA rank {rank} on {len(targets)} modules in the last {lora_layers} layers: "
        f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable; {total} steps")

    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"
    log_file = open(out_dir / "train_log.jsonl", "a", encoding="utf-8")
    step, seen, tokens_seen = 0, 0, 0
    window = {"loss": 0.0, "hit": 0, "n": 0}
    order, started = [], time.perf_counter()
    opt.zero_grad(set_to_none=True)
    while step < total:
        if not order:
            order = list(items)
            rng.shuffle(order)
        item = shuffled(order.pop(), rng)
        msgs, ids = messages(item)
        tokens = prompt_tokens(tokenizer, msgs)
        if len(tokens) > max_tokens:
            continue
        tgt = torch.tensor(target(item, ids, soft), device="cuda")
        z = letter_logits(model, tokens, letters[:len(ids)])
        loss = -(tgt * torch.log_softmax(z / train_temp, dim=-1)).sum()
        (loss / accum).backward()
        seen += 1
        tokens_seen += len(tokens)
        window["loss"] += loss.item()
        window["hit"] += int(z.argmax().item() == tgt.argmax().item())
        window["n"] += 1
        if seen % accum:
            continue
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 5 == 0 or step == total:
            el = time.perf_counter() - started
            row = {"step": step, "of": total, "loss": round(window["loss"] / window["n"], 4),
                   "acc": round(window["hit"] / window["n"], 3), "lr": sched.get_last_lr()[0],
                   "tok_s": round(tokens_seen / el), "elapsed_min": round(el / 60, 1),
                   "eta_min": round(el / step * (total - step) / 60, 1)}
            log(json.dumps(row))
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()
            window = {"loss": 0.0, "hit": 0, "n": 0}
        if step % save_every == 0:
            model.save_pretrained(adapter_dir)
        if on_checkpoint and (step % save_every == 0 or step % progress_every == 0):
            on_checkpoint()
        if max_minutes and (time.perf_counter() - started) / 60 > max_minutes:
            log(f"time budget of {max_minutes} min reached at step {step}/{total}: stopping training early")
            break
    model.save_pretrained(adapter_dir)
    (out_dir / "train_meta.json").write_text(json.dumps(
        {"base_model": model_id, "backend": "torch", "examples": seen, "steps": step, "planned_steps": total,
         "rank": rank, "alpha": alpha,
         "lora_layers": lora_layers, "lr": lr, "warmup": warm, "accum": accum, "train_temp": train_temp,
         "soft": soft, "seed": seed, "bidir": bidir,
         "minutes": round((time.perf_counter() - started) / 60, 1)}, indent=1))
    log(f"done: {step} steps, {seen} examples, {(time.perf_counter() - started) / 60:.1f} min")
    if on_checkpoint:
        on_checkpoint()
    return adapter_dir


def train_bench(model_id: str, data: list[str], attn: str, n_packs: int = 12, pack_tokens: int = 8192,
                lora_layers: int = 16, log=print) -> dict:
    """Training throughput (forward + backward with LoRA) on the same packs, for one attention implementation."""
    model, tokenizer, params = _setup(model_id, 32, 64.0, lora_layers, 5e-5, log, attn=attn)
    letters = letter_token_ids(tokenizer)
    items = read_items(data, "train")
    random.Random(0).shuffle(items)
    ex = []
    for item in items[:n_packs * 12]:
        msgs, ids = messages(item)
        ex.append((prompt_tokens(tokenizer, msgs), letters[:len(ids)]))
    groups = packs([len(t) for t, _ in ex], pack_tokens)[:n_packs]
    timings, tokens = [], 0
    for k, group in enumerate(groups):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        zs = packed_letter_logits(model, [ex[i][0] for i in group], [ex[i][1] for i in group])
        sum(z.logsumexp(-1) for z in zs).backward()
        torch.cuda.synchronize()
        if k >= 2:                                   # the first packs pay compilation and allocator warm-up
            timings.append(time.perf_counter() - t0)
            tokens += sum(len(ex[i][0]) for i in group)
        model.zero_grad(set_to_none=True)
    return {"attn": attn, "tok_s": tokens / sum(timings), "packs": len(timings),
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9}


def _setup(model_id: str, rank: int, alpha: float, lora_layers: int, lr: float, log, attn: str | None = None,
           bidir: str | None = None):
    from peft import LoraConfig, get_peft_model
    model, tokenizer = load(model_id, attn=attn, bidir=bidir)
    for p in model.parameters():
        p.requires_grad_(False)
    targets = lora_targets(model, lora_layers)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                                             target_modules=targets))
    model.dopest_bidir = bidir
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"LoRA rank {rank} on {len(targets)} modules in the last {lora_layers} layers: "
        f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable; attention {attn or 'sdpa'}, bidir {bidir}")
    return model, tokenizer, params


def _train_packed(model_id: str, data: list[str], out_dir: Path, *, split, limit, epochs, rank, alpha, lora_layers,
                  lr, warmup, max_tokens, soft, train_temp, seed, save_every, log, on_checkpoint, max_minutes,
                  progress_every, pack_tokens, step_tokens, attn=None, grad_clip=None, bidir=None) -> Path:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    items = read_items(data, split)
    rng.shuffle(items)
    items = items[:limit]
    model, tokenizer, params = _setup(model_id, rank, alpha, lora_layers, lr, log, attn=attn, bidir=bidir)
    letters = letter_token_ids(tokenizer)

    def epoch_examples():
        """One epoch in a fresh random order with freshly shuffled options: (tokens, letter ids, target)."""
        order = list(items)
        rng.shuffle(order)
        out = []
        for item in order:
            item = shuffled(item, rng)
            msgs, ids = messages(item)
            tokens = prompt_tokens(tokenizer, msgs)
            if len(tokens) <= max_tokens:
                out.append((tokens, letters[:len(ids)], target(item, ids, soft)))
        return out

    first = epoch_examples()
    epoch_tokens = sum(len(t) for t, _, _ in first)
    # A step is a fixed number of packs. A token threshold (step_size >= step_tokens) over-counted: packs hold AT
    # MOST pack_tokens, so two of them almost never reach 2 x pack_tokens and every step took three packs, which
    # made "1 epoch" 1.36 epochs in the first packed pilots.
    per_step = max(1, round(step_tokens / pack_tokens))
    epoch_packs = len(packs([len(t) for t, _, _ in first], pack_tokens))
    total = max(1, math.ceil(epoch_packs * epochs / per_step))
    warm = min(warmup, max(1, total // 10))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warm, total))
    log(f"{len(first)} examples, {epoch_tokens:,} tokens, {epoch_packs} packs of <= {pack_tokens} tokens per "
        f"epoch; {per_step} packs per step: {total} steps")

    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"
    log_file = open(out_dir / "train_log.jsonl", "a", encoding="utf-8")
    step, seen, tokens_seen, step_examples, step_packs = 0, 0, 0, 0, 0
    window = {"loss": 0.0, "hit": 0, "n": 0}
    started = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    examples = first
    while step < total:
        for group in packs([len(t) for t, _, _ in examples], pack_tokens):
            batch = [examples[i] for i in group]
            zs = packed_letter_logits(model, [t for t, _, _ in batch], [l for _, l, _ in batch])
            loss_sum = 0.0
            for z, (_, _, tgt) in zip(zs, batch):
                t = torch.tensor(tgt, device="cuda")
                loss = -(t * torch.log_softmax(z / train_temp, dim=-1)).sum()
                loss_sum = loss_sum + loss
                window["loss"] += loss.item()
                window["hit"] += int(z.argmax().item() == t.argmax().item())
                window["n"] += 1
            loss_sum.backward()                     # the sum; divided by the step's example count below
            seen += len(batch)
            step_examples += len(batch)
            tokens_seen += sum(len(t) for t, _, _ in batch)
            step_packs += 1
            if step_packs < per_step:
                continue
            for p in params:
                if p.grad is not None:
                    p.grad.div_(step_examples)
            norm = torch.nn.utils.clip_grad_norm_(params, grad_clip if grad_clip else float("inf")).item()
            window["gnorm"] = max(window.get("gnorm", 0.0), norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            step_examples, step_packs = 0, 0
            if step % 5 == 0 or step == total:
                el = time.perf_counter() - started
                row = {"step": step, "of": total, "loss": round(window["loss"] / window["n"], 4),
                       "acc": round(window["hit"] / window["n"], 3), "lr": sched.get_last_lr()[0],
                       "max_grad_norm": round(window.get("gnorm", 0.0), 3),
                       "tok_s": round(tokens_seen / el), "elapsed_min": round(el / 60, 1),
                       "eta_min": round(el / step * (total - step) / 60, 1), "examples": seen}
                log(json.dumps(row))
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
                window = {"loss": 0.0, "hit": 0, "n": 0}
            if step % save_every == 0:
                model.save_pretrained(adapter_dir)
            if on_checkpoint and (step % save_every == 0 or step % progress_every == 0):
                on_checkpoint()
            if step >= total or (max_minutes and (time.perf_counter() - started) / 60 > max_minutes):
                break
        if step >= total or (max_minutes and (time.perf_counter() - started) / 60 > max_minutes):
            if step < total:
                log(f"time budget of {max_minutes} min reached at step {step}/{total}: stopping training early")
            break
        examples = epoch_examples()
    model.save_pretrained(adapter_dir)
    (out_dir / "train_meta.json").write_text(json.dumps(
        {"base_model": model_id, "backend": "torch", "packed": True, "pack_tokens": pack_tokens, "attn": attn,
         "bidir": bidir, "grad_clip": grad_clip,
         "step_tokens": step_tokens, "packs_per_step": per_step, "examples": seen, "steps": step,
         "planned_steps": total, "rank": rank,
         "alpha": alpha, "lora_layers": lora_layers, "lr": lr, "warmup": warm, "train_temp": train_temp,
         "soft": soft, "seed": seed, "minutes": round((time.perf_counter() - started) / 60, 1)}, indent=1))
    log(f"done: {step} steps, {seen} examples, {(time.perf_counter() - started) / 60:.1f} min")
    if on_checkpoint:
        on_checkpoint()
    return adapter_dir
