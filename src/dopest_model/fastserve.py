"""Levers for faster single-decision serving of the one-pass readout, measured with bench() on the public items.

The bidirectional readout needs its own attention masks, so the model cannot simply be handed to vLLM. Instead the
forward stays the one torch_backend uses and three independent levers can be applied on top of it:

  merge      PeftModel.merge_and_unload: the LoRA deltas folded into the base weights, no adapter matmuls. The one
             lever the server uses (serve.py, unless --reason keeps the adapter apart): 42% lower median latency,
             the same argmax on all 231 public items, probabilities within 0.02.
  quantise   torchao on every nn.Linear of the decoder (not the embeddings or the lm_head): 'fp8' (dynamic float8
             activations and weights, Hopper and newer), 'int8' (dynamic int8 activations, int8 weights: the A100
             path). Bench only: fp8 was slower than bf16 on the H100 and changed 6 of 231 answers.
  graphs     torch.compile(mode='reduce-overhead'): CUDA graphs need static shapes, so every prompt is padded to the
             next bucket length. The padding is a second packed segment (positions restart at 0), so with the
             existing segment masks the real prompt never attends to it, and only the first segment's logits are read.
             Bench only: 28% faster than merged alone, but 3 of 231 answers changed and the warm-up is long.

bench() runs the same prompts through a prepared model one by one and reports latency and agreement with reference
logits, which is how a variant earns its place in the server. set_lora_scale() scales the adapter for interpolation
experiments (modal_app.py::scale_sweep).
"""
from __future__ import annotations

import bisect
import statistics
import time

BUCKETS = (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 40960)
QUANT_MODES = ("fp8", "int8")


def bucket_for(n_tokens: int, buckets: tuple = BUCKETS) -> int:
    """The smallest bucket that holds the prompt plus at least one padding token (the padding segment needs a start)."""
    i = bisect.bisect_left(buckets, n_tokens + 1)
    return buckets[i] if i < len(buckets) else n_tokens + 1


def padded_segments(tokens: list[int], pad_id: int, bucket: int) -> list[list[int]]:
    """The prompt and a padding segment that together fill the bucket exactly."""
    pad = bucket - len(tokens)
    if pad < 1:
        raise ValueError(f"bucket {bucket} does not hold {len(tokens)} tokens plus padding")
    return [list(tokens), [pad_id] * pad]


def merge_adapter(model):
    """A plain base model with the adapter folded in (keeps the readout mode flag)."""
    mode = getattr(model, "dopest_bidir", None)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.dopest_bidir = mode
    return model


def quantise(model, mode: str):
    """torchao quantisation of the decoder's linear layers; the embedding and the lm_head stay in bf16."""
    if mode not in QUANT_MODES:
        raise ValueError(f"mode must be one of {QUANT_MODES}, got {mode!r}")
    from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig, Int8DynamicActivationInt8WeightConfig,
                                      quantize_)
    config = Float8DynamicActivationFloat8WeightConfig() if mode == "fp8" else Int8DynamicActivationInt8WeightConfig()

    import torch

    def is_decoder_linear(module, name: str) -> bool:
        return isinstance(module, torch.nn.Linear) and "lm_head" not in name and "embed" not in name

    quantize_(model, config, filter_fn=is_decoder_linear)
    return model


def enable_graphs(model):
    """CUDA graphs via torch.compile; call warm() for every bucket in use before timing."""
    import torch
    model.forward = torch.compile(model.forward, mode="reduce-overhead", dynamic=False)
    return model


class Reader:
    """Letter logits of one prompt through a prepared model, padded to a bucket when graphs are on."""

    def __init__(self, model, pad_id: int, graphs: bool):
        self.model, self.pad_id, self.graphs = model, pad_id, graphs

    def __call__(self, tokens: list[int], letter_ids: list[int]):
        import torch
        from .torch_backend import packed_letter_logits
        with torch.inference_mode():
            if not self.graphs:
                return packed_letter_logits(self.model, [tokens], [letter_ids])[0]
            segs = padded_segments(tokens, self.pad_id, bucket_for(len(tokens)))
            return packed_letter_logits(self.model, segs, [letter_ids, letter_ids[:1]])[0]

    def warm(self, lengths: list[int], letter_ids: list[int]) -> list[int]:
        """Run every bucket the given lengths need once (compilation and graph capture happen here)."""
        done = []
        for b in sorted({bucket_for(n) for n in lengths}):
            self([self.pad_id] * (b - 1), letter_ids)
            done.append(b)
        return done


def bench(reader: Reader, prompts: list[tuple[str, list[int], list[int]]], reference: dict[str, list[float]] | None,
          temperature: float = 1.0) -> dict:
    """Serial latency over the prompts (id, tokens, letter ids) and agreement with reference logits of the same
    prompts: share of equal argmax and the largest probability difference at the given temperature."""
    import torch
    seconds, same, max_dp, out = [], 0, 0.0, {}
    for item_id, tokens, letters in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = reader(tokens, letters)
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - t0)
        z = logits.float().cpu()
        out[item_id] = z.tolist()
        if reference and item_id in reference:
            r = torch.tensor(reference[item_id], dtype=torch.float32)
            same += int(z.argmax().item() == r.argmax().item())
            p, q = torch.softmax(z / temperature, 0), torch.softmax(r / temperature, 0)
            max_dp = max(max_dp, float((p - q).abs().max()))
    seconds.sort()
    n = len(seconds)
    return {"n": n, "p50_s": seconds[n // 2], "p95_s": seconds[min(n - 1, int(0.95 * n))], "mean_s": statistics.mean(seconds),
            "same_argmax": same, "compared": sum(1 for i, _, _ in prompts if reference and i in reference),
            "max_prob_diff": round(max_dp, 4), "logits": out}


def set_lora_scale(model, factor: float) -> int:
    """Scale every LoRA delta of an unmerged PEFT model by `factor` relative to its trained strength (WiSE-FT style:
    W = W0 + factor * dW). Returns the number of LoRA layers touched. factor 1.0 restores the trained adapter."""
    n = 0
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict) or not hasattr(module, "lora_A"):
            continue
        base = getattr(module, "_dopest_base_scaling", None)
        if base is None:
            base = dict(scaling)
            module._dopest_base_scaling = base
        for name, value in base.items():
            scaling[name] = value * factor
        n += 1
    return n
