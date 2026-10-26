"""Second passes for many items at once, for offline experiments (the server decides one item at a time, as the
board measures it). One decode step of the 31B model is bound by reading the weights, so eight sequences cost about
as much as one; a batch of eight brings an experiment from half an hour to a few minutes.

Layout: every prompt is prefilled on its own through the single-stream Reasoner (its masks and cache are the
validated ones, and a long prompt never meets seven others in one attention), then its keys and values are copied
into a batch cache in which row i occupies the columns [L - n_i, L) of a group of L columns; the sliding-window rings
are rolled by the same shift, since their slot is the column modulo the window. Decoding then feeds one token per
row per step at the shared column L + t, with the row's own position n_i + t for the rotary embedding and masks built
per row from its first column. The result rows are the ones reason_eval writes, so scripts/reason_report.py reads
them unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .reason import STOP, Reasoned, Reasoner, split_reasoning


@dataclass
class Item:
    id: str
    tokens: list[int]
    n_options: int


def groups_of(items: list[Item], size: int) -> list[list[Item]]:
    """Items sorted by prompt length (longest first) in groups of `size`; the last group is padded by repeating its
    last item, so every group has the batch size the captured graph expects. Callers drop the repeats by position."""
    ordered = sorted(items, key=lambda it: -len(it.tokens))
    out = []
    for start in range(0, len(ordered), size):
        group = ordered[start:start + size]
        out.append(group + [group[-1]] * (size - len(group)))
    return out


class BatchReasoner:
    """Second passes in batches of `batch_size`. prompt_tokens, marker and letter readout come from the single
    Reasoner, which also does the prefills (eagerly, one prompt at a time)."""

    def __init__(self, model, tokenizer, batch_size: int = 8, max_cache_len: int = 6144, graphs: bool = True,
                 adapter_off: bool = True):
        import torch
        from transformers import StaticCache
        from .reason import _ring_layer_class
        self.single = Reasoner(model, tokenizer, max_cache_len=max_cache_len, graphs=False, adapter_off=adapter_off)
        self.batch, self.max_cache_len, self.window = batch_size, max_cache_len, self.single.window
        cfg = getattr(self.single.net.config, "text_config", None) or self.single.net.config
        self.cache = StaticCache(config=cfg, max_cache_len=max_cache_len)
        ring = _ring_layer_class()
        for i, layer in enumerate(self.cache.layers):
            if getattr(layer, "is_sliding", False):
                self.cache.layers[i] = ring(max_cache_len=max_cache_len, sliding_window=cfg.sliding_window)
        self.static = {"ids": torch.zeros((batch_size, 1), dtype=torch.long, device="cuda"),
                       "pos": torch.zeros((batch_size, 1), dtype=torch.long, device="cuda"),
                       "offset": torch.zeros(batch_size, dtype=torch.long, device="cuda")}
        self.graphs, self.graph = graphs, None

    def prompt_tokens(self, *args, **kwargs) -> list[int]:
        return self.single.prompt_tokens(*args, **kwargs)

    def max_prompt_tokens(self, max_new_tokens: int) -> int:
        return self.single.max_prompt_tokens(max_new_tokens)

    def _masks(self, pos, offset) -> dict:
        """Per-row masks of one decode step: pos [B, 1] holds each row's position, offset [B] its first column; the
        column being written is the same for every row. Full attention sees the row's columns up to it; the ring
        slot s holds the latest column congruent to s, valid when inside the row and the window."""
        current = pos[0, 0] + offset[0]
        cols = self.single.full_slots[None, :]
        full = (cols <= current) & (cols >= offset[:, None])
        col_of_slot = current - ((current - self.single.ring_slots) % self.window)
        sliding = (col_of_slot[None, :] >= offset[:, None]) & (col_of_slot[None, :] > current - self.window)
        return {"full_attention": full[:, None, None, :], "sliding_attention": sliding[:, None, None, :]}

    def _forward(self, ids, pos, offset):
        return self.single.net(input_ids=ids, position_ids=pos, attention_mask=self._masks(pos, offset),
                               past_key_values=self.cache, use_cache=True, logits_to_keep=1).logits[:, -1]

    def _step(self):
        """One decode step over the static buffers; returns float logits [B, V]."""
        if self.graph is None:
            return self._forward(self.static["ids"], self.static["pos"], self.static["offset"]).float()
        self.graph.replay()
        return self.static["logits"].float()

    def warm(self) -> float:
        """Initialise the batch cache (its tensors take the batch size from the first forward), capture the decode
        step as a CUDA graph, and run one short group. A failed capture leaves the eager step in place."""
        import torch
        t0 = time.perf_counter()
        with torch.inference_mode(), self.single._context():
            self._forward(self.static["ids"], self.static["pos"], self.static["offset"])      # allocates the cache
            if self.graphs:
                try:
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            self._forward(self.static["ids"], self.static["pos"], self.static["offset"])
                    torch.cuda.current_stream().wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        self.static["logits"] = self._forward(self.static["ids"], self.static["pos"], self.static["offset"])
                    torch.cuda.synchronize()
                    self.graph = graph
                except Exception as e:
                    print(f"CUDA graph capture of the batched step failed ({type(e).__name__}: {str(e)[:300]}); "
                          f"decoding eagerly", flush=True)
                    self.graph, self.graphs = None, False
            self.cache.reset()
        opts = [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}]
        tokens = self.prompt_tokens("Warm-up state: the invoice is from March.", "Is the invoice from March?", opts, 20)
        self.run([Item(f"warm{i}", tokens, 2) for i in range(self.batch)], max_new_tokens=8)
        return time.perf_counter() - t0

    def _load(self, group: list[Item]) -> tuple[list, list[float]]:
        """Prefill every prompt of the group on its own and place it in the batch cache; returns each row's next-token
        logits and prefill seconds."""
        import torch
        L = max(len(it.tokens) for it in group)
        firsts, seconds = [], []
        for i, it in enumerate(group):
            t0 = time.perf_counter()
            logits = self.single._prefill(it.tokens).float().clone()
            n, shift = len(it.tokens), (L - len(it.tokens)) % self.window
            for src, dst in zip(self.single.cache.layers, self.cache.layers):
                if getattr(dst, "is_sliding", False):
                    dst.keys[i].copy_(torch.roll(src.keys[0], shifts=shift, dims=1))
                    dst.values[i].copy_(torch.roll(src.values[0], shifts=shift, dims=1))
                else:
                    dst.keys[i, :, L - n:L].copy_(src.keys[0, :, :n])
                    dst.values[i, :, L - n:L].copy_(src.values[0, :, :n])
            torch.cuda.synchronize()
            firsts.append(logits)
            seconds.append(time.perf_counter() - t0)
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(L)
        self.static["offset"].copy_(torch.tensor([L - len(it.tokens) for it in group], device="cuda"))
        return firsts, seconds

    def run(self, items: list[Item], max_new_tokens: int) -> list[Reasoned]:
        """Second passes for every item (each prompt within max_prompt_tokens), in the items' order."""
        import math
        import torch
        marker, eos, variants, tok = self.single.marker_ids, self.single.eos, self.single.variants, self.single.tokenizer
        B, results = self.batch, {}
        with torch.inference_mode(), self.single._context():
            for group in groups_of(items, B):
                firsts, prefill_s = self._load(group)
                t0 = time.perf_counter()
                next_tok = [int(z.argmax()) for z in firsts]
                gen, text = [[] for _ in range(B)], [""] * B
                phase, mark_k, final = ["gen"] * B, [0] * B, [None] * B
                reached, appended, steps = [False] * B, [0] * B, 0
                while any(p != "done" for p in phase) and steps < max_new_tokens + len(marker) + 2:
                    self.static["ids"].copy_(torch.tensor([[t] for t in next_tok], device="cuda"))
                    self.static["pos"].copy_(torch.tensor([[len(it.tokens) + steps] for it in group], device="cuda"))
                    logits = self._step()
                    steps += 1
                    arg = logits.argmax(-1).tolist()
                    for i in range(B):
                        if phase[i] == "done":
                            continue
                        if phase[i] == "gen":
                            gen[i].append(next_tok[i])
                            text[i] = tok.decode(gen[i])
                            if STOP in text[i]:
                                reached[i] = True
                                if text[i].endswith(STOP):
                                    final[i], phase[i] = logits[i].clone(), "done"
                                    continue
                            if reached[i] or arg[i] in eos or len(gen[i]) >= max_new_tokens:
                                phase[i], mark_k[i], next_tok[i] = "mark", 0, marker[0]
                            else:
                                next_tok[i] = arg[i]
                        else:                                          # feeding the canonical marker, token by token
                            appended[i] += 1
                            if mark_k[i] == len(marker) - 1:
                                final[i], phase[i] = logits[i].clone(), "done"
                            else:
                                mark_k[i] += 1
                                next_tok[i] = marker[mark_k[i]]
                decode_s = time.perf_counter() - t0
                for i, it in enumerate(group):
                    if it.id in results:                               # a repeat that pads the last group
                        continue
                    z = final[i] if final[i] is not None else logits[i]
                    letters = [torch.logsumexp(z[v], 0).item() for v in variants[:it.n_options]]
                    rationale, _ = split_reasoning(text[i])
                    results[it.id] = Reasoned(letters, rationale, reached[i], len(gen[i]), appended[i],
                                              prefill_s[i] + decode_s / B, prefill_s[i], decode_s / max(1, steps))
        return [results[it.id] for it in items]
