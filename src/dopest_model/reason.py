"""Reasoning at uncertainty for choice questions: when the one-pass readout is unsure, the model writes a short
rationale (causal generation, adapter off) that ends with "Answer:", the option letters are read right after it, and
the two distributions are mixed. Gate, weight and temperature were chosen offline with scripts/reason_report.py on
OpenDecisions hard, the public JevBench items and our replica of the sealed set (results in the model card).

Why the base model writes the rationale: the adapter was trained only at the answer position under a bidirectional
mask, so its generations are untested; the base model's causal readout after its own reasoning is an independent
view of the question. Why one short rationale under a cost budget: the board bills the base model's list prices on
the tokens of every pass (the second pass reads the state again, and the unsure items are the long ones), fails a
decision slower than 10 s, and ranks only systems within 2x the reference cost. The Budget keeps the pooled extra
cost of a whole run under a fixed amount whatever the mix of items.

Serving: Reasoner handles one decision at a time with a static KV cache and the decode step replayed as a CUDA graph.
The eager transformers loop measured 95 ms per token on an A100 80 GB (Python and kernel launches for 60 layers) and
torch.compile made it slower still (graph breaks, recompilations on the cache's Python-int bookkeeping); the captured
graph replays the step in one launch, bound by the weight reads: 76 ms per token on the A100, 49 ms on an H100.
"""
from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass

import numpy as np

from .calibrate import softmax
from .prompt import LETTERS

REASON_SYSTEM = ("Apply the supplied criterion to the supplied evidence. First write a brief reasoning of at most {words} "
                 "words{sentences} that names the decisive facts and rules out the tempting wrong options. Then, on a new "
                 "line, write \"Answer:\" followed by the uppercase letter of the one listed option that follows.")
STOP = "Answer:"
IN_USD_PER_M, OUT_USD_PER_M = 0.09, 0.34       # OpenRouter list prices of google/gemma-4-31b-it: what the board bills


@dataclass(frozen=True)
class Policy:
    """When to reason and how to mix (tuned offline), plus the cost rules of the second pass."""
    gate: float = 0.8                    # reason when the one-pass top probability is below this
    types: tuple = ("choice", "score")   # question types that get the second pass (levels of a score act as options)
    gates: dict | None = None            # per-type gates overriding `gate`, e.g. {"choice": 0.75}
    weight: float = 0.5                  # weight of the reasoned distribution in the mix
    temperature: float = 3.0             # temperature of the reasoned letter logits
    words: int = 40                      # rationale length asked for: a complete short rationale kept the gain of an
    sentences: int | None = 2            # 80-word one, a rationale cut mid-sentence lost most of it; the sentence cap
    max_new_tokens: int = 80             # is what the model follows. Hard stop of the rationale in tokens.
    max_seconds: float = 6.5             # wall-clock budget of the whole request, first readout included, measured on
    #                                      the serving machine: the board fails a decision slower than 10 s measured
    min_new_tokens: int = 40             # a second pass that could not fit this many tokens in the budget is skipped
    max_prompt_tokens: int = 8000        # longer second-pass prompts keep the one-pass answer
    budget_usd_per_1000: float = 0.014   # extra cost allowed, averaged over every decision served

    @classmethod
    def from_calibration(cls, calibration: dict, **overrides) -> "Policy":
        """Defaults, then the calibration file's optional 'reasoning' section, then explicit overrides (None: keep)."""
        values = {k: v for k, v in calibration.get("reasoning", {}).items() if k in cls.__dataclass_fields__}
        values = values | {k: v for k, v in overrides.items() if v is not None}
        if "types" in values:
            values["types"] = tuple(values["types"])                  # JSON lists become tuples
        return cls(**values)

    def gate_for(self, qtype: str) -> float:
        return (self.gates or {}).get(qtype, self.gate)

    def cost(self, prompt_tokens: int, new_tokens: int) -> float:
        """What the board bills for one second pass, in dollars."""
        return (prompt_tokens * IN_USD_PER_M + new_tokens * OUT_USD_PER_M) / 1e6


class Budget:
    """Running account of the extra cost. A second pass is allowed when the total spent, including it, stays within
    the per-decision budget times the decisions served so far (this one included): spent <= per_decision * decisions
    holds after every decision, so the extra cost of any run, pooled over its decisions, never exceeds the budget,
    whatever the order and mix of its items. Early decisions therefore wait until the budget has accrued; that only
    shifts which unsure items get the second pass."""

    def __init__(self, usd_per_decision: float):
        self.per_decision = usd_per_decision
        self.spent = 0.0
        self.decisions = 0

    def allows(self, usd: float) -> bool:
        return self.spent + usd <= self.per_decision * (self.decisions + 1)

    def charge(self, usd: float) -> None:
        self.spent += usd

    def tick(self) -> None:
        """One more decision served, reasoned or not."""
        self.decisions += 1


def reason_messages(state, instructions: str, opts: list[dict], words: int = 80, sentences: int | None = None) -> list[dict]:
    """Chat messages of the second pass; `sentences` adds a sentence cap, which models follow more closely than a
    word count (the 60-word prompt still produced 90 tokens on average)."""
    payload = {"evidence": state, "criterion": instructions,
               "options": [{"letter": LETTERS[i], "description": o["description"]} for i, o in enumerate(opts)]}
    cap = f" and {sentences} sentence{'s' if sentences != 1 else ''}" if sentences else ""
    return [{"role": "system", "content": REASON_SYSTEM.format(words=words, sentences=cap)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def split_reasoning(text: str) -> tuple[str, bool]:
    """(the rationale up to and excluding the answer marker, whether the marker was reached)."""
    i = text.find(STOP)
    if i < 0:
        return text.strip(), False
    return text[:i].strip(), True


def letter_variants(tokenizer) -> list[list[int]]:
    """For every letter, the token ids that may follow 'Answer:' : the bare letter and the space-prefixed one."""
    out = []
    for letter in LETTERS:
        ids = []
        for form in (letter, " " + letter):
            enc = tokenizer.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        out.append(sorted(set(ids)))
    return out


def unsure(probs, gate: float) -> bool:
    """Does the one-pass distribution call for a second pass?"""
    return max(probs) < gate


def blend(probs: list[float], logits: list[float], weight: float, temperature: float) -> list[float]:
    """(1 - weight) * one-pass probabilities + weight * softmax(reasoned letter logits / temperature), option-wise."""
    p2 = softmax(np.array(logits), temperature).tolist()
    return [(1 - weight) * p + weight * q for p, q in zip(probs, p2)]


@dataclass
class Reasoned:
    """One second pass. logits: the letter logits after the marker (log-sum over the bare and space-prefixed letter
    tokens, soft-capped by the model). new_tokens were generated (billed as output); marker_tokens were appended by
    the reader when the model did not end on the marker (billed as input). prefill_seconds and step_seconds (mean per
    decoded token) split the wall time."""
    logits: list[float]
    rationale: str
    reached_marker: bool
    new_tokens: int
    marker_tokens: int
    seconds: float
    prefill_seconds: float = 0.0
    step_seconds: float = 0.0


def _ring_layer_class():
    """transformers' StaticSlidingWindowLayer keeps its length as a Python int to pick between its update branches
    and to size its masks, so a captured or compiled decode step would freeze or specialise on one length. This layer
    stores the window as a ring (slot = position mod window) with tensor bookkeeping only; Reasoner builds the
    matching masks itself, so the layer's mask-size methods are never consulted."""
    import torch
    from transformers.cache_utils import StaticLayer

    class RingLayer(StaticLayer):
        is_sliding = True

        def __init__(self, max_cache_len: int, sliding_window: int, **kwargs):
            super().__init__(max_cache_len=min(sliding_window, max_cache_len))

        def update(self, key_states, value_states, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            n = key_states.shape[-2]
            keep = min(n, self.max_cache_len)                      # a prefill longer than the window keeps its tail
            positions = torch.arange(n - keep, n, device=self.device) + self.cumulative_length
            slots = positions % self.max_cache_len
            self.keys.index_copy_(2, slots, key_states[:, :, n - keep:])
            self.values.index_copy_(2, slots, value_states[:, :, n - keep:])
            self.cumulative_length.add_(n)
            if n > 1:                                              # prefill: attention needs the whole prompt
                return key_states, value_states
            return self.keys, self.values

    return RingLayer


class Reasoner:
    """Rationale and letter readout for one decision at a time.

    The prompt is prefilled into a static KV cache with the eager forward, then tokens are decoded greedily one at a
    time by replaying one captured CUDA graph of the decode step (torch.cuda.CUDAGraph around the eager forward with
    static input buffers: no Python, no kernel launches, no tracing framework between steps). The attention masks are
    built from tensors and handed to the model as its per-layer-type mapping (the same entry point the bidirectional
    readout uses), so the captured step depends on nothing but its input buffers: full-attention layers see the slots
    up to the current position of the static buffer, sliding layers the ring of the last `window` positions
    (RingLayer). Decoding stops at "Answer:", at an end-of-turn token, at max_new_tokens or at max_seconds; when the
    text does not end on the marker, the marker is appended and the logits after it are read. With a PEFT model and
    adapter_off the adapter is switched off for the pass (the base model reasons); a merged model reasons with the
    adapter folded in. If the capture fails, warm() falls back to the eager forward (same answers, slower).
    """

    def __init__(self, model, tokenizer, max_cache_len: int = 8192, graphs: bool = True, adapter_off: bool = True):
        import torch
        from transformers import StaticCache
        self.model, self.tokenizer = model, tokenizer
        self.net = model.get_base_model() if hasattr(model, "get_base_model") else model   # the transformers module
        self.adapter_off = adapter_off and hasattr(model, "disable_adapter")
        cfg = getattr(self.net.config, "text_config", None) or self.net.config
        self.cache = StaticCache(config=cfg, max_cache_len=max_cache_len)
        ring = _ring_layer_class()
        for i, layer in enumerate(self.cache.layers):
            if getattr(layer, "is_sliding", False):
                self.cache.layers[i] = ring(max_cache_len=max_cache_len, sliding_window=cfg.sliding_window)
        self.max_cache_len = max_cache_len
        self.window = min(cfg.sliding_window, max_cache_len)
        self.full_slots = torch.arange(max_cache_len, device="cuda")
        self.ring_slots = torch.arange(self.window, device="cuda")
        self.variants = letter_variants(tokenizer)
        self.marker_ids = tokenizer.encode("\n" + STOP, add_special_tokens=False)
        eos = getattr(getattr(self.net, "generation_config", None), "eos_token_id", None)
        self.eos = {tokenizer.eos_token_id} | set(eos if isinstance(eos, list) else [eos])
        self.eos.discard(None)
        self.graphs = graphs                 # capture the decode step in warm(); False after a failed capture
        self.graph, self.static = None, None
        self.step_seconds = 0.08             # per decoded token, measured on every pass (A100-like until then)
        self.prefill_rate = 2000.0           # prompt tokens per second, measured on prompts of 256 tokens or more

    def max_prompt_tokens(self, max_new_tokens: int) -> int:
        """The longest prompt that leaves room in the cache for the rationale and the marker."""
        return self.max_cache_len - max_new_tokens - len(self.marker_ids) - 1

    def _reserve(self) -> float:
        """Time kept for the appended marker and the readout after the decoding loop."""
        return (len(self.marker_ids) + 1) * self.step_seconds + 0.05

    def estimate(self, prompt_tokens: int, new_tokens: int) -> float:
        """Expected seconds of a pass with this prompt and this many decoded tokens, from the measured rates."""
        return prompt_tokens / self.prefill_rate + new_tokens * self.step_seconds + self._reserve()

    def prompt_tokens(self, state, instructions: str, opts: list[dict], words: int, sentences: int | None = None) -> list[int]:
        from .torch_backend import prompt_tokens
        return prompt_tokens(self.tokenizer, reason_messages(state, instructions, opts, words, sentences))

    @contextlib.contextmanager
    def _context(self):
        """The adapter switched off for the pass. PEFT re-enables it through set_adapter(), which also marks the
        adapter weights trainable; the one-pass readout then returns tensors that require grad, so that is undone."""
        if not self.adapter_off:
            yield
            return
        with self.model.disable_adapter():
            yield
        for p in self.model.parameters():
            if p.requires_grad:
                p.requires_grad_(False)

    def _prefill_masks(self, n: int) -> dict:
        """Boolean masks (True: attend) of the n prompt tokens over what each layer type returns from its cache: the
        whole static buffer for full attention, the prompt itself for sliding attention."""
        import torch
        q = torch.arange(n, device="cuda")[:, None]
        kv = torch.arange(n, device="cuda")[None, :]
        return {"full_attention": (self.full_slots[None, :] <= q)[None, None],
                "sliding_attention": ((kv <= q) & (kv > q - self.window))[None, None]}

    def _decode_masks(self, position) -> dict:
        """Masks of one token at `position` (a [1, 1] tensor): static slots up to the position; every ring slot once
        the ring has wrapped, else the slots written so far."""
        p = position[0, 0]
        return {"full_attention": (self.full_slots <= p)[None, None, None, :],
                "sliding_attention": ((self.ring_slots <= p) | (p >= self.window - 1))[None, None, None, :]}

    def _forward(self, ids, pos, masks: dict):
        """Logits of the last position for the tokens `ids` at positions `pos` through the model and its cache."""
        return self.net(input_ids=ids, position_ids=pos, attention_mask=masks, past_key_values=self.cache, use_cache=True,
                        logits_to_keep=1).logits

    def _prefill(self, tokens: list[int]):
        import torch
        self.cache.reset()
        ids = torch.tensor([tokens], device="cuda")
        pos = torch.arange(len(tokens), device="cuda")[None]
        return self._forward(ids, pos, self._prefill_masks(len(tokens)))[0, -1]

    def _step(self, token: int, position: int):
        """Next-token logits after feeding one token at the given position: a graph replay, or the eager forward."""
        import torch
        if self.graph is None:
            pos = torch.tensor([[position]], device="cuda")
            return self._forward(torch.tensor([[token]], device="cuda"), pos, self._decode_masks(pos))[0, -1]
        self.static["ids"].fill_(token)
        self.static["pos"].fill_(position)
        self.graph.replay()
        return self.static["logits"][0, -1]

    def _capture(self) -> None:
        """One decode step as a CUDA graph over static input buffers (the masks are computed inside the graph from
        the position buffer). Needs an initialised cache, so a prefill must have run; the warm-up replays required
        before capture run on a side stream, as the CUDA graphs documentation prescribes."""
        import torch
        static = {"ids": torch.zeros((1, 1), dtype=torch.long, device="cuda"),
                  "pos": torch.zeros((1, 1), dtype=torch.long, device="cuda")}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._forward(static["ids"], static["pos"], self._decode_masks(static["pos"]))
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static["logits"] = self._forward(static["ids"], static["pos"], self._decode_masks(static["pos"]))
        torch.cuda.synchronize()
        self.graph, self.static = graph, static

    def __call__(self, tokens: list[int], n_options: int, max_new_tokens: int, max_seconds: float = float("inf")) -> Reasoned:
        """The second pass of one decision. The rationale is cut at max_new_tokens or when the pass as a whole
        (prefill, decoding, marker and readout) would exceed max_seconds, whichever comes first."""
        import torch
        if len(tokens) > self.max_prompt_tokens(max_new_tokens):
            raise ValueError(f"prompt of {len(tokens)} tokens does not fit the cache of {self.max_cache_len}")
        t0 = time.perf_counter()
        stop_at = t0 + max_seconds - self._reserve()
        with torch.inference_mode(), self._context():
            logits = self._prefill(tokens)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            position, generated, text, steps = len(tokens), [], "", 0
            while len(generated) < max_new_tokens and time.perf_counter() < stop_at:
                token = int(logits.argmax())
                if token in self.eos:
                    break
                generated.append(token)
                logits = self._step(token, position)
                position, steps = position + 1, steps + 1
                text = self.tokenizer.decode(generated)
                if STOP in text:
                    break
            rationale, reached = split_reasoning(text)
            appended = 0
            if not (reached and text.endswith(STOP)):          # no marker, or a marker inside a longer last token
                for token in self.marker_ids:
                    logits = self._step(token, position)
                    position, steps = position + 1, steps + 1
                appended = len(self.marker_ids)
            z = logits.float()
            letters = [torch.logsumexp(z[v], 0).item() for v in self.variants[:n_options]]
        t2 = time.perf_counter()
        step_seconds = (t2 - t1) / max(1, steps)
        if steps >= 4:                                         # keep the rates current for the time estimates
            self.step_seconds = 0.5 * self.step_seconds + 0.5 * step_seconds
        if len(tokens) >= 256:
            self.prefill_rate = 0.5 * self.prefill_rate + 0.5 * len(tokens) / max(1e-3, t1 - t0)
        return Reasoned(letters, rationale, reached, len(generated), appended, t2 - t0, t1 - t0, step_seconds)

    def warm(self, steps: int = 8) -> float:
        """A dummy prefill (initialises the cache), the graph capture and one short pass, so the first real decision
        is already fast. Returns the seconds taken. A failed capture leaves the eager step in place."""
        import torch
        t0 = time.perf_counter()
        opts = [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}]
        tokens = self.prompt_tokens("Warm-up state: the invoice is from March.", "Is the invoice from March?", opts, 20)
        if self.graphs:
            try:
                with torch.inference_mode(), self._context():
                    self._prefill(tokens)
                    self._capture()
            except Exception as e:
                print(f"CUDA graph capture of the decode step failed ({type(e).__name__}: {str(e)[:300]}); "
                      f"decoding eagerly", flush=True)
                self.graph, self.static, self.graphs = None, None, False
        self(tokens, len(opts), steps)
        return time.perf_counter() - t0
