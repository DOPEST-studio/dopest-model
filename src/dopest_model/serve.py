"""DOPEST decision server: TypeSafe-compatible `POST /v1/systemone`, the wire format JevBench's harness speaks.

    uv run python -m dopest_model.serve --model google/gemma-4-31B-it --adapter <dir or HF repo> \\
        --calibration calibration.json --bidir full --port 8000

Request:  {"state": <text or JSON>, "model": "...", "questions": {"<name>": {"type": "choice"|"noul"|"score",
          "instructions": "...", "criteria": {...} | [...], "option_keys": "show"|"hide"}}}
Response: {"model": "...", "answers": {"<name>": {"type": "choice", "choice": "<key>", "probabilities": {...},
          "confidence": p} | {"type": "noul", "noul": p_yes, "probabilities": {"true": p, "false": 1-p},
          "confidence": p} | {"type": "score", "score": <expected level>, "probabilities": {"0": p, ...},
          "confidence": p}}, "usage": {"input_tokens": n, "output_tokens": m, "questions": k, "latency_ms": t}}

One forward pass per question: the prompt ends at the answer position and the logits of the option letters are
read (bidirectional attention inside the prompt on the global layers when --bidir full, as the adapter was
trained). Choice questions with more than 16 options are read in groups of 16: the two most likely options of every
group go to a final round. Calibration (temperatures per type, yes/no commit floor) comes from the JSON file and is
applied exactly as in evaluate.py.

With --reason, a choice question whose top probability is below the gate gets a second pass (reason.py): the base
model writes a short rationale and the letters are read after it; the two distributions are mixed. The second pass
reads the state again and generates tokens, so usage counts both (output_tokens is non-zero only then), a cost budget
keeps the pooled extra cost of a run bounded, and a time budget per request keeps every decision well inside the
board's 10-second limit (the board sends one question per request; a request with several questions shares the one
budget, so later questions may get no second pass).
"""
from __future__ import annotations

import argparse
import json
import threading
import time

from .calibrate import commit_floor, softmax
from .prompt import LETTERS, messages_for, options
from .reason import Budget, Policy, blend, unsure

GROUP = len(LETTERS)
PER_GROUP = 2


def read_in_groups(opts: list[dict], read) -> tuple[list[str], list[float]]:
    """Distribution over all option ids using a reader that takes at most GROUP options at a time."""
    if len(opts) <= GROUP:
        return read(opts)
    finalists = []
    for i in range(0, len(opts), GROUP):
        group = opts[i:i + GROUP]
        ids, probs = read(group)
        keep = {k for k, _ in sorted(zip(ids, probs), key=lambda x: -x[1])[:PER_GROUP]}
        finalists += [o for o in group if o["id"] in keep]
    ids, probs = read_in_groups(finalists, read)         # more rounds when the finalists still exceed one group
    full = {o["id"]: 1e-6 for o in opts}
    full.update(dict(zip(ids, probs)))
    total = sum(full.values())
    keys = [o["id"] for o in opts]
    return keys, [full[k] / total for k in keys]


def average_orders(a: tuple[list[str], list[float]], b: tuple[list[str], list[float]]) -> tuple[list[str], list[float]]:
    """Mean of two distributions over the same option ids (read in different orders), in the order of the first."""
    pb = dict(zip(*b))
    return a[0], [(p + pb[k]) / 2 for k, p in zip(*a)]


def answer_from(qtype: str, ids: list[str], probs: list[float], floor: float | None, criteria) -> dict:
    """The answer object of one question from calibrated option probabilities (in option order)."""
    by_id = dict(zip(ids, probs))
    if qtype == "noul":
        p = by_id["true"]
        if floor:
            p = commit_floor(p, floor)
        return {"type": "noul", "noul": p, "probabilities": {"true": p, "false": 1.0 - p}, "confidence": max(p, 1 - p)}
    if qtype == "choice":
        best = max(ids, key=lambda k: by_id[k])
        return {"type": "choice", "choice": best, "probabilities": by_id, "confidence": by_id[best]}
    levels = list(range(len(ids)))
    expected = float(sum(l * by_id[str(l)] for l in levels))
    best = max(levels, key=lambda l: (by_id[str(l)], -l))
    legend = {str(i): text for i, text in enumerate(criteria)} if isinstance(criteria, list) else {}
    return {"type": "score", "score": expected, "probabilities": {str(l): by_id[str(l)] for l in levels},
            "legend": legend, "confidence": by_id[str(best)]}


class Decider:
    """The model, its calibration and the readout; thread-safe (one forward at a time)."""

    def __init__(self, model_id: str, adapter: str | None, calibration: dict, bidir: str | None = "full",
                 max_prompt_tokens: int = 12288, orders: int = 1, merge: bool = True, reasoning: Policy | None = None,
                 reason_with_adapter: bool = False, graphs: bool = True):
        from .torch_backend import letter_logits, letter_token_ids, load, prompt_tokens
        self.model, self.tokenizer = load(model_id, adapter=adapter, bidir=bidir)
        base_reasons = reasoning is not None and bool(adapter) and not reason_with_adapter
        if adapter and merge and not base_reasons:            # the adapter folded into the weights: 42% lower latency,
            from .fastserve import merge_adapter               # identical answers (measured on the public items);
            self.model = merge_adapter(self.model)            # kept apart when the base model must reason
        self.letters = letter_token_ids(self.tokenizer)
        self.temps = calibration["temperatures"]
        self.floor = calibration.get("noul_floor")
        self.max_prompt_tokens = max_prompt_tokens
        self.orders = orders                                  # 2: choice questions are read in both option orders
        self._logits, self._tokens = letter_logits, prompt_tokens
        self.lock = threading.Lock()
        self.last_input_tokens = self.last_output_tokens = 0
        self.policy, self.reasoner, self.budget, self.reasoned = reasoning, None, None, 0
        self.skipped = {"length": 0, "budget": 0, "time": 0, "error": 0}   # second passes not taken, by reason
        if reasoning:
            from .reason import Reasoner
            self.reasoner = Reasoner(self.model, self.tokenizer, graphs=graphs, adapter_off=not reason_with_adapter)
            self.budget = Budget(reasoning.budget_usd_per_1000 / 1000)
            seconds = self.reasoner.warm()
            print(f"second pass ready in {seconds:.0f} s ({'CUDA graph' if self.reasoner.graph else 'eager'} decode, "
                  f"{'base model' if self.reasoner.adapter_off else 'adapted model'} reasons)", flush=True)

    def read(self, state, instructions: str, qtype: str):
        def reader(opts: list[dict]) -> tuple[list[str], list[float]]:
            msgs = messages_for(state, instructions, opts)
            tokens = self._tokens(self.tokenizer, msgs)
            if len(tokens) > self.max_prompt_tokens:               # keep the start of a very long state
                text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
                keep = int(len(text) * self.max_prompt_tokens / len(tokens) * 0.95)
                msgs = messages_for(text[:keep] + " [...]", instructions, opts)
                tokens = self._tokens(self.tokenizer, msgs)
            self.last_input_tokens += len(tokens)
            import torch
            with torch.inference_mode():
                z = self._logits(self.model, tokens, self.letters[:len(opts)]).cpu().numpy()
            return [o["id"] for o in opts], softmax(z, self.temps[qtype]).tolist()
        return reader

    def reason(self, state, instructions: str, opts: list[dict], probs: list[float], started: float) -> list[float]:
        """The one-pass distribution mixed with a reasoned one, when the prompt fits, the budget allows and enough
        of the decision's time budget is left (`started`: perf_counter at the start of this decision)."""
        policy = self.policy
        tokens = self.reasoner.prompt_tokens(state, instructions, opts, policy.words, policy.sentences)
        limit = min(policy.max_prompt_tokens, self.reasoner.max_prompt_tokens(policy.max_new_tokens))
        remaining = policy.max_seconds - (time.perf_counter() - started)
        if len(tokens) > limit:
            self.skipped["length"] += 1
            return probs
        if not self.budget.allows(policy.cost(len(tokens) + len(self.reasoner.marker_ids), policy.max_new_tokens)):
            self.skipped["budget"] += 1
            return probs
        if self.reasoner.estimate(len(tokens), policy.min_new_tokens) > remaining:
            self.skipped["time"] += 1
            return probs
        try:
            r = self.reasoner(tokens, len(opts), policy.max_new_tokens, remaining)
        except Exception as e:                                 # the one-pass answer stands; the failure is logged
            self.skipped["error"] += 1
            print(f"second pass failed ({type(e).__name__}: {str(e)[:200]}); keeping the one-pass answer", flush=True)
            return probs
        self.budget.charge(policy.cost(len(tokens) + r.marker_tokens, r.new_tokens))
        self.last_input_tokens += len(tokens) + r.marker_tokens
        self.last_output_tokens += r.new_tokens
        self.reasoned += 1
        return blend(probs, r.logits, policy.weight, policy.temperature)

    def decide(self, state, questions: dict) -> tuple[dict, int, int]:
        """Answers, input tokens and output tokens of one request."""
        answers, used_in, used_out = {}, 0, 0
        with self.lock:
            started = time.perf_counter()                      # the time budget covers the whole request
            for name, q in questions.items():
                qtype = q.get("type")
                if qtype not in ("choice", "noul", "score"):
                    raise ValueError(f"question {name!r}: unsupported type {qtype!r}")
                if qtype != "noul" and not q.get("criteria"):
                    raise ValueError(f"question {name!r}: criteria required")
                opts = options(q, show_keys=q.get("option_keys", "show") != "hide")
                self.last_input_tokens = self.last_output_tokens = 0
                reader = self.read(state, q.get("instructions", ""), qtype)
                ids, probs = read_in_groups(opts, reader)
                if qtype == "choice" and self.orders == 2 and len(opts) > 1:
                    ids, probs = average_orders((ids, probs), read_in_groups(opts[::-1], reader))
                if qtype == "choice" and self.reasoner and len(opts) <= GROUP and unsure(probs, self.policy.gate):
                    probs = self.reason(state, q.get("instructions", ""), opts, probs, started)
                if self.budget:
                    self.budget.tick()
                answers[name] = answer_from(qtype, ids, probs, self.floor, q.get("criteria"))
                used_in += self.last_input_tokens
                used_out += self.last_output_tokens
        return answers, used_in, used_out


def create_app(decider: Decider, model_name: str):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    app = FastAPI(title="DOPEST decision model")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "reasoning": decider.policy is not None}

    @app.post("/v1/systemone")
    async def systemone(body: dict):
        t0 = time.perf_counter()
        questions = body.get("questions")
        if "state" not in body or not isinstance(questions, dict) or not questions:
            return JSONResponse({"error": "body needs 'state' and a non-empty 'questions' object"}, status_code=400)
        try:
            answers, used_in, used_out = decider.decide(body["state"], questions)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"model": model_name, "answers": answers,
                "usage": {"input_tokens": used_in, "output_tokens": used_out, "questions": len(questions),
                          "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}}
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory or Hugging Face repo")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--bidir", default="full", help="'full' (as trained) or '' for causal attention")
    parser.add_argument("--orders", type=int, default=1, choices=(1, 2),
                        help="read choice questions in one option order (default), or in both and average "
                             "(about 70%% more input tokens per decision: outside the JevBench Jev-class cost cap)")
    parser.add_argument("--no-merge", action="store_true", help="keep the adapter unmerged (slower, same answers)")
    parser.add_argument("--reason", action="store_true",
                        help="second pass with a short rationale for unsure choice questions of at most 16 options "
                             "(gate, mix, length and budgets from the calibration file's 'reasoning' section or "
                             "reason.Policy defaults); the adapter then stays unmerged, so the base model can write "
                             "the rationale (unless --reason-with-adapter)")
    parser.add_argument("--reason-budget", type=float, default=None,
                        help="extra cost allowed for the second passes, $ per 1000 decisions, pooled over the run")
    parser.add_argument("--reason-with-adapter", action="store_true",
                        help="let the adapted model write the rationale instead of the base model")
    parser.add_argument("--no-graphs", action="store_true", help="eager decode step for the second pass (slower)")
    parser.add_argument("--name", default="dopest-1.0-31b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn
    calibration = json.load(open(args.calibration, encoding="utf-8"))
    policy = Policy.from_calibration(calibration, budget_usd_per_1000=args.reason_budget) if args.reason else None
    decider = Decider(args.model, args.adapter, calibration, bidir=args.bidir or None, orders=args.orders,
                      merge=not args.no_merge, reasoning=policy, reason_with_adapter=args.reason_with_adapter,
                      graphs=not args.no_graphs)
    decider.decide("warm-up", {"w": {"type": "noul", "instructions": "Is this a warm-up?"}})
    uvicorn.run(create_app(decider, args.name), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
