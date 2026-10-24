"""DOPEST decision server: TypeSafe-compatible `POST /v1/systemone`, the wire format JevBench's harness speaks.

    uv run python -m dopest_model.serve --model google/gemma-4-31B-it --adapter <dir or HF repo> \\
        --calibration calibration.json --bidir full --port 8000

Request:  {"state": <text or JSON>, "model": "...", "questions": {"<name>": {"type": "choice"|"noul"|"score",
          "instructions": "...", "criteria": {...} | [...], "option_keys": "show"|"hide"}}}
Response: {"model": "...", "answers": {"<name>": {"type": "choice", "choice": "<key>", "probabilities": {...},
          "confidence": p} | {"type": "noul", "noul": p_yes, "probabilities": {"true": p, "false": 1-p},
          "confidence": p} | {"type": "score", "score": <expected level>, "probabilities": {"0": p, ...},
          "confidence": p}}, "usage": {"input_tokens": n, "output_tokens": 0, "questions": k, "latency_ms": t}}

One forward pass per question: the prompt ends at the answer position and the logits of the option letters are
read (bidirectional attention inside the prompt on the global layers when --bidir full, as the adapter was
trained). Choice questions with more than 16 options are read in groups of 16: the two most likely options of every
group go to a final round. Calibration (temperatures per type, yes/no commit floor) comes from the JSON file and is
applied exactly as in evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np

from .calibrate import commit_floor, softmax
from .prompt import LETTERS, messages_for, options

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
                 max_prompt_tokens: int = 12288, orders: int = 1):
        from .torch_backend import letter_logits, letter_token_ids, load, prompt_tokens
        self.model, self.tokenizer = load(model_id, adapter=adapter, bidir=bidir)
        self.letters = letter_token_ids(self.tokenizer)
        self.temps = calibration["temperatures"]
        self.floor = calibration.get("noul_floor")
        self.max_prompt_tokens = max_prompt_tokens
        self.orders = orders                                  # 2: choice questions are read in both option orders
        self._logits, self._tokens = letter_logits, prompt_tokens
        self.lock = threading.Lock()
        self.last_input_tokens = 0

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
            z = self._logits(self.model, tokens, self.letters[:len(opts)]).cpu().numpy()
            return [o["id"] for o in opts], softmax(z, self.temps[qtype]).tolist()
        return reader

    def decide(self, state, questions: dict) -> tuple[dict, int]:
        answers, used = {}, 0
        with self.lock:
            for name, q in questions.items():
                qtype = q.get("type")
                if qtype not in ("choice", "noul", "score"):
                    raise ValueError(f"question {name!r}: unsupported type {qtype!r}")
                if qtype != "noul" and not q.get("criteria"):
                    raise ValueError(f"question {name!r}: criteria required")
                opts = options(q, show_keys=q.get("option_keys", "show") != "hide")
                self.last_input_tokens = 0
                reader = self.read(state, q.get("instructions", ""), qtype)
                ids, probs = read_in_groups(opts, reader)
                if qtype == "choice" and self.orders == 2 and len(opts) > 1:
                    ids, probs = average_orders((ids, probs), read_in_groups(opts[::-1], reader))
                answers[name] = answer_from(qtype, ids, probs, self.floor, q.get("criteria"))
                used += self.last_input_tokens
        return answers, used


def create_app(decider: Decider, model_name: str):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    app = FastAPI(title="DOPEST decision model")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name}

    @app.post("/v1/systemone")
    async def systemone(body: dict):
        t0 = time.perf_counter()
        questions = body.get("questions")
        if "state" not in body or not isinstance(questions, dict) or not questions:
            return JSONResponse({"error": "body needs 'state' and a non-empty 'questions' object"}, status_code=400)
        try:
            answers, used = decider.decide(body["state"], questions)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"model": model_name, "answers": answers,
                "usage": {"input_tokens": used, "output_tokens": 0, "questions": len(questions),
                          "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}}
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="google/gemma-4-31B-it")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory or Hugging Face repo")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--bidir", default="full", help="'full' (as trained) or '' for causal attention")
    parser.add_argument("--orders", type=int, default=2, choices=(1, 2),
                        help="read choice questions in one option order, or in both and average (default)")
    parser.add_argument("--name", default="dopest-1.0-31b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn
    calibration = json.load(open(args.calibration, encoding="utf-8"))
    decider = Decider(args.model, args.adapter, calibration, bidir=args.bidir or None, orders=args.orders)
    decider.decide("warm-up", {"w": {"type": "noul", "instructions": "Is this a warm-up?"}})
    uvicorn.run(create_app(decider, args.name), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
