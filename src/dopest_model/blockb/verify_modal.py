"""Verify block B items with two open-weight models on Modal (vLLM 0.32), blind to the teacher's answer:

  Qwen3.8-27B-FP8 (1x H100): the decision prompt with thinking off; one generated token with top-20 logprobs gives
      the distribution over the option letters (same prompt and readout as the decision model itself).
  gpt-oss-120b (1x H200, MXFP4): pass 1 reasons in the analysis channel (medium effort, capped at REASONING_BUDGET);
      pass 2 opens the final channel and reads the letter distribution of the next token, as for Qwen.

    uv run modal run src/dopest_model/blockb/verify_modal.py --inp data/blockb/batch1_raw.jsonl \
        --out 'data/blockb/batch1_votes_{name}.jsonl' --only qwen [--chunk 1000] [--limit N]

Each verifier is one container (max_containers=1) that loads its model once and takes the items in chunks; votes
are appended to --out after every chunk, and a rerun skips items that already have a vote. Run the two verifiers
as two parallel processes (--only qwen / --only gptoss) and cap each with run-bounded: a chunk has its own timeout,
the whole run does not.

The image is CUDA 13.0 devel (torch 2.13 is built for CUDA 13.0), so kernels that vLLM compiles at start-up find
nvcc. DeepGEMM stays off: FP8 block GEMMs run on precompiled CUTLASS kernels, with no JIT warm-up on paid GPU time.
Prompts are rendered here with the models' own chat templates (checked locally), not by vLLM's chat front end.
Weights are cached on the Volume 'dopest-hf-cache' (gpt-oss: only the root MXFP4 checkpoint, not original/ or
metal/). Open weights run on our own containers, so no API provider's terms apply to the outputs.
"""
import json
import os
import random
import time
from pathlib import Path

import modal

app = modal.App("dopest-blockb-verify")
hf_cache = modal.Volume.from_name("dopest-hf-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.32.0", "huggingface_hub")
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_USE_DEEP_GEMM": "0", "VLLM_MOE_USE_DEEP_GEMM": "0"})
    .add_local_python_source("dopest_model")
)
QWEN = "Qwen/Qwen3.8-27B-FP8"
GPTOSS = "openai/gpt-oss-120b"
USD_PER_HOUR = {"H100": 3.95, "H200": 4.54}
GPTOSS_EFFORT = "medium"                      # the chat template's default effort
REASONING_BUDGET = 3072
CONTAINER = dict(volumes={"/cache": hf_cache}, max_containers=1, scaledown_window=10, retries=1,
                 timeout=15 * 60, startup_timeout=20 * 60)


def token_prompts(tok, convs, **template_kwargs) -> list[list[int]]:
    return [tok.encode(tok.apply_chat_template(c, add_generation_prompt=True, tokenize=False, **template_kwargs),
                       add_special_tokens=False) for c in convs]


def find_last(seq: list[int], sub: list[int]) -> int:
    """Start of the last occurrence of sub in seq, or -1."""
    for i in range(len(seq) - len(sub), -1, -1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


def last_message(ids: list[int], tok) -> tuple[str | None, str]:
    """(channel, text) of the last harmony message that is not the analysis channel: gpt-oss answers in `final`, but
    short structured answers sometimes come in `commentary`. (None, "") when there is no such message."""
    channel_id, message_id = tok.convert_tokens_to_ids("<|channel|>"), tok.convert_tokens_to_ids("<|message|>")
    stops = {tok.convert_tokens_to_ids(t) for t in ("<|end|>", "<|return|>", "<|call|>")}
    found = (None, "")
    i = 0
    while i < len(ids):
        if ids[i] == channel_id and message_id in ids[i + 1:i + 8]:
            j = ids.index(message_id, i + 1)
            name = tok.decode(ids[i + 1:j], skip_special_tokens=True).strip()
            k = j + 1
            while k < len(ids) and ids[k] not in stops:
                k += 1
            if name != "analysis":
                found = (name, tok.decode(ids[j + 1:k], skip_special_tokens=True))
            i = k
        i += 1
    return found


def generation_stats(rows: list[dict], load_s: float, t0: float) -> dict:
    return {"rows": rows, "container": os.environ.get("MODAL_TASK_ID", "?"), "load_s": load_s,
            "infer_s": time.perf_counter() - t0, "out_tokens": sum(r["tokens"] for r in rows)}


@app.cls(image=image, gpu="H100", **CONTAINER)
class Qwen:
    eager: bool = modal.parameter(default=True)     # False for generation jobs: CUDA graphs make decoding faster

    @modal.enter()
    def load(self):
        from vllm import LLM
        t0 = time.perf_counter()
        # text only (no vision profiling); the hybrid model needs a state block per running sequence, so cap the
        # batch; eager mode skips torch.compile and CUDA graphs, which a one-token prefill readout does not use
        self.llm = LLM(model=QWEN, max_model_len=16384, gpu_memory_utilization=0.92,
                       max_num_seqs=256 if self.eager else 96, enforce_eager=self.eager,
                       limit_mm_per_prompt={"image": 0, "video": 0})
        hf_cache.commit()
        self.tok = self.llm.get_tokenizer()
        self.load_s = time.perf_counter() - t0

    @modal.method()
    def votes(self, items: list[dict]) -> dict:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from dopest_model.blockb.readout import letter_distribution
        from dopest_model.prompt import messages
        t0 = time.perf_counter()
        convs, ids = zip(*(messages(it) for it in items))
        prompts = [TokensPrompt(prompt_token_ids=p) for p in token_prompts(self.tok, convs, enable_thinking=False)]
        outs = self.llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20), use_tqdm=False)
        votes = []
        for it, opt_ids, o in zip(items, ids, outs):
            top = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            dist = letter_distribution(top, len(opt_ids), self.tok.decode)
            votes.append({"id": it["id"], "probs": dict(zip(opt_ids, dist)) if dist else None})
        return {"votes": votes, "container": os.environ.get("MODAL_TASK_ID", "?"), "load_s": self.load_s,
                "infer_s": time.perf_counter() - t0}


    @modal.method()
    def letter_rows(self, items: list[dict]) -> list[dict]:
        """Zero-shot decision rows in the project's logits format: the log-probabilities of the option letters at the
        answer position (a letter outside the top-20 gets the lowest seen log-probability minus 1)."""
        import math
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from dopest_model.prompt import LETTERS, messages
        convs, ids = zip(*(messages(it) for it in items))
        toks = token_prompts(self.tok, convs, enable_thinking=False)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in toks],
                                 SamplingParams(max_tokens=1, temperature=0.0, logprobs=20), use_tqdm=False)
        rows = []
        for it, opt_ids, t, o in zip(items, ids, toks, outs):
            top = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            seen = {}
            for tid, lp in top.items():
                text = (lp.decoded_token if lp.decoded_token is not None else self.tok.decode([tid])).strip()
                if len(text) == 1 and text in LETTERS[:len(opt_ids)]:
                    seen[text] = max(seen.get(text, -math.inf), lp.logprob)
            floor = (min(lp.logprob for lp in top.values()) if top else 0.0) - 1.0
            rows.append({"id": it["id"], "type": it["question"]["type"], "ids": list(opt_ids),
                         "logits": [round(seen.get(L, floor), 5) for L in LETTERS[:len(opt_ids)]],
                         "n_tokens": len(t), "seconds": 0.0})
        return rows

    @modal.method()
    def generate(self, convs: list[list[dict]], max_tokens: int = 6000, think_budget: int = 0) -> dict:
        """Full generations for the replica pipeline. think_budget 0: thinking off, the answer is generated directly.
        Otherwise two passes: the thinking block up to the budget (cut there when longer), then the answer."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        t0 = time.perf_counter()
        sampling = dict(temperature=0.7, top_p=0.95, top_k=20)
        if not think_budget:
            prompts = token_prompts(self.tok, convs, enable_thinking=False)
            outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                     SamplingParams(max_tokens=max_tokens, **sampling), use_tqdm=False)
            rows = [{"text": o.outputs[0].text, "tokens": len(o.outputs[0].token_ids),
                     "finish": o.outputs[0].finish_reason, "thought": False} for o in outs]
            return generation_stats(rows, self.load_s, t0)
        end = self.tok.convert_tokens_to_ids("</think>")
        prompts = token_prompts(self.tok, convs, enable_thinking=True)
        first = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                  SamplingParams(max_tokens=think_budget, stop_token_ids=[end], **sampling),
                                  use_tqdm=False)
        thoughts = [[t for t in o.outputs[0].token_ids if t != end] for o in first]
        close = self.tok.encode("</think>\n\n", add_special_tokens=False)
        second = self.llm.generate([TokensPrompt(prompt_token_ids=p + th + close) for p, th in zip(prompts, thoughts)],
                                   SamplingParams(max_tokens=max_tokens, **sampling), use_tqdm=False)
        rows = [{"text": o.outputs[0].text, "tokens": len(th) + len(o.outputs[0].token_ids),
                 "finish": o.outputs[0].finish_reason, "thought": True,
                 "think_cut": f.outputs[0].finish_reason == "length"}
                for th, f, o in zip(thoughts, first, second)]
        return generation_stats(rows, self.load_s, t0)


@app.cls(image=image, gpu="H200", **CONTAINER)
class GptOss:
    @modal.enter()
    def load(self):
        from huggingface_hub import snapshot_download
        from vllm import LLM
        t0 = time.perf_counter()
        path = snapshot_download(GPTOSS, allow_patterns=["*.json", "*.jinja", "*.txt", "model-*.safetensors"])
        hf_cache.commit()
        self.llm = LLM(model=path, max_model_len=16384, gpu_memory_utilization=0.9)
        self.tok = self.llm.get_tokenizer()
        self.end_id = self.tok.convert_tokens_to_ids("<|end|>")
        self.to_final = self.tok.encode("<|end|><|start|>assistant<|channel|>final<|message|>",
                                        add_special_tokens=False)
        self.load_s = time.perf_counter() - t0

    @modal.method()
    def votes(self, items: list[dict]) -> dict:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        from dopest_model.blockb.readout import final_continuation, letter_distribution
        from dopest_model.prompt import messages
        t0 = time.perf_counter()
        convs, ids = zip(*(messages(it) for it in items))
        prompts = token_prompts(self.tok, convs, reasoning_effort=GPTOSS_EFFORT)
        first = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                  SamplingParams(max_tokens=REASONING_BUDGET, temperature=1.0,
                                                 stop_token_ids=[self.end_id]), use_tqdm=False)
        gens = [list(o.outputs[0].token_ids) for o in first]
        tail = self.to_final[3:]                     # <|channel|>final<|message|>
        second = self.llm.generate(
            [TokensPrompt(prompt_token_ids=p + final_continuation(g, self.end_id, self.to_final, tail))
             for p, g in zip(prompts, gens)],
            SamplingParams(max_tokens=1, temperature=0.0, logprobs=20), use_tqdm=False)
        votes = []
        for it, opt_ids, g, o1, o2 in zip(items, ids, gens, first, second):
            top = o2.outputs[0].logprobs[0] if o2.outputs[0].logprobs else {}
            dist = letter_distribution(top, len(opt_ids), self.tok.decode)
            votes.append({"id": it["id"], "probs": dict(zip(opt_ids, dist)) if dist else None,
                          "reasoning_tokens": len(g), "capped": o1.outputs[0].finish_reason == "length",
                          "reasoning": self.tok.decode(g, skip_special_tokens=True)[:800]})
        return {"votes": votes, "container": os.environ.get("MODAL_TASK_ID", "?"), "load_s": self.load_s,
                "infer_s": time.perf_counter() - t0, "out_tokens": sum(map(len, gens))}

    @modal.method()
    def generate(self, convs: list[list[dict]], max_tokens: int = 6000, think_budget: int = 0) -> dict:
        """Full generations (analysis, then the final channel): the final-channel text per conversation. The reasoning
        effort comes from the chat template; think_budget is accepted for a uniform call and ignored."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        t0 = time.perf_counter()
        prompts = token_prompts(self.tok, convs, reasoning_effort=GPTOSS_EFFORT)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts],
                                 SamplingParams(max_tokens=max_tokens, temperature=1.0), use_tqdm=False)
        rows = []
        for o in outs:
            ids = list(o.outputs[0].token_ids)
            channel, text = last_message(ids, self.tok)
            rows.append({"text": text, "tokens": len(ids), "finish": o.outputs[0].finish_reason,
                         "thought": channel is not None, "channel": channel,
                         **({} if text else {"raw": self.tok.decode(ids[-400:], skip_special_tokens=False)[-600:]})})
        return generation_stats(rows, self.load_s, t0)


VERIFIERS = {"qwen": (Qwen, "H100"), "gptoss": (GptOss, "H200")}


@app.local_entrypoint()
def replica(stage: str, inp: str, out: str, engine: str, blind: str = "", chunk: int = 48, limit: int = 0,
            max_tokens: int = 6000, think_budget: int = 0):
    """One stage of the replica pipeline on one engine. Rows are appended to --out; a rerun skips the rows that
    succeeded and retries the failed ones (downstream readers take the last row of an id). --think-budget N (Qwen):
    at most N thinking tokens before the answer, 0 = thinking off.

        author:  --inp data/replica/briefs.jsonl (this engine's briefs)        -> {id (brief), ok, why, item}
        blind:   --inp data/replica/items.jsonl (the other author's items)     -> {id, answer, probs, note, ok}
        verdict: --inp data/replica/items.jsonl --blind data/replica/blind_<engine>.jsonl -> {id, verdict, reason, fix}
    """
    from dopest_model.replica import stages
    cls, gpu = VERIFIERS[engine]
    rows_in = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
    if stage == "author":
        rows_in = [b for b in rows_in if b["author"] == engine]
        convs_fn, rows_fn = stages.author_convs, lambda rows, outs: stages.author_rows(rows, outs, engine)
    elif stage in ("blind", "verdict"):
        rows_in = [it for it in rows_in if it["provenance"]["author_model"] != engine]   # the other model reviews
        if stage == "blind":
            convs_fn, rows_fn = stages.blind_convs, lambda rows, outs: stages.blind_rows(rows, outs, engine)
        else:
            blind_by_id = ({r["id"]: r for r in map(json.loads, open(blind, encoding="utf-8"))} if blind
                           else {})                       # no blind pass: the verdict prompt omits the blind answer
            convs_fn = lambda rows: stages.verdict_convs(rows, blind_by_id)                       # noqa: E731
            rows_fn = lambda rows, outs: stages.verdict_rows(rows, outs, engine)                 # noqa: E731
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    path = Path(out)
    done = ({r["id"] for r in map(json.loads, open(path, encoding="utf-8")) if r.get("ok")}
            if path.exists() else set())
    todo = [r for r in rows_in if r["id"] not in done]
    random.Random(0).shuffle(todo)                    # every chunk mixes cells and state lengths
    todo = todo[:limit] if limit else todo
    chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    print(f"{stage}/{engine}: {len(done)} rows done, {len(todo)} to go in {len(chunks)} chunks", flush=True)
    if not chunks:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = cls(eager=False) if engine == "qwen" else cls()
    loads, infer_s, out_tokens, ok = {}, 0.0, 0, 0
    with open(path, "a", encoding="utf-8") as f:
        results = obj.generate.map([convs_fn(c) for c in chunks],
                                   kwargs={"max_tokens": max_tokens, "think_budget": think_budget},
                                   return_exceptions=True)
        for k, (rows, r) in enumerate(zip(chunks, results), 1):
            if isinstance(r, Exception):                  # a failed chunk stays undone; a rerun retries it
                print(f"chunk {k} FAILED {type(r).__name__}: {str(r)[:300]}", flush=True)
                continue
            parsed = rows_fn(rows, r["rows"])
            for row in parsed:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            ok += sum(bool(row.get("ok")) for row in parsed)
            loads[r["container"]] = r["load_s"]
            infer_s += r["infer_s"]
            out_tokens += r["out_tokens"]
            usd = (sum(loads.values()) + infer_s) / 3600 * USD_PER_HOUR[gpu]
            print(f"chunk {k}/{len(chunks)}: {r['infer_s']:.0f} s, {r['out_tokens']:,} tokens; ok so far {ok}/"
                  f"{k * chunk if k < len(chunks) else len(todo)}; load {sum(loads.values()) / 60:.1f} min; "
                  f"GPU ~${usd:.2f}", flush=True)


@app.local_entrypoint()
def main(inp: str, out: str, only: str = "qwen,gptoss", chunk: int = 1000, limit: int = 0):
    items = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
    items = items[:limit] if limit else items
    names = only.split(",")
    if len(names) > 1 and "{name}" not in out:
        raise SystemExit("several verifiers need {name} in --out")
    for name in names:
        cls, gpu = VERIFIERS[name]
        path = Path(out.format(name=name))
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if path.exists() else set()
        todo = [it for it in items if it["id"] not in done]
        chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
        print(f"{name}: {len(done)} items already voted, {len(todo)} to go in {len(chunks)} chunks", flush=True)
        loads, infer_s, usable = {}, 0.0, 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for k, r in enumerate(cls().votes.map(chunks, return_exceptions=True), 1):
                if isinstance(r, Exception):                 # a failed chunk stays unvoted; a rerun retries it
                    print(f"{name}: chunk {k} FAILED {type(r).__name__}: {str(r)[:300]}", flush=True)
                    continue
                for v in r["votes"]:
                    f.write(json.dumps({"id": v.pop("id"), name: v}, ensure_ascii=False) + "\n")
                f.flush()
                loads[r["container"]] = r["load_s"]
                infer_s += r["infer_s"]
                usable += sum(v["probs"] is not None for v in r["votes"])
                usd = (sum(loads.values()) + infer_s) / 3600 * USD_PER_HOUR[gpu]
                print(f"{name}: chunk {k}/{len(chunks)} in {r['infer_s']:.0f} s"
                      + (f" ({r['out_tokens']:,} reasoning tokens)" if "out_tokens" in r else "")
                      + f"; usable so far {usable}; load {sum(loads.values()) / 60:.1f} min; GPU ~${usd:.2f}",
                      flush=True)


@app.local_entrypoint()
def zeroshot(out: str, data: str = "data/raw/opendecisions/opendecisions.jsonl,data/dev/jevbench_public.jsonl",
             chunk: int = 1000):
    """Qwen3.8-27B zero-shot letter logits on the given data files -> <out>/<file stem>.jsonl (evaluate.py format)."""
    for path in data.split(","):
        items = list({it["id"]: it for it in map(json.loads, open(path, encoding="utf-8"))}.values())
        chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
        target = Path(out) / (Path(path).stem + ".jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        t0, n = time.perf_counter(), 0
        with open(target, "w", encoding="utf-8") as f:
            for rows in Qwen().letter_rows.map(chunks):
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                n += len(rows)
                print(f"{target.name}: {n}/{len(items)} rows, {time.perf_counter() - t0:.0f} s", flush=True)



@app.local_entrypoint()
def review(inp: str, out_dir: str, engine: str, chunk: int = 48, limit: int = 0, max_tokens: int = 2000,
           think_budget: int = 0):
    """Blind pass and verdict pass on the other author's items in one container session (one model load):
    <out_dir>/blind_<engine>.jsonl and <out_dir>/verdict_<engine>.jsonl. A rerun skips items that have a verdict."""
    from dopest_model.replica import stages
    cls, gpu = VERIFIERS[engine]
    items = [it for it in map(json.loads, open(inp, encoding="utf-8")) if it["provenance"]["author_model"] != engine]
    out_b, out_v = Path(out_dir) / f"blind_{engine}.jsonl", Path(out_dir) / f"verdict_{engine}.jsonl"
    done = ({r["id"] for r in map(json.loads, open(out_v, encoding="utf-8")) if r.get("ok")}
            if out_v.exists() else set())
    todo = [it for it in items if it["id"] not in done]
    random.Random(0).shuffle(todo)
    todo = todo[:limit] if limit else todo
    chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    print(f"review/{engine}: {len(done)} done, {len(todo)} to go in {len(chunks)} chunks", flush=True)
    if not chunks:
        return
    out_b.parent.mkdir(parents=True, exist_ok=True)
    kept = cls.with_options(scaledown_window=300)            # the container must survive the gap between two calls
    obj = kept(eager=False) if engine == "qwen" else kept()
    loads, infer_s, ok_b, ok_v = {}, 0.0, 0, 0
    with open(out_b, "a", encoding="utf-8") as fb, open(out_v, "a", encoding="utf-8") as fv:
        for k, rows in enumerate(chunks, 1):
            try:
                rb = obj.generate.remote(stages.blind_convs(rows), max_tokens=max_tokens, think_budget=think_budget)
                blind = stages.blind_rows(rows, rb["rows"], engine)
                rv = obj.generate.remote(stages.verdict_convs(rows, {b["id"]: b for b in blind}),
                                         max_tokens=max_tokens, think_budget=think_budget)
                verdict = stages.verdict_rows(rows, rv["rows"], engine)
            except Exception as e:                        # the chunk stays undone; a rerun retries it
                print(f"chunk {k} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            for row in blind:
                fb.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in verdict:
                fv.write(json.dumps(row, ensure_ascii=False) + "\n")
            fb.flush()
            fv.flush()
            ok_b += sum(r["ok"] for r in blind)
            ok_v += sum(r["ok"] for r in verdict)
            for r in (rb, rv):
                loads[r["container"]] = r["load_s"]
                infer_s += r["infer_s"]
            usd = (sum(loads.values()) + infer_s) / 3600 * USD_PER_HOUR[gpu]
            print(f"chunk {k}/{len(chunks)}: blind {rb['infer_s']:.0f} s + verdict {rv['infer_s']:.0f} s, "
                  f"{rb['out_tokens'] + rv['out_tokens']:,} tokens; ok blind {ok_b}, verdict {ok_v}; "
                  f"load {sum(loads.values()) / 60:.1f} min; GPU ~${usd:.2f}", flush=True)
