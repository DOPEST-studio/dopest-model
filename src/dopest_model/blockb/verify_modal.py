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
from __future__ import annotations

import json
import os
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


@app.cls(image=image, gpu="H100", **CONTAINER)
class Qwen:
    @modal.enter()
    def load(self):
        from vllm import LLM
        t0 = time.perf_counter()
        # text only (no vision profiling); the hybrid model needs a state block per running sequence, so cap the
        # batch; eager mode skips torch.compile and CUDA graphs, which a one-token prefill readout does not use
        self.llm = LLM(model=QWEN, max_model_len=16384, gpu_memory_utilization=0.92, max_num_seqs=256,
                       enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
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


VERIFIERS = {"qwen": (Qwen, "H100"), "gptoss": (GptOss, "H200")}


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

