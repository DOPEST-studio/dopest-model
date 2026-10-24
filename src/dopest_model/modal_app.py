"""DOPEST model on Modal (NVIDIA GPUs): the PyTorch backend with the same recipe as the MLX code.

Always run detached, so a dropped connection on the laptop does not stop the job:

    uv run modal run --detach src/dopest_model/modal_app.py::smoke                # ~5 min: torch vs MLX logits
    uv run modal run --detach src/dopest_model/modal_app.py::pilot --run pilot-12b-modal-v2

Every result is written to the Volume 'dopest-runs' as it is produced. If the local client lost the connection,
fetch the results without recomputing anything:

    uv run modal run src/dopest_model/modal_app.py::smoke --fetch
    uv run modal run src/dopest_model/modal_app.py::pilot --run pilot-12b-modal-v2 --fetch

Base weights come from the original Google repo (transformers format), cached on the Volume 'dopest-hf-cache'.
Logits land in runs/<run>/ locally, so evaluate.py and compare.py work on them unchanged.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
# Locally: <repo>/src/dopest_model/modal_app.py. In the container Modal mounts this file as /root/modal_app.py,
# where ROOT / "data" is exactly the mounted data directory.
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/root")
GPU = "H100"
USD_PER_HOUR = 3.95                     # Modal H100, 2026-10-23 (per-second billing; CPU and memory extra)
GPU_USD_PER_HOUR = {"H100": 3.95, "H200": 4.54}
BASE = "google/gemma-4-12B-it"
TRAIN_FILES = ["gen/water_claim.jsonl", "gen/temporal_numeric.jsonl", "gen/probability.jsonl",
               "gen/jaggedness.jsonl", "train/blockb_v1.jsonl",          # block A generators + block B (docs/008)
               "gen/temporal_v2.jsonl", "gen/judge_response.jsonl",      # phase 2 generators (docs/011)
               "gen/oncall_routing.jsonl", "gen/state_tracking.jsonl"]
# Most valuable first: if the time budget runs out, the later sets are skipped, never the run's cost cap.
EVAL_SETS = {"jevbench_public": "dev/jevbench_public.jsonl", "gen_test2": "dev/gen_test2_clean.jsonl",
             "opendecisions": "raw/opendecisions/opendecisions.jsonl", "blockb_test": "dev/blockb_test.jsonl",
             "gen_hard_test": "dev/gen_hard_test.jsonl", "gen_test": "dev/gen_test_clean.jsonl"}
# A + B: 15.9M tokens per epoch, ~38 min at the measured 6938 tok/s; evaluation of all sets ~8 min
PILOT_TIMEOUT_MIN = 95                  # hard cap: 95 min x ($3.95 GPU + ~$0.35 CPU/RAM) = ~$6.8 worst case
PILOT_TRAIN_MIN = 60                    # training stops here at the latest, leaving time to evaluate
PILOT_EVAL_DEADLINE_MIN = 80            # no new eval set starts after this
EVAL_PACK_TOKENS = 8192                 # forward-only scoring packs, whatever the training pack size
OUTPUTS = [f"{name}.jsonl" for name in EVAL_SETS] + ["train_log.jsonl", "train_meta.json", "timing.json"]

app = modal.App("dopest-model")
hf_cache = modal.Volume.from_name("dopest-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("dopest-runs", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("torch==2.14.1", "transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0", "safetensors",
                    "numpy", "fastapi>=0.115", "uvicorn>=0.30",
                    "jevbench @ git+https://github.com/fstandhartinger/jevbench")   # the board's conformance kit
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("dopest_model")
    .add_local_dir(ROOT / "data", "/root/data", ignore=["raw/basal/**", "blockb/**"])     # blockb/: work files
)
VOLUMES = {"/cache": hf_cache, "/runs": runs}


def read(rel: str) -> list[dict]:
    with open(Path("/root/data") / rel, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def volume_text(path: str) -> str | None:
    """A text file from the 'dopest-runs' Volume (local side), or None if it is not there."""
    try:
        return b"".join(runs.read_file(path)).decode("utf-8")
    except Exception:                    # missing file: the run did not get that far
        return None


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=30 * 60)
def smoke_remote(n: int) -> dict:
    from dopest_model import torch_backend as tb
    t0 = time.perf_counter()
    model, tokenizer = tb.load(BASE)
    hf_cache.commit()
    out = Path(f"/runs/smoke/first{n}.jsonl")
    tb.predict(model, tokenizer, read(EVAL_SETS["opendecisions"])[:n], out)
    seconds = time.perf_counter() - t0
    (out.parent / "timing.json").write_text(json.dumps({"seconds": seconds}))
    runs.commit()
    return {"logits": out.read_text(), "seconds": seconds}


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=30 * 60)
def pack_check_remote(n_short: int, n_long: int, pack_tokens: int, attn: str) -> dict:
    """The same items scored unpacked (reference: default sdpa) and packed with the given attention
    implementation: logits must agree (masking and positions are per prompt)."""
    from dopest_model import torch_backend as tb
    items = read(EVAL_SETS["opendecisions"])[:n_short] + \
        [it for it in read(EVAL_SETS["gen_test"]) if it["family"] == "long_policy"][:n_long]
    res = {}
    for name, pt, impl in (("unpacked", 0, None), ("packed", pack_tokens, attn)):
        model, tokenizer = tb.load(BASE, attn=impl)
        out = Path(f"/runs/pack_check/{name}-{attn}.jsonl")
        tb.predict(model, tokenizer, items[:4], Path("/tmp/warmup.jsonl"), pack_tokens=pt)   # compile, warm up
        t0 = time.perf_counter()
        tb.predict(model, tokenizer, items, out, pack_tokens=pt)
        res[name] = {"logits": out.read_text(), "seconds": time.perf_counter() - t0}
        del model
    runs.commit()
    return res


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=30 * 60)
def train_bench_remote(attns: list[str]) -> list[dict]:
    import torch
    from dopest_model import torch_backend as tb
    out = []
    for attn in attns:
        out.append(tb.train_bench(BASE, [f"/root/data/{p}" for p in TRAIN_FILES], attn))
        torch.cuda.empty_cache()
    return out


@app.local_entrypoint()
def attn_check(attn: str = "flex_attention"):
    """Packed scoring with `attn` vs the unpacked sdpa reference, then training throughput sdpa vs `attn`."""
    report_pack_check(pack_check_remote.remote(64, 32, 8192, attn))
    for r in train_bench_remote.remote(["sdpa", attn]):
        print(f"train {r['attn']:15s}: {r['tok_s']:,.0f} tok/s over {r['packs']} packs, peak {r['peak_gb']:.1f} GB")


@app.local_entrypoint()
def pack_check(n_short: int = 64, n_long: int = 32, pack_tokens: int = 8192, attn: str = "sdpa"):
    report_pack_check(pack_check_remote.remote(n_short, n_long, pack_tokens, attn))


def report_pack_check(res: dict) -> None:
    rows = {k: {r["id"]: r for r in map(json.loads, v["logits"].splitlines())} for k, v in res.items()}
    diffs, same, worst = [], 0, (0.0, "")
    for i, a in rows["unpacked"].items():
        b = rows["packed"][i]
        d = max(abs(x - y) for x, y in zip(a["logits"], b["logits"]))
        diffs.append(d)
        worst = max(worst, (d, f"{i} ({a['n_tokens']} tokens)"))
        same += max(range(len(a["logits"])), key=a["logits"].__getitem__) == \
            max(range(len(b["logits"])), key=b["logits"].__getitem__)
    tokens = sum(r["n_tokens"] for r in rows["unpacked"].values())
    print(f"{len(diffs)} items: same argmax {same}/{len(diffs)}, max |packed - unpacked| = {max(diffs):.3f} "
          f"(worst: {worst[1]}), mean = {sum(diffs) / len(diffs):.4f}")
    for k, v in res.items():
        print(f"{k:9s}: {v['seconds']:.1f} s, {tokens / v['seconds']:,.0f} tok/s")


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=PILOT_TIMEOUT_MIN * 60)
def pilot_remote(run: str, train_args: dict, eval_sets: dict, base_sets: dict | None = None,
                 train_min: float = PILOT_TRAIN_MIN, eval_deadline_min: float = PILOT_EVAL_DEADLINE_MIN,
                 model_id: str = BASE) -> dict:
    import gc
    import torch
    from dopest_model import torch_backend as tb
    t0 = time.perf_counter()
    minutes = lambda: (time.perf_counter() - t0) / 60
    out = Path("/runs") / run
    if base_sets:                               # the base model on these sets first, in the same container
        model, tokenizer = tb.load(model_id)
        for name, rel in base_sets.items():
            tb.predict(model, tokenizer, read(rel), out / f"base_{name}.jsonl", pack_tokens=EVAL_PACK_TOKENS)
            runs.commit()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    adapter = tb.train(model_id, [f"/root/data/{p}" for p in TRAIN_FILES], out, on_checkpoint=runs.commit,
                       max_minutes=train_min, **train_args)
    hf_cache.commit()
    model, tokenizer = tb.load(model_id, adapter=str(adapter), bidir=train_args.get("bidir"))
    done, skipped = [], []
    for name, rel in eval_sets.items():
        if minutes() > eval_deadline_min:
            skipped.append(name)
            continue
        tb.predict(model, tokenizer, read(rel), out / f"{name}.jsonl", pack_tokens=EVAL_PACK_TOKENS)
        done.append(name)
        runs.commit()
    seconds = time.perf_counter() - t0
    (out / "timing.json").write_text(json.dumps({"seconds": seconds, "evaluated": done, "skipped": skipped}))
    runs.commit()
    return {"seconds": seconds}


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=25 * 60)
def bidir_check_remote(modes: list[str], n: int, attns: list[str]) -> dict:
    """Base model on a random sample of n OpenDecisions items: causal vs bidirectional-within-prompt letter logits
    for every mode, per attention implementation. Checks that the masks build and run, how much they change the
    logits, and the zero-shot argmax accuracy of each mode (the least damaged mode is the easiest to adapt)."""
    import gc
    import random
    import torch
    from dopest_model import torch_backend as tb
    from dopest_model.evaluate import gold_index
    from dopest_model.prompt import messages
    items = random.Random(0).sample(read("raw/opendecisions/opendecisions.jsonl"), n)
    report = {}
    for attn in attns:
        model, tokenizer = tb.load(BASE, attn=attn)
        letters = tb.letter_token_ids(tokenizer)
        prepared = [(tb.prompt_tokens(tokenizer, messages(it)[0]), messages(it)[1]) for it in items]
        seqs, ids = [t for t, _ in prepared], [i for _, i in prepared]
        gold = [gold_index(it, i) for it, i in zip(items, ids)]
        packs = tb.packs([len(t) for t in seqs], 8192)
        out = {}
        for mode in [None] + modes:
            model.dopest_bidir = mode
            zs = []
            with torch.inference_mode():
                for group in packs:                                  # warm-up pass is not timed
                    zs += tb.packed_letter_logits(model, [seqs[i] for i in group], [letters[:len(ids[i])] for i in group])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                for group in packs:
                    tb.packed_letter_logits(model, [seqs[i] for i in group], [letters[:len(ids[i])] for i in group])
            torch.cuda.synchronize()
            logits = [z.cpu().tolist() for z in zs]
            out[mode or "causal"] = {"seconds": time.perf_counter() - t0, "logits": logits,
                                     "accuracy": sum(max(range(len(z)), key=z.__getitem__) == g for z, g in zip(logits, gold)) / len(gold)}
        a = out["causal"]["logits"]
        rep = {"items": len(a), "packs": len(packs), "modes": {}}
        for label, o in out.items():
            b = o["logits"]
            rep["modes"][label] = {"accuracy": o["accuracy"], "seconds": o["seconds"],
                                   "same_argmax": sum(max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__) for x, y in zip(a, b)),
                                   "mean_abs_diff": sum(abs(u - v) for x, y in zip(a, b) for u, v in zip(x, y)) / sum(map(len, a))}
        report[attn] = rep
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return report


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=50 * 60)
def score_modes_remote(model_id: str, runs_by_mode: dict, rels: dict, pack_tokens: int, adapter: str | None = None,
                       reverse_choice: bool = False) -> dict:
    """One model load (base, or base + the adapter of an earlier run), several attention modes (key: '' = causal,
    else a bidir mode) and several data files. Writes /runs/<run of the mode>/<name>.jsonl for every (mode, file).
    reverse_choice presents the options of choice questions in reverse order (for order averaging)."""
    from dopest_model import torch_backend as tb
    t0 = time.perf_counter()
    model, tokenizer = tb.load(model_id, adapter=f"/runs/{adapter}/adapter" if adapter else None)
    hf_cache.commit()
    load_s = time.perf_counter() - t0
    done = []
    for mode, run in runs_by_mode.items():
        model.dopest_bidir = mode or None
        for name, rel in rels.items():
            t1 = time.perf_counter()
            items = read(rel)
            if reverse_choice:
                items = [dict(it, question=dict(it["question"], criteria=dict(reversed(list(it["question"]["criteria"].items())))))
                         if it["question"]["type"] == "choice" and isinstance(it["question"]["criteria"], dict) else it
                         for it in items]
            tb.predict(model, tokenizer, items, Path("/runs") / run / f"{name}.jsonl", pack_tokens=pack_tokens)
            runs.commit()
            done.append({"run": run, "name": name, "seconds": time.perf_counter() - t1})
    return {"load_s": load_s, "done": done, "seconds": time.perf_counter() - t0}


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=40 * 60)
def serve_bench_remote(run: str, calibration: dict, model_id: str, bidir: str | None, rel: str, n: int) -> dict:
    """The real server (uvicorn, loopback HTTP) on the adapter of `run`: one request per decision, serially, as the
    board measures self-hosted systems. Reports p50/p95 latency, tokens per decision, and agreement of the served
    answers with the offline logits of the same run (a check of the serving path)."""
    import json as _json
    import statistics
    import threading
    import urllib.request
    import uvicorn
    from dopest_model.serve import Decider, create_app
    from dopest_model.evaluate import predictions
    items = read(rel)[:n]
    decider = Decider(model_id, f"/runs/{run}/adapter", calibration, bidir=bidir)
    app_ = create_app(decider, run)
    config = uvicorn.Config(app_, host="127.0.0.1", port=8111, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen("http://127.0.0.1:8111/health", timeout=2).read()
            break
        except Exception:
            time.sleep(1)
    decider.decide("warm-up state", {"w": {"type": "noul", "instructions": "warm?"}})
    lat, toks, answers = [], [], {}
    for it in items:
        q = {"type": it["question"]["type"], "instructions": it["question"]["instructions"]}
        if it["question"].get("criteria") is not None:
            q["criteria"] = it["question"]["criteria"]
        body = _json.dumps({"state": it["state"], "model": run, "questions": {"decision": q}}).encode()
        req = urllib.request.Request("http://127.0.0.1:8111/v1/systemone", body, {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            out = _json.loads(r.read())
        lat.append(time.perf_counter() - t0)
        toks.append(out["usage"]["input_tokens"])
        answers[it["id"]] = out["answers"]["decision"]
    # agreement with the offline logits (same adapter, same mask) under the same calibration
    offline = {}
    try:
        rows = [r for r in map(_json.loads, open(f"/runs/{run}/{Path(rel).stem}.jsonl")) if r["id"] in answers]
        for p in predictions(rows, calibration):
            offline[p["id"]] = p["probs"]
    except FileNotFoundError:
        pass
    same, diff = 0, []
    for i, a in answers.items():
        if i not in offline:
            continue
        served = a["probabilities"] if a["type"] != "noul" else {"yes": a["noul"], "no": 1 - a["noul"]}
        off = offline[i]
        same += max(served, key=served.get) == max(off, key=off.get)
        diff.append(max(abs(served[k] - off.get(k, 0.0)) for k in served))
    lat.sort()
    conformance = "jevbench not installed"
    try:
        from jevbench.adapters.conformance import check_adapter
        from jevbench.adapters.typesafe import TypeSafeAdapter
        report = check_adapter(TypeSafeAdapter(endpoint="http://127.0.0.1:8111", model=run, key_env=None))
        conformance = report.summary()
    except Exception as e:                      # the kit is optional; report what happened
        conformance = f"conformance kit failed: {type(e).__name__}: {str(e)[:300]}"
    return {"n": len(lat), "p50_s": lat[len(lat) // 2], "p95_s": lat[int(0.95 * len(lat)) - 1], "mean_s": statistics.mean(lat),
            "tokens_per_decision": statistics.mean(toks), "compared": len(diff), "same_argmax": same,
            "max_prob_diff": max(diff) if diff else None, "conformance": conformance}


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=40 * 60)
def score_remote(run: str, rel: str, name: str, split: str | None = None, adapter: str | None = None,
                 bidir: str | None = None) -> dict:
    """Letter logits of the base model (or the adapter of an earlier run) on one data file, saved to the Volume."""
    from dopest_model import torch_backend as tb
    t0 = time.perf_counter()
    items = [it for it in read(rel) if split is None or it.get("split") == split]
    model, tokenizer = tb.load(BASE, adapter=f"/runs/{adapter}/adapter" if adapter else None, bidir=bidir)
    out = Path("/runs") / run
    tb.predict(model, tokenizer, items, out / f"{name}.jsonl", pack_tokens=EVAL_PACK_TOKENS)
    seconds = time.perf_counter() - t0
    (out / f"{name}.timing.json").write_text(json.dumps({"seconds": seconds, "items": len(items)}))
    runs.commit()
    return {"seconds": seconds, "items": len(items)}


def compare_with_mlx(torch_text: str) -> None:
    torch_rows = {r["id"]: r for r in map(json.loads, torch_text.splitlines())}
    mlx_rows = {}
    with open(ROOT / "runs/g4-12b/opendecisions.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in torch_rows:
                mlx_rows[r["id"]] = r
    diffs, same = [], 0
    for i, r in torch_rows.items():
        a, b = r["logits"], mlx_rows[i]["logits"]
        diffs.append(max(abs(x - y) for x, y in zip(a, b)))
        same += max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
    print(f"{len(diffs)} items: max |torch - MLX| logit = {max(diffs):.3f}, mean = {sum(diffs) / len(diffs):.3f}, "
          f"same argmax {same}/{len(diffs)}")


def report_cost(seconds: float) -> None:
    print(f"remote time {seconds / 60:.1f} min, GPU cost ~${seconds / 3600 * USD_PER_HOUR:.2f}, with CPU/RAM "
          f"~${seconds / 3600 * (USD_PER_HOUR + 0.35):.2f} (plus container start)")


@app.local_entrypoint()
def smoke(n: int = 50, fetch: bool = False):
    """Base-model logits on the first n OpenDecisions items, compared with the local MLX run."""
    if not fetch:
        smoke_remote.remote(n)
    text = volume_text(f"smoke/first{n}.jsonl")
    if text is None:
        print("no smoke results on the Volume yet")
        return
    (ROOT / "runs/modal").mkdir(parents=True, exist_ok=True)
    (ROOT / f"runs/modal/smoke_first{n}.jsonl").write_text(text)
    compare_with_mlx(text)
    timing = volume_text("smoke/timing.json")
    if timing:
        report_cost(json.loads(timing)["seconds"])


@app.local_entrypoint()
def pilot(run: str = "pilot-12b-modal", limit: int = 0, lora_layers: int = 16, lr: float = 5e-5,
          train_temp: float = 4.0, soft: bool = False, evals: str = ",".join(EVAL_SETS), fetch: bool = False,
          pack_tokens: int = 8192, step_tokens: int = 16384, train_attn: str = "flex_attention",
          base_evals: str = "", train_min: int = PILOT_TRAIN_MIN, eval_deadline_min: int = PILOT_EVAL_DEADLINE_MIN,
          timeout_min: int = PILOT_TIMEOUT_MIN, grad_clip: float = 0.0, bidir: str = "", model: str = BASE,
          gpu: str = ""):
    """Train on the train splits of block A (four generators) and block B, evaluate on the dev sets, copy back.
    pack_tokens = 0 trains one example per forward (the first pilot's setting). Training uses FlexAttention
    (block-sparse packed masks: 2.4x faster than sdpa in backward), evaluation stays on packed sdpa (2.9x faster
    than packed flex in forward-only scoring)."""
    if not fetch:
        train_args = {"limit": limit or None, "lora_layers": lora_layers, "lr": lr, "train_temp": train_temp,
                      "soft": soft, "pack_tokens": pack_tokens, "step_tokens": step_tokens,
                      "attn": train_attn or None, "grad_clip": grad_clip or None, "bidir": bidir or None}
        base_sets = {k: EVAL_SETS[k] for k in base_evals.split(",") if k}
        opts = {"timeout": timeout_min * 60} | ({"gpu": gpu} if gpu else {})
        pilot_remote.with_options(**opts).remote(
            run, train_args, {k: EVAL_SETS[k] for k in evals.split(",")}, base_sets, train_min, eval_deadline_min,
            model)
    local = ROOT / "runs" / run
    local.mkdir(parents=True, exist_ok=True)
    got = []
    for name in OUTPUTS + [f"base_{k}.jsonl" for k in EVAL_SETS]:
        text = volume_text(f"{run}/{name}")
        if text is not None:
            (local / name).write_text(text)
            got.append(name)
    print(f"{run}: fetched {', '.join(got) or 'nothing yet'} into {local}; adapter stays on Volume dopest-runs/{run}/adapter")
    timing = volume_text(f"{run}/timing.json")
    if timing:
        report_cost(json.loads(timing)["seconds"])


@app.local_entrypoint()
def score(run: str, data: str, name: str, split: str = "", adapter: str = "", bidir: str = "", fetch: bool = False):
    """Score one data file (base model, or --adapter <run>) and copy the logits back to runs/<run>/<name>.jsonl:

        uv run modal run --detach src/dopest_model/modal_app.py::score --run base-12b \\
            --data train/blockb_v1.jsonl --name blockb_train --split train
    """
    if not fetch:
        res = score_remote.remote(run, data, name, split or None, adapter or None, bidir or None)
        print(f"{res['items']} items scored")
        report_cost(res["seconds"])
    text = volume_text(f"{run}/{name}.jsonl")
    if text is None:
        print("no results on the Volume yet")
        return
    local = ROOT / "runs" / run
    local.mkdir(parents=True, exist_ok=True)
    (local / f"{name}.jsonl").write_text(text)
    print(f"{run}: {len(text.splitlines())} rows -> {local / (name + '.jsonl')}")


@app.local_entrypoint()
def bidir_check(modes: str = "all,full,sliding", n: int = 240, attns: str = "sdpa"):
    """~$0.3: do the bidirectional-within-prompt masks build and run, and how much does each mode cost zero-shot?"""
    report = bidir_check_remote.remote(modes.split(","), n, attns.split(","))
    for attn, r in report.items():
        print(f"{attn}: {r['items']} items in {r['packs']} packs")
        for label, m in r["modes"].items():
            print(f"  {label:8s} accuracy {m['accuracy']:.1%}  same argmax as causal {m['same_argmax']}/{r['items']}  "
                  f"mean |diff| {m['mean_abs_diff']:.2f}  {m['seconds']:.1f} s")


@app.local_entrypoint()
def score_modes(model: str, runs_by_mode: str, data: str, pack_tokens: int = EVAL_PACK_TOKENS, gpu: str = "",
                adapter: str = "", reverse_choice: bool = False):
    """Frozen model, several mask modes and data files in one container:

        uv run modal run --detach src/dopest_model/modal_app.py::score_modes --model google/gemma-4-31B-it \\
            --runs-by-mode "=g4-31b,full=g4-31b-bidir-full" \\
            --data "opendecisions=raw/opendecisions/opendecisions.jsonl,jevbench_public=dev/jevbench_public.jsonl"
    """
    modes = dict(part.split("=", 1) for part in runs_by_mode.split(","))
    rels = dict(part.split("=", 1) for part in data.split(","))
    fn = score_modes_remote.with_options(gpu=gpu) if gpu else score_modes_remote
    res = fn.remote(model, modes, rels, pack_tokens, adapter or None, reverse_choice)
    print(f"load {res['load_s'] / 60:.1f} min; " + "; ".join(f"{d['run']}/{d['name']} {d['seconds']:.0f} s" for d in res["done"]))
    report_cost(res["seconds"])
    for run in modes.values():
        local = ROOT / "runs" / run
        local.mkdir(parents=True, exist_ok=True)
        for name in rels:
            text = volume_text(f"{run}/{name}.jsonl")
            if text is not None:
                (local / f"{name}.jsonl").write_text(text)
                print(f"{run}: {len(text.splitlines())} rows -> {local / (name + '.jsonl')}")


@app.local_entrypoint()
def serve_bench(run: str = "pilot-31b-ab-bidir-full", calibration: str = "runs/pilot-31b-ab-bidir-full/calibration.json",
                model: str = "google/gemma-4-31B-it", bidir: str = "full", data: str = "dev/jevbench_public.jsonl",
                n: int = 231, gpu: str = ""):
    """~$0.5: latency and correctness of the real server on the board's protocol (serial loopback requests)."""
    calib = json.load(open(calibration, encoding="utf-8"))
    fn = serve_bench_remote.with_options(gpu=gpu) if gpu else serve_bench_remote
    r = fn.remote(run, calib, model, bidir or None, data, n)
    adj = lambda s: 2 * s + 0.15
    print(f"{r['n']} decisions: p50 {r['p50_s'] * 1000:.0f} ms, p95 {r['p95_s'] * 1000:.0f} ms, mean {r['mean_s'] * 1000:.0f} ms; "
          f"board-adjusted (2x + 150 ms): p50 {adj(r['p50_s']):.2f} s, p95 {adj(r['p95_s']):.2f} s; "
          f"{r['tokens_per_decision']:.0f} input tokens per decision")
    print(f"served vs offline logits: same argmax {r['same_argmax']}/{r['compared']}, max |Δp| {r['max_prob_diff']}")
    print("conformance kit:\n" + str(r.get("conformance")))

