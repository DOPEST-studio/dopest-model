"""Write block B items with DeepSeek V4.1 Flash (official API, thinking off, JSON mode).

    uv run python -m dopest_model.blockb.writer --calls 4700 --seed 101 --concurrency 32 \
        --out data/blockb/batch1_raw.jsonl

Briefs are drawn from --seed, so a run is resumable: items are appended as each call returns, and a rerun with the
same arguments skips the calls already in --out. The API key is read from 1Password
(op://dopest-infra/deepseek-api/password) into memory only. Calls during DeepSeek peak hours (Mon-Fri 01:00-04:00
and 06:00-10:00 UTC, double price) are skipped unless --allow-peak; a rerun later finishes them. Every item is
validated before it is written; cost comes from the token usage the API reports.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import random
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .specs import FAMILIES, LANGUAGE_WEIGHTS, MULTILINGUAL_FAMILIES, SYSTEM, TIERS, language_rule

MODEL = "deepseek-flash"
PRICES = {"off": {"hit": 0.003, "miss": 0.15, "out": 0.60}, "peak": {"hit": 0.006, "miss": 0.30, "out": 1.20}}
SNAKE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def peak_now(t: datetime | None = None) -> bool:
    t = t or datetime.now(timezone.utc)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


def api_key() -> str:
    """DEEPSEEK_API_KEY when started under `op run --env-file=scripts/blockb.op.env`, else read from 1Password."""
    return os.environ.get("DEEPSEEK_API_KEY") or subprocess.run(
        ["op", "read", "op://dopest-infra/deepseek-api/password"], capture_output=True, text=True, check=True).stdout.strip()


def state_key(state) -> str:
    """A state as text: JSON objects with sorted keys, so the same object always gives the same key."""
    return json.dumps(state, ensure_ascii=False, sort_keys=True) if isinstance(state, (dict, list)) else str(state)


def normalise(raw: dict, brief: dict) -> dict | None:
    """One raw item from the teacher into the project's item format, or None if it breaks a rule."""
    state, q = raw.get("state"), raw.get("question") or {}
    qtype, crit, exp = q.get("type"), q.get("criteria"), raw.get("expected")
    if not isinstance(state, (str, dict)) or not 40 <= len(state_key(state)) <= 12000 or not q.get("instructions"):
        return None
    if qtype == "choice":
        if not isinstance(crit, dict) or not 3 <= len(crit) <= 7 or not all(SNAKE.match(k) for k in crit):
            return None
        trap = raw.get("trap_answer")
        if exp not in crit or (trap is not None and (trap not in crit or trap == exp)):
            return None
        labels = list(crit)
    elif qtype == "noul":
        exp = {"true": "yes", "false": "no", True: "yes", False: "no"}.get(exp, exp)   # criteria keys used as answers
        if not isinstance(crit, dict) or set(crit) != {"true", "false"} or exp not in ("yes", "no"):
            return None
        labels, trap = ["no", "yes"], None
    elif qtype == "score":
        if not isinstance(crit, list) or not 3 <= len(crit) <= 6:
            return None
        try:
            exp = int(exp)
        except (TypeError, ValueError):
            return None
        if not 0 <= exp < len(crit):
            return None
        labels, trap = [str(i) for i in range(len(crit))], None
    else:
        return None
    tier = raw.get("tier") if raw.get("tier") in TIERS else brief["tier"]
    lang = brief.get("language", "en")
    item = {"family": "multilingual" if lang != "en" else brief["family"], "base_family": brief["family"],
            "state": state,
            "question": {"type": qtype, "instructions": q["instructions"], "criteria": crit},
            "labels": labels, "expected": exp, "trap_answer": trap, "tier": tier, "language": lang,
            "use_case": brief["use_case"], "topic": brief.get("domain", brief.get("task", brief.get("setting", ""))),
            "label_source": "teacher", "target_probs": None,
            "provenance": {"source": f"dopest-blockb/{MODEL}", "license": "Apache-2.0",
                           "plan": raw.get("plan", ""), "rationale": raw.get("reasoning") or raw.get("rationale", ""),
                           "generated_by": "DeepSeek V4.1 Flash (AI-generated)"}}
    if raw.get("person_name"):
        item["person_name"] = raw["person_name"]
    item["id"] = "bb-" + hashlib.sha1(f"{state_key(state)}\x1f{q['instructions']}\x1f{qtype}".encode()).hexdigest()[:12]
    return shuffled_options(item)


def shuffled_options(item: dict) -> dict:
    """Choice options in an order fixed by the item id (the teacher tends to list the answer first).

    Starts from the sorted ids, so applying it twice gives the same order."""
    if item["question"]["type"] != "choice":
        return item
    keys = sorted(item["question"]["criteria"])
    random.Random(item["id"]).shuffle(keys)
    crit = {k: item["question"]["criteria"][k] for k in keys}
    return dict(item, labels=keys, question=dict(item["question"], criteria=crit))


def split_of(group: str) -> str:
    """80/10/10 by group, so items that share a state never land in different splits."""
    h = int(hashlib.sha1(group.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "calib" if h == 1 else "train"


def parse_langs(spec: str) -> dict[str, float]:
    """'od' (OpenDecisions shares), 'pl,es' (equal shares) or 'pl=2,es=1'."""
    if spec == "od":
        return dict(LANGUAGE_WEIGHTS)
    pairs = (part.partition("=") for part in spec.split(",") if part)
    return {code: float(w or 1) for code, _, w in pairs}


def plan_briefs(calls: int, mix: dict[str, float], langs: dict[str, float], non_en: float,
                seed: int) -> list[tuple[dict, str]]:
    """(brief, prompt) per call, fixed by the seed. Non-English calls go to the language furthest below its share."""
    rng = random.Random(seed)
    briefs, counts, n_other, total = [], collections.Counter(), 0, sum(langs.values())
    for _ in range(calls):
        lang = "en"
        if langs and rng.random() < non_en:
            n_other += 1
            lang = max(langs, key=lambda c: langs[c] / total * n_other - counts[c])
            counts[lang] += 1
        fams = mix if lang == "en" else MULTILINGUAL_FAMILIES
        brief_fn, prompt_fn = FAMILIES[rng.choices(list(fams), weights=list(fams.values()))[0]]
        b = dict(brief_fn(rng), language=lang)
        briefs.append((b, prompt_fn(b) + language_rule(lang)))
    return briefs


async def _create(client, prompt: str):
    for attempt in range(3):
        try:
            return await client.chat.completions.create(
                model=MODEL, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}, temperature=1.0, max_tokens=6000,
                extra_body={"thinking": {"type": "disabled"}})
        except Exception as e:                              # rate limits, transient server errors
            if attempt == 2:
                print(f"call failed: {type(e).__name__}: {str(e)[:120]}")
                return None
            await asyncio.sleep(2 ** attempt * 3)


def _count(usage: dict, u) -> None:
    price = PRICES["peak" if peak_now() else "off"]
    hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
    miss = getattr(u, "prompt_cache_miss_tokens", 0) or (u.prompt_tokens or 0)
    out = u.completion_tokens or 0
    usage["hit"], usage["miss"], usage["out"], usage["calls"] = (usage["hit"] + hit, usage["miss"] + miss,
                                                                 usage["out"] + out, usage["calls"] + 1)
    usage["usd"] += (hit * price["hit"] + miss * price["miss"] + out * price["out"]) / 1e6


async def one_call(client, sem, brief: dict, prompt: str, usage: dict, allow_peak: bool) -> tuple[list[dict], int]:
    async with sem:
        if not allow_peak and peak_now():
            usage["skipped"] += 1
            return [], 0
        for _ in range(2):                                  # one more try when the JSON comes back broken
            resp = await _create(client, prompt)
            if resp is None:
                return [], 0
            _count(usage, resp.usage)
            try:
                raw_items = json.loads(resp.choices[0].message.content or "{}").get("items", [])
                break
            except json.JSONDecodeError:
                usage["json_errors"] += 1
        else:
            return [], 0
    group = "bbg-" + hashlib.sha1((prompt + str(time.time_ns())).encode()).hexdigest()[:12]
    items = []
    for raw in raw_items if isinstance(raw_items, list) else []:
        it = normalise(raw, brief) if isinstance(raw, dict) else None
        if it is None:
            continue
        it["group"] = group if brief["family"] == "llm_router" else f"{group}-{len(items)}"
        it["split"] = split_of(it["group"])
        items.append(it)
    if brief["family"] == "llm_router" and len({state_key(it["state"]) for it in items}) > 1:
        return [], len(raw_items)                           # the questions must share one message verbatim
    return items, len(raw_items)


async def run(briefs: list[tuple[dict, str]], out: Path, concurrency: int, allow_peak: bool) -> None:
    from openai import AsyncOpenAI
    done, seen = set(), set()
    if out.exists():
        for it in map(json.loads, open(out, encoding="utf-8")):
            done.add(it.get("call"))
            seen.add(it["id"])
    todo = [(i, b, p) for i, (b, p) in enumerate(briefs) if i not in done]
    print(f"{len(briefs) - len(todo)} of {len(briefs)} calls already in {out}; {len(todo)} to go", flush=True)
    client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=api_key(), timeout=180)
    sem = asyncio.Semaphore(concurrency)
    usage = {"hit": 0, "miss": 0, "out": 0, "calls": 0, "skipped": 0, "json_errors": 0, "usd": 0.0}
    by_lang = collections.defaultdict(lambda: [0, 0])
    written, proposed, t0 = 0, 0, time.perf_counter()

    async def job(i: int, b: dict, p: str):
        return i, b, await one_call(client, sem, b, p, usage, allow_peak)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for k, fut in enumerate(asyncio.as_completed([job(i, b, p) for i, b, p in todo]), 1):
            i, b, (items, n) = await fut
            proposed += n
            by_lang[b["language"]][1] += n
            for it in items:
                if it["id"] not in seen:
                    seen.add(it["id"])
                    f.write(json.dumps(dict(it, call=i), ensure_ascii=False) + "\n")
                    written += 1
                    by_lang[b["language"]][0] += 1
            f.flush()
            if k % 250 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} calls, {written} items, ${usage['usd']:.2f}, "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)
    print(f"{usage['calls']} calls answered; items: {proposed} proposed, {written} new valid -> {out}")
    print("valid/proposed by language:", ", ".join(f"{k} {v}/{n}" for k, (v, n) in sorted(by_lang.items())))
    print(f"tokens: cache hit {usage['hit']:,}, miss {usage['miss']:,}, out {usage['out']:,}; cost ${usage['usd']:.3f}; "
          f"broken JSON retried {usage['json_errors']}x")
    if usage["skipped"]:
        print(f"{usage['skipped']} calls skipped in peak hours: rerun the same command off-peak to finish")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calls", type=int, required=True)
    parser.add_argument("--mix", default="llm_router=0.5,usecase=0.3,fairness=0.2")
    parser.add_argument("--langs", default="od", help="'od' (OpenDecisions shares), 'pl,es' or 'pl=2,es=1'")
    parser.add_argument("--non-en", type=float, default=0.2, help="share of calls written in those languages")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-peak", action="store_true")
    args = parser.parse_args()
    if peak_now() and not args.allow_peak:
        raise SystemExit("DeepSeek peak hours (double price): rerun later or pass --allow-peak")
    mix = {k: float(v) for k, v in (p.split("=") for p in args.mix.split(","))}
    briefs = plan_briefs(args.calls, mix, parse_langs(args.langs), args.non_en, args.seed)
    asyncio.run(run(briefs, Path(args.out), args.concurrency, args.allow_peak))


if __name__ == "__main__":
    main()
