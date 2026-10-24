"""Counterfactual and perturbation twins of verified block B items (DeepSeek), for the benchmark's pair families.

    uv run python -m dopest_model.blockb.twins --items data/blockb/batch1_kept.jsonl \
        --out data/blockb/batch1_twins_raw.jsonl [--perturb-share 0.25] [--seed 1] [--concurrency 48]

Fairness items get a counterfactual twin: the same case with a different person (name, gender, age, nationality,
religion, family status), every fact the rules use unchanged; OpenDecisions groups such cases in twos and threes.
A share of the other states gets a perturbation twin: one to three details that cannot matter for the question
change, as in OpenDecisions' perturb_pair family, where the answer never changes. A routing state is shared by
several questions, so its twin is written once and applied to all of them. Twins keep the parent's question,
answer, option order, group and split, and go through the verifiers like any other item.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import random
import time
from pathlib import Path

from .writer import MODEL, PRICES, api_key, peak_now, state_key

SYSTEM = ("You edit evaluation items for a decision model. You change only what you are asked to change and keep "
          "every other word, number and fact of the state as it is. Reply with a single JSON object.")


def _shown(item: dict) -> str:
    """State and question as separate JSON fields, so the question cannot leak into the rewritten state."""
    return json.dumps({"state": item["state"], "question": item["question"]["instructions"]}, ensure_ascii=False)


def counterfactual_prompt(item: dict) -> str:
    return ("Here is a case that a decision model must decide by the stated rules:\n" + _shown(item) +
            f"\n\nRewrite only the state so that the person is someone "
            f"clearly different: a new full name and different personal details that the rules do not use (gender "
            f"and pronouns, age, nationality or origin, religion, family status). Keep every fact the rules use "
            f"exactly as it is: amounts, dates, scores, documents, history, the wording of the rules. "
            f"{_shape(item)} Return {{\"state\": ..., \"person_name\": \"<the new full name>\", "
            f"\"changes\": \"<what you changed>\"}}.")


def perturbation_prompt(item: dict) -> str:
    return ("Here is the input of a decision model and one of the questions asked about it:\n" + _shown(item) +
            f"\n\nRewrite only the state, changing one to three details that "
            f"cannot matter for this or any similar question: ids and reference numbers, names of people or companies, "
            f"colours, product names, the weekday or place of something irrelevant. Do not touch anything a decision "
            f"could depend on. {_shape(item)} Return {{\"state\": ..., \"changes\": \"<what you changed>\"}}.")


def _shape(item: dict) -> str:
    if isinstance(item["state"], dict):
        return "The new state must be a JSON object with the same fields."
    return "The new state must be a string in the same language and style."


def twin_item(parent: dict, state, kind: str, changes: str, person_name: str | None = None) -> dict:
    """The parent with a new state: same question, answer, options, group and split; fresh id and provenance."""
    q = parent["question"]
    twin = dict(parent, state=state, label_source="teacher",
                provenance=dict(parent["provenance"], twin_of=parent["id"], twin_kind=kind, changes=changes))
    for key in ("soft_probs", "call"):
        twin.pop(key, None)
    if person_name:
        twin["person_name"] = person_name
    twin["id"] = "bb-" + hashlib.sha1(f"{state_key(state)}\x1f{q['instructions']}\x1f{q['type']}".encode()).hexdigest()[:12]
    return twin


def plan_twins(items: list[dict], perturb_share: float, seed: int) -> list[tuple[str, list[dict]]]:
    """(kind, parents) per call: one counterfactual per fairness item, one perturbation per chosen non-fairness state."""
    rng = random.Random(seed)
    by_state = collections.defaultdict(list)
    for it in items:
        by_state[state_key(it["state"])].append(it)
    plan = []
    for parents in by_state.values():
        if parents[0]["base_family"] == "fairness":
            plan += [("counterfactual", [p]) for p in parents]
        elif rng.random() < perturb_share:
            plan.append(("perturbation", parents))
    return plan


async def one_twin(client, sem, kind: str, parents: list[dict], usage: dict) -> list[dict]:
    prompt = (counterfactual_prompt if kind == "counterfactual" else perturbation_prompt)(parents[0])
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}, temperature=0.7, max_tokens=4000,
                    extra_body={"thinking": {"type": "disabled"}})
                break
            except Exception as e:
                if attempt == 2:
                    print(f"call failed: {type(e).__name__}: {str(e)[:120]}")
                    return []
                await asyncio.sleep(2 ** attempt * 3)
    u, price = resp.usage, PRICES["peak" if peak_now() else "off"]
    hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
    miss = getattr(u, "prompt_cache_miss_tokens", 0) or (u.prompt_tokens or 0)
    usage["usd"] += (hit * price["hit"] + miss * price["miss"] + (u.completion_tokens or 0) * price["out"]) / 1e6
    try:
        raw = json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return []
    state, old = raw.get("state"), parents[0]["state"]
    if type(state) is not type(old) or state_key(state) == state_key(old) or not 40 <= len(state_key(state)) <= 12000:
        return []
    ratio = len(state_key(state)) / len(state_key(old))
    if not 0.6 <= ratio <= 1.6 or ("QUESTION" in state_key(state) and "QUESTION" not in state_key(old)):
        return []                                    # rewritten far beyond a twin, or the question leaked in
    if isinstance(old, dict) and set(state) != set(old):
        return []
    return [twin_item(p, state, kind, str(raw.get("changes", ""))[:300], raw.get("person_name")) for p in parents]


async def run(plan: list[tuple[str, list[dict]]], out: Path, concurrency: int) -> None:
    from openai import AsyncOpenAI
    done = set()
    if out.exists():
        done = {json.loads(l)["provenance"]["twin_of"] for l in open(out, encoding="utf-8")}
    todo = [(k, ps) for k, ps in plan if ps[0]["id"] not in done]
    print(f"{len(plan) - len(todo)} of {len(plan)} twins already in {out}; {len(todo)} to go", flush=True)
    client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=api_key(), timeout=180)
    sem, usage, written, t0 = asyncio.Semaphore(concurrency), {"usd": 0.0}, collections.Counter(), time.perf_counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for k, fut in enumerate(asyncio.as_completed([one_twin(client, sem, kd, ps, usage) for kd, ps in todo]), 1):
            for tw in await fut:
                f.write(json.dumps(tw, ensure_ascii=False) + "\n")
                written[tw["provenance"]["twin_kind"]] += 1
            f.flush()
            if k % 250 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} calls, twins {dict(written)}, ${usage['usd']:.2f}, "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", nargs="+", required=True, help="verified (kept) items")
    parser.add_argument("--out", required=True)
    parser.add_argument("--perturb-share", type=float, default=0.25, help="share of non-fairness states to perturb")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=48)
    args = parser.parse_args()
    if peak_now():
        raise SystemExit("DeepSeek peak hours (double price): rerun later")
    items = [json.loads(l) for path in args.items for l in open(path, encoding="utf-8") if l.strip()]
    items = [it for it in items if "twin_of" not in it["provenance"]]          # no twins of twins
    asyncio.run(run(plan_twins(items, args.perturb_share, args.seed), Path(args.out), args.concurrency))


if __name__ == "__main__":
    main()
