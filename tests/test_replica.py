import json

from dopest_model.replica import spec
from dopest_model.replica.assemble import collect, select
from dopest_model.replica.parse import extract_json, normalise, parse_blind, parse_verdict
from dopest_model.replica.prompts import author_prompt, blind_prompt, verdict_prompt
from dopest_model.replica.stages import author_rows


def test_quotas_follow_the_sealed_layout():
    q = spec.quotas()
    assert sum(q.values()) == 720
    for qtype, tiers in spec.SEALED.items():
        for tier, n in tiers.items():
            assert sum(v for (t, ti, _), v in q.items() if t == qtype and ti == tier) == n
        st = sum(v for (t, _, f), v in q.items() if t == qtype and f == "state_tracking")
        assert st >= 0.12 * sum(tiers.values())
    for (qtype, tier, fam) in q:
        assert tier in spec.FAMILIES[fam]["tiers"] and qtype in spec.FAMILIES[fam]["types"]


def test_plan_over_authors_and_splits_between_the_two_authors():
    briefs = spec.plan(seed=3)
    assert 720 * 1.2 <= len(briefs) <= 720 * 1.32
    assert len({b["id"] for b in briefs}) == len(briefs)
    share = sum(b["author"] == "gptoss" for b in briefs) / len(briefs)
    assert 0.42 <= share <= 0.58
    assert all(b["json_state"] for b in briefs if b["tier"] == "judge")
    assert all(b["type"] == "choice" for b in briefs if b["family"] == "probability")
    assert spec.plan(seed=3) == briefs                                        # deterministic


def test_author_prompt_carries_the_brief_and_the_tier_rules():
    b = next(b for b in spec.plan(seed=1) if b["family"] == "probability")
    msgs = author_prompt(b)
    user = msgs[1]["content"]
    assert spec.FAMILIES["probability"]["brief"][:40] in user and "Hard tier" in user and "gold_probs" in user
    assert b["setting"] in user and "snake_case" in msgs[0]["content"]
    n = next(b for b in spec.plan(seed=1) if b["type"] == "noul" and b["tier"] == "easy")
    assert f'must be "{n["gold"]}"' in author_prompt(n)[1]["content"]


ITEM_RAW = {"state": "Ticket 4411: the customer asks for a refund of the March invoice.",
            "question": {"type": "choice", "instructions": "Route the ticket.",
                         "criteria": {"billing": "Charges and refunds", "shipping": "Delivery", "other": "Anything else"}},
            "expected": "billing", "rationale": "The customer asks for a refund, which the billing team handles under the criteria.",
            "surface_answer": "other", "why_hard": "Not hard.", "gold_probs": None}
BRIEF = {"id": "rp-x", "type": "choice", "tier": "standard", "family": "routing", "words": 12, "gold": None}


def test_extract_json_skips_thinking_and_fences():
    text = "<think>{not json}</think>Here you go:\n```json\n" + json.dumps(ITEM_RAW) + "\n```\nDone."
    assert extract_json(text)["expected"] == "billing"
    assert extract_json('prefix {"a": "brace } inside", "b": {"c": 1}} suffix') == {"a": "brace } inside", "b": {"c": 1}}
    assert extract_json("no object here") is None and extract_json("") is None


def test_normalise_checks_every_rule():
    item, why = normalise(ITEM_RAW, BRIEF, "qwen")
    assert why == "" and item["id"].startswith("rp-") and item["expected"] == "billing"
    assert set(item["labels"]) == {"billing", "shipping", "other"} and item["provenance"]["author_model"] == "qwen"
    assert normalise(dict(ITEM_RAW, state={"gold": "billing", "text": "x" * 60}), BRIEF, "qwen")[1] == "banned key in state"
    assert "length" in normalise(dict(ITEM_RAW, state="too short"), BRIEF, "qwen")[1]
    assert "type" in normalise(ITEM_RAW, dict(BRIEF, type="noul"), "qwen")[1]
    noul = dict(ITEM_RAW, question={"type": "noul", "instructions": "Refund asked?", "criteria": {"true": "yes", "false": "no"}},
                expected="true", surface_answer=False)
    it, why = normalise(noul, dict(BRIEF, type="noul"), "gptoss")
    assert why == "" and it["expected"] == "yes" and it["labels"] == ["no", "yes"] and it["provenance"]["surface_answer"] == "no"
    score = dict(ITEM_RAW, question={"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]},
                 expected=2, surface_answer=1)
    it, why = normalise(score, dict(BRIEF, type="score"), "gptoss")
    assert why == "" and it["expected"] == "2" and it["labels"] == ["0", "1", "2"] and it["provenance"]["surface_answer"] == 1
    prob = dict(ITEM_RAW, gold_probs={"billing": 0.7, "shipping": 0.2, "other": 0.1})
    assert normalise(prob, dict(BRIEF, family="probability"), "qwen")[1] == ""
    assert "gold_probs" in normalise(dict(prob, gold_probs={"billing": 0.4, "shipping": 0.5, "other": 0.1}),
                                     dict(BRIEF, family="probability"), "qwen")[1]


def test_review_parsers_and_prompts():
    item, _ = normalise(ITEM_RAW, BRIEF, "qwen")
    assert parse_blind({"answer": "billing", "probs": {"billing": 0.9}, "note": "refund"}, item)["ok"]
    assert not parse_blind({"answer": "nonsense"}, item)["ok"] and not parse_blind(None, item)["ok"]
    noul_item = dict(item, labels=["no", "yes"], question=dict(item["question"], type="noul"))
    assert parse_blind({"answer": "True"}, noul_item)["answer"] == "yes"
    assert parse_verdict({"verdict": " Accept ", "reason": "fine"}) == {"verdict": "accept", "reason_class": None, "reason": "fine", "fix": None, "ok": True}
    assert parse_verdict({"verdict": "reject", "reason_class": "Gold wrong", "reason": "x"})["reason_class"] == "gold_wrong"
    assert not parse_verdict({"verdict": "maybe"})["ok"]
    assert '"labels"' in blind_prompt(item)[1]["content"] and "probs" in blind_prompt(item)[1]["content"]
    v = verdict_prompt(item, {"answer": "other"})[1]["content"]
    assert '"gold": "billing"' in v and '"your_blind_answer": "other"' in v and "rationale" in v


def test_author_rows_and_assembly():
    briefs = [dict(BRIEF, id="rp-a"), dict(BRIEF, id="rp-b")]
    outs = [{"text": json.dumps(ITEM_RAW), "tokens": 500, "finish": "stop"}, {"text": "garbage", "tokens": 9000, "finish": "length"}]
    rows = author_rows(briefs, outs, "gptoss")
    assert rows[0]["ok"] and not rows[1]["ok"] and rows[1]["why"].startswith("no JSON")
    items = collect(rows + rows)                                               # duplicates collapse
    assert len(items) == 1
    item = items[0]
    chosen, report = select([item], [{"id": item["id"], "verdict": "accept", "ok": True}],
                            [{"id": item["id"], "answer": "billing", "ok": True}])
    assert chosen == [item] and report["families"]["routing"]["blind_accuracy"] == 1.0
    assert select([item], [{"id": item["id"], "verdict": "reject", "ok": True}], [])[0] == []


def test_topup_briefs_cover_only_the_short_cells_of_one_tier():
    have = {cell: n for cell, n in spec.quotas().items()}           # everything full
    have[("choice", "hard", "trap")] -= 4
    have[("noul", "hard", "judge_hard")] -= 2
    have[("choice", "easy", "intent")] -= 5                          # another tier: ignored
    briefs = spec.plan_topup(seed=5, have=have, tier="hard", over=2.0, author="gptoss")
    cells = {b["cell"] for b in briefs}
    assert cells == {"choice/hard/trap", "noul/hard/judge_hard"} and len(briefs) == 12
    assert all(b["author"] == "gptoss" and b["tier"] == "hard" for b in briefs)
    assert len({b["id"] for b in briefs}) == 12 and not ({b["id"] for b in briefs} & {b["id"] for b in spec.plan(1)})
    assert "Final check" in author_prompt(briefs[0])[1]["content"]


def test_training_round_briefs_follow_the_allocation_with_disjoint_ids():
    briefs = spec.plan_round(seed=9, allocation={("noul", "judge", "judge_math"): 3, ("choice", "hard", "trap"): 2}, author="gptoss")
    assert [b["cell"] for b in briefs] == ["choice/hard/trap"] * 2 + ["noul/judge/judge_math"] * 3
    assert all(b["id"].startswith("rt-") and b["author"] == "gptoss" for b in briefs) and len({b["id"] for b in briefs}) == 5
    assert sum(spec.TRAINING_ROUND.values()) > 2000 and all(t in spec.FAMILIES[fam]["tiers"] and q in spec.FAMILIES[fam]["types"]
                                                            for (q, t, fam) in spec.TRAINING_ROUND)
    half = spec.plan_round(seed=9, allocation={k: max(1, round(v * 0.5)) for k, v in spec.TRAINING_ROUND.items()})
    assert 0.45 * sum(spec.TRAINING_ROUND.values()) < len(half) < 0.55 * sum(spec.TRAINING_ROUND.values())


def test_training_selection_keeps_accepts_and_harmless_rejects_only():
    from dopest_model.replica.assemble import select_training
    items = [dict(ITEM_RAW, id=f"rp-{i}", tier="standard", family="routing", provenance={"author_model": "gptoss"},
                  question=ITEM_RAW["question"], labels=["billing", "shipping", "other"]) for i in range(5)]
    verdicts = [{"id": "rp-0", "ok": True, "verdict": "accept", "reason_class": "none"},
                {"id": "rp-1", "ok": True, "verdict": "reject", "reason_class": "leak"},
                {"id": "rp-2", "ok": True, "verdict": "reject", "reason_class": "gold_wrong"},
                {"id": "rp-3", "ok": True, "verdict": "reject", "reason_class": None}]
    kept, why = select_training(items, verdicts)
    assert [it["id"] for it in kept] == ["rp-0", "rp-1"] and all(it["split"] == "train" for it in kept)
    assert kept[1]["provenance"]["review"] == "reject:leak"
    assert why == {"accept": 1, "reject:leak": 1, "dropped:gold_wrong": 1, "dropped:unclassified": 1, "no verdict": 1}
