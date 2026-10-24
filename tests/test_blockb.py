from datetime import datetime, timezone

from dopest_model.blockb.agree import decide
from dopest_model.blockb.writer import normalise, peak_now

BRIEF = {"family": "usecase", "use_case": "customer_support", "tier": "hard", "task": "x"}
CHOICE = {"state": "A customer writes that the parcel arrived broken and asks for a replacement right away.",
          "question": {"type": "choice", "instructions": "Which queue?",
                       "criteria": {"damage": "Damaged goods", "billing": "Billing", "other": "Other"}},
          "expected": "damage", "trap_answer": "other", "tier": "standard", "rationale": "It is damage."}


def test_normalise_accepts_valid_and_rejects_broken_items():
    it = normalise(CHOICE, BRIEF)
    assert sorted(it["labels"]) == ["billing", "damage", "other"] and it["id"].startswith("bb-")
    assert it["labels"] == list(it["question"]["criteria"])                          # letters follow labels
    assert normalise(dict(CHOICE, expected="refund"), BRIEF) is None                  # gold not an option
    assert normalise(dict(CHOICE, trap_answer="damage"), BRIEF) is None               # trap equals gold
    bad = dict(CHOICE, question=dict(CHOICE["question"], criteria={"Damage!": "x", "b": "y", "c": "z"}))
    assert normalise(bad, BRIEF) is None                                               # ids must be snake_case
    score = {"state": CHOICE["state"], "question": {"type": "score", "instructions": "How urgent?",
             "criteria": ["low", "mid", "high"]}, "expected": "2"}
    assert normalise(score, BRIEF)["expected"] == 2
    noul = {"state": CHOICE["state"], "question": {"type": "noul", "instructions": "Refund?",
            "criteria": {"true": "t", "false": "f"}}, "expected": "maybe"}
    assert normalise(noul, BRIEF) is None


def test_peak_hours():
    assert peak_now(datetime(2026, 10, 23, 7, 0, tzinfo=timezone.utc))         # Friday 07:00 UTC
    assert not peak_now(datetime(2026, 10, 23, 17, 0, tzinfo=timezone.utc))    # Friday 17:00 UTC
    assert not peak_now(datetime(2026, 10, 24, 7, 0, tzinfo=timezone.utc))     # Saturday


def test_keep_rule():
    it = normalise(CHOICE, BRIEF)
    agree = {"qwen": {"probs": {"damage": 0.9, "billing": 0.05, "other": 0.05}},
             "gptoss": {"probs": {"damage": 1.0, "billing": 0.0, "other": 0.0}}}
    assert decide(it, agree)[:2] == (True, "unanimous")
    split = {"qwen": {"probs": {"damage": 0.8, "billing": 0.1, "other": 0.1}},
             "gptoss": {"probs": {"damage": 0.3, "billing": 0.0, "other": 0.7}}}
    assert decide(it, split)[:2] == (True, "majority")                         # mean 0.55 on the gold
    weak = {"qwen": {"probs": {"damage": 0.55, "billing": 0.0, "other": 0.45}},
            "gptoss": {"probs": {"damage": 0.0, "billing": 0.0, "other": 1.0}}}
    assert decide(it, weak)[:2] == (False, "weak_majority")                    # mean 0.275 < 0.5
    no = {"qwen": {"probs": {"damage": 0.1, "other": 0.9, "billing": 0.0}},
          "gptoss": {"probs": {"damage": 0.0, "other": 1.0, "billing": 0.0}}}
    assert decide(it, no)[:2] == (False, "both_disagree")


def test_letter_distribution_pools_spaced_letters():
    from collections import namedtuple
    from math import log
    from dopest_model.blockb.readout import letter_distribution
    LP = namedtuple("LP", "logprob decoded_token")
    top = {1: LP(log(0.4), "A"), 2: LP(log(0.2), " A"), 3: LP(log(0.3), "B"), 4: LP(log(0.05), "D"), 5: LP(log(0.05), "x")}
    dist = letter_distribution(top, 3)                                          # D is not an option when k=3
    assert [round(p, 4) for p in dist] == [round(0.6 / 0.9, 4), round(0.3 / 0.9, 4), 0.0]
    assert letter_distribution({9: LP(-0.1, "Sure")}, 3) is None


def test_final_continuation():
    from dopest_model.blockb.readout import final_continuation
    END, TO_FINAL, FINAL = 7, [7, 6, 50, 5, 51, 8], [5, 51, 8]
    assert final_continuation([5, 60, 8, 101, 102, END], END, TO_FINAL, FINAL) == [5, 60, 8, 101, 102] + TO_FINAL
    assert final_continuation([5, 60, 8, 101, 102], END, TO_FINAL, FINAL) == [5, 60, 8, 101, 102] + TO_FINAL  # cut by budget
    assert final_continuation([5, 51, 8, 66], END, TO_FINAL, FINAL) == FINAL                                 # answered directly
    assert final_continuation([], END, TO_FINAL, FINAL) == FINAL


def test_shuffled_options_is_stable_and_spreads_the_answer():
    from dopest_model.blockb.writer import shuffled_options
    it = normalise(CHOICE, BRIEF)
    assert shuffled_options(it) == it                                   # idempotent: same order on a second pass
    assert sorted(it["labels"]) == ["billing", "damage", "other"] and it["labels"] == list(it["question"]["criteria"])
    firsts = {normalise(dict(CHOICE, state=CHOICE["state"] + f" Order {n}."), BRIEF)["labels"][0] for n in range(30)}
    assert firsts == {"billing", "damage", "other"}                     # the gold is not stuck in first place


def test_keep_rule_for_ordered_levels():
    it = normalise({"state": CHOICE["state"], "question": {"type": "score", "instructions": "How urgent?",
                    "criteria": ["low", "mid", "high", "critical"]}, "expected": 2}, BRIEF)
    near = {"qwen": {"probs": {"0": 0.0, "1": 0.5, "2": 0.5, "3": 0.0}},
            "gptoss": {"probs": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0}}}
    assert decide(it, near)[:2] == (True, "score_close")                        # mean level 2.25
    far = {"qwen": {"probs": {"0": 1.0, "1": 0.0, "2": 0.0, "3": 0.0}},
           "gptoss": {"probs": {"0": 0.0, "1": 1.0, "2": 0.0, "3": 0.0}}}
    assert decide(it, far)[:2] == (False, "score_far")                          # mean level 0.5


def test_plan_briefs_follows_language_shares_and_seed():
    import collections
    from dopest_model.blockb.specs import LANGUAGE_WEIGHTS
    from dopest_model.blockb.writer import parse_langs, plan_briefs
    mix = {"llm_router": 0.5, "usecase": 0.3, "fairness": 0.2}
    briefs = plan_briefs(3000, mix, parse_langs("od"), 0.2, seed=1)
    langs = collections.Counter(b["language"] for b, _ in briefs if b["language"] != "en")
    n, total = sum(langs.values()), sum(LANGUAGE_WEIGHTS.values())
    assert all(abs(langs[c] - w / total * n) <= 1 for c, w in LANGUAGE_WEIGHTS.items())   # exact shares, +-1
    assert all(b["family"] in ("llm_router", "usecase") for b, _ in briefs if b["language"] != "en")
    assert [p for _, p in plan_briefs(50, mix, parse_langs("pl,es"), 0.2, seed=1)] == \
           [p for _, p in plan_briefs(50, mix, parse_langs("pl,es"), 0.2, seed=1)]       # resumable: same seed, same briefs
    assert parse_langs("pl=2,es") == {"pl": 2.0, "es": 1.0}


def test_json_states_are_accepted_and_keyed_by_content():
    obj = {"conversation": [{"role": "user", "content": "hi, can you fix the failing SQL join in my report query?"}]}
    a = normalise(dict(CHOICE, state=obj), BRIEF)
    b = normalise(dict(CHOICE, state=dict(reversed(list(obj.items())))), BRIEF)
    assert a is not None and a["state"] == obj and a["id"] == b["id"]
    assert normalise(dict(CHOICE, state=["not", "an", "object"]), BRIEF) is None


def test_twins_keep_question_answer_group_and_split():
    from dopest_model.blockb.twins import plan_twins, twin_item
    parent = dict(normalise(CHOICE, BRIEF), group="g1", split="test", soft_probs={"damage": 1.0}, call=3)
    twin = twin_item(parent, "A customer writes that the parcel came smashed and wants a new one sent today.",
                     "perturbation", "wording")
    assert twin["id"] != parent["id"] and twin["provenance"]["twin_of"] == parent["id"]
    assert (twin["question"], twin["expected"], twin["labels"], twin["group"], twin["split"]) == \
           (parent["question"], parent["expected"], parent["labels"], "g1", "test")
    assert "soft_probs" not in twin and "call" not in twin and twin["label_source"] == "teacher"
    fair = dict(parent, base_family="fairness", id="f1")
    router = [dict(parent, base_family="llm_router", id=f"r{i}", state="same routed message, long enough to pass")
              for i in range(3)]
    plan = plan_twins([fair] + router, perturb_share=1.0, seed=0)
    assert sorted((k, len(ps)) for k, ps in plan) == [("counterfactual", 1), ("perturbation", 3)]  # one twin per state


def test_noul_answers_given_as_criteria_keys_are_accepted():
    base = {"state": CHOICE["state"], "question": {"type": "noul", "instructions": "Refund due?",
            "criteria": {"true": "t", "false": "f"}}}
    assert normalise(dict(base, expected="false"), BRIEF)["expected"] == "no"
    assert normalise(dict(base, expected=True), BRIEF)["expected"] == "yes"


def test_assemble_rewords_dedups_and_keeps_twins_with_parents():
    from dopest_model.blockb.assemble import assemble
    copied = "Does the request ask for code, a query or a script to be written, fixed or explained?"
    parent = dict(normalise(CHOICE, BRIEF), question=dict(CHOICE["question"], instructions=copied))
    twin = dict(parent, id="bb-twin", provenance=dict(parent["provenance"], twin_of=parent["id"]))
    orphan = dict(parent, id="bb-orphan", provenance=dict(parent["provenance"], twin_of="bb-gone"))
    out = assemble([parent, parent, twin, orphan])
    assert [it["id"] for it in out] == [parent["id"], "bb-twin"]
    assert all(it["question"]["instructions"].startswith("Is the user asking for code") for it in out)
