import math

import pytest

from dopest_model.prompt import options
from dopest_model.reason import Budget, Policy, Reasoned, blend, reason_messages, split_reasoning, unsure
from dopest_model.serve import Decider


def test_reason_prompt_lists_the_options_as_letters_and_asks_for_the_marker():
    q = {"type": "choice", "instructions": "Route it.", "criteria": {"billing": "Charges", "other": "Else"}}
    msgs = reason_messages("Refund of the March invoice.", q["instructions"], options(q), words=60)
    assert "at most 60 words that names" in msgs[0]["content"] and '"Answer:"' in msgs[0]["content"]
    assert '"letter": "A"' in msgs[1]["content"] and '"letter": "B"' in msgs[1]["content"] and "March invoice" in msgs[1]["content"]
    short = reason_messages("s", "c", options(q), words=40, sentences=2)[0]["content"]
    assert "at most 40 words and 2 sentences that names" in short


def test_split_reasoning_finds_the_marker():
    assert split_reasoning("The invoice is from March, so billing.\nAnswer: A") == ("The invoice is from March, so billing.", True)
    assert split_reasoning("Ran out of budget") == ("Ran out of budget", False)
    assert split_reasoning("  \nAnswer:") == ("", True)


def test_blend_and_gate():
    mixed = blend([0.6, 0.4], [0.0, 0.0], weight=0.75, temperature=3.0)        # flat second pass: pulls towards 1/2
    assert mixed[0] == pytest.approx(0.25 * 0.6 + 0.75 * 0.5) and sum(mixed) == pytest.approx(1.0)
    assert blend([0.6, 0.4], [1.0, 5.0], weight=1.0, temperature=1.0)[1] > 0.98   # weight 1: the second pass alone
    assert unsure([0.6, 0.4], 0.7) and not unsure([0.6, 0.4], 0.5)


def test_policy_reads_the_calibration_section_and_prices_a_pass():
    policy = Policy.from_calibration({"temperatures": {}, "reasoning": {"gate": 0.7, "unknown": 1}}, budget_usd_per_1000=None)
    assert policy.gate == 0.7 and policy.weight == 0.5 and policy.budget_usd_per_1000 == 0.014 and policy.types == ("choice", "score")
    assert Policy.from_calibration({"reasoning": {"types": ["choice"]}}).types == ("choice",)
    typed = Policy.from_calibration({"reasoning": {"gate": 0.8, "gates": {"score": 0.7}}})
    assert typed.gate_for("score") == 0.7 and typed.gate_for("choice") == 0.8 and Policy().gate_for("score") == 0.8
    assert Policy.from_calibration({}, budget_usd_per_1000=0.02).budget_usd_per_1000 == 0.02
    assert policy.cost(2500, 96) == pytest.approx((2500 * 0.09 + 96 * 0.34) / 1e6)


def test_budget_keeps_the_pooled_cost_under_the_per_decision_amount():
    budget = Budget(usd_per_decision=1.0)
    pass_cost = 10.0
    reasoned = 0
    for _ in range(100):
        if budget.allows(pass_cost):
            budget.charge(pass_cost)
            reasoned += 1
        budget.tick()
        assert budget.spent <= budget.decisions * budget.per_decision
    assert reasoned == 10 and not Budget(1.0).allows(10.0)                      # nothing is advanced to the first decisions


class FakeReasoner:
    """Stands in for reason.Reasoner: fixed prompt length, fixed letter logits."""

    marker_ids = [1, 2, 3]

    def __init__(self, prompt_len: int, logits: list[float]):
        self.prompt_len, self.logits, self.calls = prompt_len, logits, 0

    def prompt_tokens(self, state, instructions, opts, words, sentences=None):
        return list(range(self.prompt_len))

    def max_prompt_tokens(self, max_new_tokens):
        return 8192 - max_new_tokens - 4

    def estimate(self, prompt_tokens, new_tokens):
        return prompt_tokens / 2000 + new_tokens * 0.08 + 0.4

    def __call__(self, tokens, n_options, max_new_tokens, max_seconds=float("inf")):
        self.calls += 1
        self.last_max_seconds = max_seconds
        return Reasoned(self.logits[:n_options], "because", True, 40, 0, 0.5)


def decider_with(reasoner, policy: Policy) -> Decider:
    d = Decider.__new__(Decider)
    d.policy, d.reasoner, d.budget, d.reasoned = policy, reasoner, Budget(policy.budget_usd_per_1000 / 1000), 0
    d.skipped = {"length": 0, "budget": 0, "time": 0, "error": 0}
    d.last_input_tokens = d.last_output_tokens = 0
    return d


def test_decider_mixes_and_bills_a_second_pass_only_within_prompt_limit_budget_and_time():
    import time
    opts = [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}]
    reasoner = FakeReasoner(prompt_len=1000, logits=[0.0, 10.0])
    d = decider_with(reasoner, Policy(budget_usd_per_1000=1000.0))               # a budget that never binds
    probs = d.reason("state", "pick", opts, [0.6, 0.4], started=time.perf_counter())
    p2_b = 1 / (1 + math.exp(-10 / 3))                                              # softmax of the logits at T2 = 3
    assert probs[1] == pytest.approx(0.5 * 0.4 + 0.5 * p2_b) and reasoner.calls == 1
    assert d.last_input_tokens == 1000 and d.last_output_tokens == 40 and d.reasoned == 1
    assert d.budget.spent == pytest.approx((1000 * 0.09 + 40 * 0.34) / 1e6)
    assert 6.0 < reasoner.last_max_seconds <= 6.5                                   # the rest of the decision's budget

    d = decider_with(FakeReasoner(prompt_len=1000, logits=[0.0, 10.0]), Policy(max_prompt_tokens=500, budget_usd_per_1000=1000.0))
    assert d.reason("state", "pick", opts, [0.6, 0.4], time.perf_counter()) == [0.6, 0.4] and d.skipped["length"] == 1

    d = decider_with(FakeReasoner(prompt_len=1000, logits=[0.0, 10.0]), Policy(budget_usd_per_1000=0.014))
    assert d.reason("state", "pick", opts, [0.6, 0.4], time.perf_counter()) == [0.6, 0.4] and d.skipped["budget"] == 1
    d.budget.decisions = 100                                                         # accrued: reasoned
    assert d.reason("state", "pick", opts, [0.6, 0.4], time.perf_counter()) != [0.6, 0.4]

    d = decider_with(FakeReasoner(prompt_len=1000, logits=[0.0, 10.0]), Policy(budget_usd_per_1000=1000.0))
    late = time.perf_counter() - 5.0                                                 # the one-pass readout took 5 s already
    assert d.reason("state", "pick", opts, [0.6, 0.4], late) == [0.6, 0.4] and d.skipped["time"] == 1
