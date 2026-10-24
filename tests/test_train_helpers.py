import random

from dopest_model.prompt import messages
from dopest_model.data import shuffled, target

CHOICE = {"id": "c", "state": "s", "expected": "b", "target_probs": {"a": 0.2, "b": 0.7, "c": 0.1},
          "question": {"type": "choice", "instructions": "q", "criteria": {"a": "A", "b": "B", "c": "C"}}}
NOUL = {"id": "n", "state": "s", "expected": "no", "target_probs": {"yes": 0.3, "no": 0.7},
        "question": {"type": "noul", "instructions": "q", "criteria": {"true": "T", "false": "F"}}}


def test_shuffle_moves_options_but_keeps_the_gold_label():
    orders = set()
    for seed in range(30):
        item = shuffled(CHOICE, random.Random(seed))
        _, ids = messages(item)
        orders.add(tuple(ids))
        assert target(item, ids, soft=False)[ids.index("b")] == 1.0
    assert len(orders) > 3
    assert shuffled(NOUL, random.Random(0)) is NOUL          # yes/no keeps true-then-false, as at evaluation


def test_targets_map_yes_no_and_soft_probabilities():
    _, ids = messages(NOUL)
    assert ids == ["true", "false"]
    assert target(NOUL, ids, soft=False) == [0.0, 1.0]
    assert target(NOUL, ids, soft=True) == [0.3, 0.7]
    _, ids = messages(CHOICE)
    assert target(CHOICE, ids, soft=True) == [0.2, 0.7, 0.1]
