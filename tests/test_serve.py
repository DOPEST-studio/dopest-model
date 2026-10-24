from dopest_model.prompt import LETTERS, options
from dopest_model.serve import answer_from, average_orders, read_in_groups


def fake_reader(favourite: str):
    """Reads any option list: the favourite gets 0.7, the rest share 0.3."""
    def read(opts):
        ids = [o["id"] for o in opts]
        if favourite in ids:
            p = [0.7 if i == favourite else 0.3 / (len(ids) - 1) for i in ids]
        else:
            p = [1 / len(ids)] * len(ids)
        return ids, p
    return read


def test_group_reading_keeps_every_option_and_finds_the_favourite():
    opts = [{"id": f"o{i}", "description": f"o{i}: option {i}"} for i in range(40)]
    ids, probs = read_in_groups(opts, fake_reader("o27"))
    assert ids == [o["id"] for o in opts] and abs(sum(probs) - 1) < 1e-9
    assert ids[probs.index(max(probs))] == "o27"
    ids, probs = read_in_groups(opts[:16], fake_reader("o3"))                  # 16 options: one round, untouched
    assert len(ids) == 16 and probs[3] == 0.7


def test_answers_follow_the_typesafe_shape():
    a = answer_from("noul", ["true", "false"], [0.6, 0.4], 0.801, {"true": "t", "false": "f"})
    assert a["noul"] == 0.801 and abs(a["probabilities"]["true"] + a["probabilities"]["false"] - 1) < 1e-9
    b = answer_from("choice", ["x", "y", "z"], [0.2, 0.5, 0.3], 0.801, {"x": "", "y": "", "z": ""})
    assert b["choice"] == "y" and b["confidence"] == 0.5 and set(b["probabilities"]) == {"x", "y", "z"}
    c = answer_from("score", ["0", "1", "2"], [0.1, 0.2, 0.7], None, ["low", "mid", "high"])
    assert abs(c["score"] - 1.6) < 1e-9 and c["legend"]["2"] == "high" and c["confidence"] == 0.7


def test_hidden_option_keys_drop_the_key_prefix():
    q = {"type": "choice", "criteria": {"billing": "About invoices", "other": "Anything else"}}
    assert options(q)[0]["description"] == "billing: About invoices"
    assert options(q, show_keys=False)[0]["description"] == "About invoices"
    assert options({"type": "noul", "criteria": None})[0]["description"].startswith("true: ")
    assert len(LETTERS) == 16


def test_group_reading_handles_several_rounds():
    opts = [{"id": f"o{i}", "description": f"o{i}"} for i in range(200)]          # 13 groups -> 26 finalists -> 2 rounds
    ids, probs = read_in_groups(opts, fake_reader("o150"))
    assert len(ids) == 200 and ids[probs.index(max(probs))] == "o150" and abs(sum(probs) - 1) < 1e-9


def test_average_orders_is_order_independent():
    a = (["x", "y", "z"], [0.6, 0.3, 0.1])
    b = (["z", "y", "x"], [0.2, 0.3, 0.5])
    ids, probs = average_orders(a, b)
    assert ids == ["x", "y", "z"] and [round(p, 3) for p in probs] == [0.55, 0.3, 0.15]
