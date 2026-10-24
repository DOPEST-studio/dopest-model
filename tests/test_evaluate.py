import json

from dopest_model.calibrate import commit_floor
from dopest_model.evaluate import (fit_calibration, load_calibration, official_outcome, outcomes,
                                   paired_bootstrap, predictions, save_calibration)
from dopest_model import jevscore

NOUL = {"id": "n1", "family": "f", "tier": "hard", "state": "s", "labels": ["yes", "no"], "expected": "yes",
        "question": {"type": "noul", "instructions": "q", "criteria": {"true": "t", "false": "f"}}}
SCORE = {"id": "s1", "family": "f", "tier": "judge", "state": "s", "labels": ["0", "1", "2", "3"], "expected": 3,
         "question": {"type": "score", "instructions": "q", "criteria": ["a", "b", "c", "d"]}}
CHOICE = {"id": "c1", "family": "f", "tier": "easy", "state": "s", "labels": ["a", "b", "c"], "expected": "b",
          "question": {"type": "choice", "instructions": "q", "criteria": {"a": "A", "b": "B", "c": "C"}}}


def test_commit_floor_moves_undecided_answers_to_their_side():
    assert commit_floor(0.6) == 0.801 and commit_floor(0.3) == 1 - 0.801
    assert commit_floor(0.9) == 0.9 and commit_floor(0.1) == 0.1


def test_official_rules_abstention_and_uniform_score_chance():
    undecided = {"yes": 0.6, "no": 0.4}
    assert jevscore.item_outcome(NOUL, undecided)["correct"] is True
    assert official_outcome(NOUL, undecided)["correct"] is False          # 0.2 < P(yes) < 0.8: abstention
    assert official_outcome(NOUL, {"yes": 0.85, "no": 0.15})["correct"] is True
    o = official_outcome(SCORE, {"0": 0.1, "1": 0.1, "2": 0.2, "3": 0.6})
    assert o["nmae_chance_mid"] == o["nmae_chance"]                      # uniform random-level baseline
    assert jevscore.item_outcome(SCORE, {"0": 0.1, "1": 0.1, "2": 0.2, "3": 0.6})["nmae_chance_mid"] == 0.5


def test_calibration_artifact_roundtrip_and_floor_in_predictions(tmp_path):
    rows = [{"id": "n1", "type": "noul", "ids": ["true", "false"], "logits": [0.4, 0.0]},
            {"id": "c1", "type": "choice", "ids": ["a", "b", "c"], "logits": [0.0, 3.0, 0.0]},
            {"id": "s1", "type": "score", "ids": ["0", "1", "2", "3"], "logits": [0.0, 0.0, 1.0, 3.0]}]
    tasks = {"n1": NOUL, "c1": CHOICE, "s1": SCORE}
    calib = fit_calibration(rows, tasks, source="unit")
    save_calibration(calib, tmp_path / "c.json")
    loaded = load_calibration(tmp_path / "c.json")
    assert loaded["noul_floor"] == 0.801 and set(loaded["temperatures"]) == {"choice", "noul", "score"}
    p = {x["id"]: x["probs"] for x in predictions(rows, loaded)}
    assert p["n1"]["yes"] == 0.801 and abs(p["n1"]["yes"] + p["n1"]["no"] - 1) < 1e-9   # floored, still a distribution
    no_floor = {x["id"]: x["probs"] for x in predictions(rows, {"choice": 1.0, "noul": 1.0, "score": 1.0})}
    assert 0.5 < no_floor["n1"]["yes"] < 0.801                                           # legacy temps: no floor
    outs = outcomes(rows, [NOUL, CHOICE, SCORE], loaded)
    assert [o["correct"] for o in outs] == [True, True, True]


def test_paired_bootstrap_identical_runs_give_zero_difference():
    tasks = [dict(CHOICE, id=f"c{i}", tier=t) for i, t in enumerate(["easy", "hard"] * 10)]
    rows = [{"id": t["id"], "type": "choice", "ids": ["a", "b", "c"], "logits": [0.0, 2.0 if i % 3 else -2.0, 0.0]}
            for i, t in enumerate(tasks)]
    calib = {"temperatures": {"choice": 1.0, "noul": 1.0, "score": 1.0}, "noul_floor": 0.801}
    outs = outcomes(rows, tasks, calib)
    bs = paired_bootstrap({"a": outs, "b": list(outs)}, n_boot=50)
    assert bs["b"] == (0.0, 0.0, 0.0) and bs["_point"]["a"] == bs["_point"]["b"]
