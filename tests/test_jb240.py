from dopest_model.jb240 import convert
from dopest_model.prompt import options

BASE = {"case_id": "customer_support-037", "domain": "customer_support", "state": "Subject: export failing", "options": None,
        "levels": None}


def test_noul_gold_maps_to_yes_no_and_unknown_is_dropped():
    item = convert({**BASE, "question_key": "feature_failing", "type": "noul", "gold": "true",
                    "instructions": "Is a feature failing?"})
    assert item["id"] == "jb240-customer_support-037-feature_failing" and item["expected"] == "yes"
    assert item["labels"] == ["no", "yes"] and item["question"]["criteria"] is None and item["tier"] == "standard"
    assert [o["id"] for o in options(item["question"])] == ["true", "false"]
    assert convert({**BASE, "question_key": "k", "type": "noul", "gold": "unknown", "instructions": "?"}) is None
    assert convert({**BASE, "question_key": "k", "type": "noul", "gold": "false", "instructions": "?"})["expected"] == "no"


def test_choice_and_score_keep_option_ids_and_level_indices():
    choice = convert({**BASE, "question_key": "team", "type": "choice", "gold": "billing", "instructions": "Which team?",
                      "options": [{"id": "engineering", "description": "Fixes bugs"}, {"id": "billing", "description": "Invoices"}]})
    assert choice["labels"] == ["engineering", "billing"] and choice["expected"] == "billing"
    assert choice["question"]["criteria"] == {"engineering": "Fixes bugs", "billing": "Invoices"}
    score = convert({**BASE, "question_key": "churn", "type": "score", "gold": 1, "instructions": "How likely to leave?",
                     "levels": ["No sign", "Some dissatisfaction", "Considering", "Decided"]})
    assert score["labels"] == ["0", "1", "2", "3"] and score["expected"] == "1"
    assert score["question"]["criteria"][1] == "Some dissatisfaction"
    assert convert({**BASE, "question_key": "x", "type": "choice", "gold": "nope", "instructions": "?",
                    "options": [{"id": "a", "description": "A"}]}) is None
