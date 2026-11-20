from diabetes_agents import rules


def test_classify():
    assert rules.classify("he is unconscious")[0] == "emergency"
    assert rules.classify("I want to die")[0] == "self_harm"
    assert rules.classify("how many units should I take")[0] == "dosing"
    assert rules.classify("what foods are good for diabetes") is None


def test_draft_violations():
    assert rules.draft_violations("Take 10 units of insulin")
    assert not rules.draft_violations("Eat 15 grams of fast-acting carbohydrate")
