"""Tests for metric helpers."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

from harness.metrics import accuracy, compute_micro_f1, macro_average, make_result, normalize_label, per_dataset_scores


def test_normalize_label_matches_choices_case_insensitively():
    assert normalize_label(" alpha ", ["Alpha", "Beta"]) == "Alpha"
    assert normalize_label("The answer is Beta.", ["Alpha", "Beta"]) == "Beta"
    assert normalize_label("Gamma", ["Alpha", "Beta"]) == "Gamma"


def test_accuracy_uses_prediction_and_label_fields():
    rows = [
        {"prediction": "A", "label": "A"},
        {"prediction": "B", "label": "A"},
        {"prediction": "B", "label": "B"},
    ]
    assert accuracy(rows) == 2 / 3


def test_accuracy_prefers_official_was_correct_when_present():
    rows = [
        {"prediction": "wrong-text", "label": "A", "was_correct": True},
        {"prediction": "A", "label": "A", "was_correct": False},
    ]
    assert accuracy(rows) == 0.5


def test_per_dataset_scores_and_macro_average():
    rows = [
        {"dataset": "d1", "prediction": "A", "label": "A"},
        {"dataset": "d1", "prediction": "B", "label": "A"},
        {"dataset": "d2", "prediction": "C", "label": "C"},
    ]
    scores = per_dataset_scores(rows)
    assert scores == {"d1": 0.5, "d2": 1.0}
    assert macro_average(scores) == 0.75


def test_empty_inputs_score_zero():
    assert accuracy([]) == 0.0
    assert per_dataset_scores([]) == {}
    assert macro_average({}) == 0.0


def test_make_result_matches_reference_summary_shape():
    rows = [
        {"was_correct": True, "metrics": {"f1": 1.0, "tp": 1, "fp": 0, "fn": 0}},
        {"was_correct": False, "metrics": {"f1": 0.0, "tp": 0, "fp": 1, "fn": 1}},
    ]
    result = make_result(rows)

    assert result["accuracy"] == 0.5
    assert result["correct"] == 1
    assert result["total"] == 2
    assert result["avg_f1"] == 0.5
    assert compute_micro_f1(rows) == 0.5
