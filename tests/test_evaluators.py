"""Tests for official-style evaluator compatibility."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

from harness.evaluators import extract_final_answer, get_evaluator, unpack_eval_result


def test_extract_final_answer_matches_reference_json_field_contract():
    assert extract_final_answer('{"final_answer": "A"}') == "A"
    assert extract_final_answer('```json\n{"final_answer": "C"}\n```') == "C"
    assert extract_final_answer('{"label": "B"}') == '{"label": "B"}'


def test_symptom2disease_evaluator_matches_reference_normalization():
    evaluator = get_evaluator("Symptom2Disease")

    assert evaluator("[DIAGNOSIS]Common Cold.[/DIAGNOSIS]", "common cold") is True
    assert evaluator("diagnosis: migraine", "migraine") is True


def test_uspto_evaluator_uses_set_equality_and_jaccard():
    evaluator = get_evaluator("USPTO")
    ok, metrics = unpack_eval_result(evaluator('{"final_answer": "A.B"}', "B.A"))

    assert ok is True
    assert metrics["jaccard_similarity"] == 1.0


def test_lawbench_evaluator_returns_f1_metrics():
    evaluator = get_evaluator("LawBench")
    ok, metrics = unpack_eval_result(evaluator("[罪名]盗窃、诈骗<eoa>", "盗窃、诈骗"))

    assert ok is True
    assert metrics["f1"] == 1.0
    assert metrics["tp"] == 2
