"""Tests for memory contracts and helpers."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

from harness.memory import (
    Example,
    Prediction,
    RetrievedExample,
    extract_label,
    format_choices,
    safe_json_loads,
    tokenize,
)


def test_memory_dataclasses_hold_expected_fields():
    example = Example("1", "Hello world", "A", ["A", "B"], {"dataset": "demo"})
    retrieved = RetrievedExample("r1", "Prior text", "B", 0.5)
    prediction = Prediction("A", '{"label": "A"}', [], [retrieved], {"ok": True})

    assert example.metadata["dataset"] == "demo"
    assert prediction.retrieved[0].label == "B"
    assert prediction.metadata["ok"] is True


def test_tokenize_lowercases_and_keeps_words():
    assert tokenize("Patent-Class 42, improved!") == ["patent", "class", "42", "improved"]


def test_format_choices_is_compact_and_ordered():
    assert format_choices(["A", "B", "C"]) == "A, B, C"


def test_safe_json_loads_handles_plain_and_fenced_json():
    assert safe_json_loads('{"label": "A"}') == {"label": "A"}
    assert safe_json_loads('```json\n{"label": "B"}\n```') == {"label": "B"}
    assert safe_json_loads("not json") is None


def test_extract_label_prefers_json_then_choice_mentions():
    choices = ["alpha", "beta"]
    assert extract_label('{"label": "alpha"}', choices) == "alpha"
    assert extract_label("The answer is beta.", choices) == "beta"
    assert extract_label("unknown", choices) == "unknown"


def test_extract_label_returns_json_answer_without_choices():
    assert extract_label('{"final_answer": "gastroesophageal reflux disease"}', []) == "gastroesophageal reflux disease"
