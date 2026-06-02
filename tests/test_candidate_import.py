"""Tests for candidate memory-system imports."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

from harness.eval import import_memory_system
from harness.llm import StubLLM
from harness.memory import Example, Prediction
from systems.baseline import LexicalFewShotMemory, build_memory_system


def test_build_baseline_memory_system():
    system = build_memory_system({"k": 2})

    assert isinstance(system, LexicalFewShotMemory)
    assert system.k == 2


def test_baseline_learn_and_predict_with_stub_llm():
    system = build_memory_system({"k": 1})
    llm = StubLLM('{"label": "B"}')
    system.learn(
        [
            Example("1", "alpha patent", "A", ["A", "B"]),
            Example("2", "beta disease", "B", ["A", "B"]),
        ],
        llm,
    )

    prediction = system.predict(Example("q", "beta symptom", "A", ["A", "B"]), llm)

    assert isinstance(prediction, Prediction)
    assert prediction.label == "B"
    assert prediction.retrieved[0].id == "2"
    assert "Choices: A, B" in prediction.prompt[-1]["content"]


def test_import_memory_system_from_path():
    system = import_memory_system("systems/baseline.py", config={"k": 1})

    assert system.name == "lexical_fewshot"
