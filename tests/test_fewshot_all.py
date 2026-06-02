"""Tests for the fewshot_all system."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

from harness.memory import Example
from harness.llm import StubLLM
from systems.fewshot_all import build_memory_system


def test_fewshot_all_uses_training_examples_in_prompt():
    system = build_memory_system()
    system.learn(
        [
            Example("train-1", "alpha symptom", "A", ["A", "B"]),
            Example("train-2", "beta symptom", "B", ["A", "B"]),
        ],
        StubLLM(),
    )

    prediction = system.predict(Example("test-1", "alpha again", "A", ["A", "B"]), StubLLM('{"final_answer": "A"}'))

    prompt_text = "\n".join(message["content"] for message in prediction.prompt)
    assert prediction.label == "A"
    assert "Q: alpha symptom" in prompt_text
    assert "A: A" in prompt_text
    assert prediction.metadata["num_examples"] == 2
