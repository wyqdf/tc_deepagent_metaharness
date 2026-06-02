"""Tests for prompt builders."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

from harness.memory import Example, Prediction, RetrievedExample
from harness.prompts import build_classifier_prompt, build_proposer_prompt, build_verification_prompt


def test_classifier_prompt_contains_text_choices_and_retrieved_examples():
    example = Example("q", "query text", "A", ["A", "B"])
    retrieved = [RetrievedExample("r", "prior text", "B", 1.0)]

    prompt = build_classifier_prompt(example, retrieved)
    content = prompt[-1]["content"]

    assert "query text" in content
    assert "Choices: A, B" in content
    assert "prior text" in content
    assert 'Return JSON: {"final_answer": "..."}' in content


def test_classifier_prompt_omits_choices_when_not_provided():
    example = Example("q", "query text", "A", [])
    prompt = build_classifier_prompt(example, [])

    assert "Choices:" not in prompt[-1]["content"]
    assert "Task prompt:" in prompt[-1]["content"]


def test_verification_prompt_contains_first_prediction():
    example = Example("q", "query text", "A", ["A", "B"])
    first = Prediction("B", '{"label": "B"}', [])

    prompt = build_verification_prompt(example, first)

    assert "First prediction: B" in prompt[-1]["content"]
    assert "query text" in prompt[-1]["content"]


def test_proposer_prompt_contains_contract_and_memory_axes():
    prompt = build_proposer_prompt(
        frontier={"best": "baseline"},
        leaderboard=[{"name": "baseline", "average": 0.5}],
        recent_traces=[{"id": "x1", "prediction": "A", "label": "B"}],
        candidate_contract={"required_protocol": "BaseAgentMemory subclass"},
        source_files={"agents/no_memory.py": "class NoMemory(BaseAgentMemory):\n    pass\n"},
        dataset_specs=[{"name": "USPTO", "train_path": "/data/train.jsonl"}],
        result_summaries=[{"name": "baseline", "per_dataset": {"USPTO": 0.1}}],
        artifact_paths=["runs/demo/baseline_scores.json"],
    )

    assert "BaseAgentMemory subclass" in prompt
    assert "what training examples or experience to store" in prompt
    assert "agents/" in prompt
    assert "State file paths to inspect:" in prompt
    assert "Result file index:" in prompt
    assert "Source file index:" in prompt
    assert "- agents/no_memory.py" in prompt
    assert "class NoMemory" not in prompt
    assert "\n    pass" not in prompt
    assert "runs/demo/baseline_scores.json" in prompt
    assert "evolution_summary.jsonl" in prompt
    assert "logs/<dataset>/<agent>/<model>/log.jsonl" in prompt
    assert "logs/<dataset>/<memory>/<model>/log.jsonl" in prompt


def test_proposer_prompt_uses_hard_workflow_language():
    prompt = build_proposer_prompt(
        frontier={},
        leaderboard=[],
        recent_traces=[],
        candidate_contract={"required_protocol": "BaseAgentMemory subclass"},
        source_files={},
        dataset_specs=[],
        result_summaries=[],
        artifact_paths=[],
    )

    assert "WORKFLOW" in prompt
    assert "Step 1: Analyze" in prompt
    assert "MUST read all state files" in prompt
    assert "Step 2: Prototype — MANDATORY" in prompt
    assert "Do not run the full benchmark" in prompt
