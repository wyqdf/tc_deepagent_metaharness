"""Tests for official-like local agent protocol."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

import json

from harness.agent_protocol import extract_json_field, load_agent_memory, validate_agent_memory


def test_extract_json_field_accepts_final_answer_only_json():
    assert extract_json_field('{"reasoning": "r", "final_answer": "A"}', "final_answer") == "A"
    assert extract_json_field('```json\n{"final_answer": "B"}\n```', "final_answer") == "B"


def test_no_memory_agent_satisfies_protocol_and_tracks_prompt():
    memory = load_agent_memory("no_memory", llm=lambda prompt: '{"final_answer": "ok"}')

    validate_agent_memory(memory)
    pred, meta = memory.predict("question")

    assert pred == "ok"
    assert meta["full_response"] == '{"final_answer": "ok"}'
    assert memory.get_last_prompt_info()["prompt_len"] > 0
    memory.learn_from_batch([{"input": "x", "ground_truth": "y"}])
    assert memory.get_state() == "{}"


def test_fewshot_all_agent_state_and_raw_question_preference():
    prompts = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return '{"final_answer": "target"}'

    memory = load_agent_memory("fewshot_all", llm=llm)
    memory.learn_from_batch(
        [
            {"input": "wrapped prompt one", "raw_question": "raw one", "ground_truth": "A"},
            {"input": "wrapped prompt two", "raw_question": "raw two", "ground_truth": "B"},
        ]
    )
    state = memory.get_state()
    restored = load_agent_memory("fewshot_all", llm=llm)
    restored.set_state(state)

    pred, meta = restored.predict("eval input")

    assert pred == "target"
    assert meta["num_examples"] == 2
    assert json.loads(state)["examples"][0]["raw_question"] == "raw one"
    assert "Q: raw " in prompts[-1]
    assert "wrapped prompt one" not in prompts[-1]
