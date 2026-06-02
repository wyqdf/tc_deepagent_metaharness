"""Tests for LLM client wrappers."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

import sys
import types

from harness.llm import (
    LLMConfig,
    OpenAICompatibleClient,
    StubLLM,
    build_solver_client,
    load_solver_config,
    preflight_solver,
    _parse_harmony_response,
    _prepare_messages_for_reference,
)


def test_stub_llm_returns_configured_response_and_records_messages():
    llm = StubLLM('{"label": "A"}')
    messages = [{"role": "user", "content": "classify"}]

    assert llm.complete(messages) == '{"label": "A"}'
    assert llm.calls[0]["messages"] == messages


def test_stub_llm_can_choose_first_available_choice_from_prompt():
    llm = StubLLM()
    response = llm.complete([{"role": "user", "content": "Choices: Alpha, Beta"}])

    assert response == '{"label": "Alpha"}'


def test_load_solver_config_uses_env_names(monkeypatch):
    monkeypatch.setenv("SOLVER_API_KEY", "test-key")
    monkeypatch.setenv("SOLVER_BASE_URL", "https://example.test/v1")
    config = load_solver_config(
        {
            "solver": {
                "model": "gpt-oss-120b",
                "api_key_env": "SOLVER_API_KEY",
                "base_url_env": "SOLVER_BASE_URL",
                "temperature": 0.0,
                "max_tokens": 123,
                "timeout_ms": 456,
            }
        }
    )

    assert config == LLMConfig(
        model="gpt-oss-120b",
        base_url="https://example.test/v1",
        api_key="test-key",
        temperature=0.0,
        max_tokens=123,
        timeout_ms=456,
        max_retries=3,
        retry_sleep_seconds=2.0,
    )


def test_build_solver_client_can_return_stub():
    client = build_solver_client({"solver": {"stub": True, "stub_response": '{"label": "B"}'}})

    assert isinstance(client, StubLLM)
    assert client.complete([]) == '{"label": "B"}'


def test_openai_client_can_be_constructed_without_calling_api():
    client = OpenAICompatibleClient(LLMConfig("model", "https://example.test/v1", "key"))

    assert client.config.model == "model"


def test_openai_client_retries_transient_failures(monkeypatch):
    calls = {"count": 0}

    class FakeCompletions:
        def create(self, **request):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("temporary connection failure")
            message = types.SimpleNamespace(content="ok")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr("harness.llm.time.sleep", lambda seconds: None)
    client = OpenAICompatibleClient(
        LLMConfig(
            "plain-model",
            "https://example.test/v1",
            "key",
            max_retries=1,
            retry_sleep_seconds=0,
        )
    )

    assert client.complete([{"role": "user", "content": "hello"}]) == "ok"
    assert calls["count"] == 2


def test_preflight_solver_uses_stub_client():
    result = preflight_solver({"solver": {"stub": True, "stub_response": '{"final_answer": "ok"}'}})

    assert result["ok"] is True
    assert result["response"] == '{"final_answer": "ok"}'


def test_prepare_messages_for_reference_inserts_gpt_oss_reasoning_system_prompt():
    messages = [{"role": "user", "content": "hello"}]
    prepared = _prepare_messages_for_reference(messages, "gpt-oss-120b")

    assert prepared[0]["role"] == "system"
    assert prepared[0]["content"] == "Reasoning: medium"


def test_parse_harmony_response_falls_back_to_plain_openai_content():
    raw = '{"final_answer":"ok"}'

    assert _parse_harmony_response(raw) == raw
