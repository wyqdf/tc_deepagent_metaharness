# LLM 客户端封装模块。
# 负责读取 solver 配置、创建测试 Stub 或 OpenAI-compatible 客户端，
# 并统一处理请求重试、prompt 截断和 GPT-OSS harmony 响应解析。

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

logger = logging.getLogger(__name__)
MAX_PROMPT_CHARS = 224_000


# solver LLM 的连接参数和生成参数。
@dataclass
class LLMConfig:

    model: str
    base_url: str
    api_key: str
    temperature: float | None = 0.0
    max_tokens: int = 512
    timeout_ms: int = 300000
    max_retries: int = 3
    retry_sleep_seconds: float = 2.0


# 真实 LLM 客户端和测试 Stub 共同遵守的最小接口。
class LLMClient(Protocol):

    # 发送 chat messages 并返回模型补全文本。
    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        raise NotImplementedError


# 测试和 dry-run 使用的确定性假 LLM。
class StubLLM:

    # 保存可选固定回复，并初始化调用历史。
    def __init__(self, response: str | None = None):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    # 记录请求，并返回固定回复或从 prompt 中推导出的玩具答案。
    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.response is not None:
            return self.response
        prompt = "\n".join(message.get("content", "") for message in messages)
        choice = _first_choice_from_prompt(prompt)
        return f'{{"label": "{choice}"}}'
  

# OpenAI-compatible chat completions API 的轻量封装。
class OpenAICompatibleClient:

    # 保存 solver 配置，供后续 API 调用使用。
    def __init__(self, config: LLMConfig):
        self.config = config

    # 发送 chat completion 请求，处理重试、超时和 GPT-OSS 输出清理。
    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        from openai import OpenAI

        # 总尝试次数 = 第一次请求 + 配置的重试次数。
        attempts = max(1, int(kwargs.get("max_retries", self.config.max_retries)) + 1)
        sleep_seconds = float(kwargs.get("retry_sleep_seconds", self.config.retry_sleep_seconds))
        last_exc: Exception | None = None
        # 每次失败会记录异常；未达到上限时等待后重试。
        for attempt in range(1, attempts + 1):
            try:
                # 先整理并截断消息，避免超长 prompt 或 GPT-OSS 缺少 system 消息。
                call_messages = _prepare_messages_for_reference(messages, self.config.model)
                # 每次请求创建 OpenAI-compatible 客户端，使用配置中的 base_url 和 api_key。
                client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout_ms / 1000,
                )
                # 组装 chat completions 请求体，允许 kwargs 覆盖模型和生成参数。
                request: dict[str, Any] = {
                    "model": kwargs.get("model", self.config.model),
                    "messages": call_messages,
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                }
                # temperature 为 None 时不写入请求，兼容不支持 temperature 的接口。
                temperature = kwargs.get("temperature", self.config.temperature)
                if temperature is not None:
                    request["temperature"] = temperature
                response = client.chat.completions.create(**request)
                content = response.choices[0].message.content or ""
                # GPT-OSS 可能返回 harmony 编码，需要额外抽取 final channel。
                if _is_gpt_oss(str(request["model"])):
                    return _parse_harmony_response(content)
                return content
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts:
                    raise
                # 记录失败尝试，便于排查接口、限流或网络问题。
                logger.warning(
                    "LLM call failed on attempt %s/%s: %s",
                    attempt,
                    attempts,
                    exc,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        raise RuntimeError("LLM call failed") from last_exc


# 合并配置文件和环境变量覆盖项，生成 LLMConfig。
def load_solver_config(config: Mapping[str, Any]) -> LLMConfig:
    # solver 字段提供默认值，环境变量可覆盖敏感连接信息。
    solver = dict(config.get("solver", {}))
    api_key_env = solver.get("api_key_env", "SOLVER_API_KEY")
    base_url_env = solver.get("base_url_env", "SOLVER_BASE_URL")
    return LLMConfig(
        model=str(solver.get("model", "gpt-oss-120b")),
        # 优先使用环境变量中的 base_url；否则使用配置文件值。
        base_url=os.environ.get(
            base_url_env,
            str(solver.get("api_base") or solver.get("base_url") or solver.get("default_base_url", "")),
        ),
        # 优先使用环境变量中的 api_key，避免把密钥写死在配置里。
        api_key=os.environ.get(api_key_env, str(solver.get("api_key", ""))),
        temperature=solver.get("temperature", 0.0),
        max_tokens=int(solver.get("max_tokens", 512)),
        timeout_ms=int(solver.get("timeout_ms", 300000)),
        max_retries=int(solver.get("max_retries", 3)),
        retry_sleep_seconds=float(solver.get("retry_sleep_seconds", 2.0)),
    )


# 根据配置创建 Stub solver 或真实 OpenAI-compatible 客户端。
def build_solver_client(config: Mapping[str, Any]) -> LLMClient:
    solver = dict(config.get("solver", {}))
    # stub 模式用于快速测试流程，不调用真实模型接口。
    if solver.get("stub", False):
        return StubLLM(solver.get("stub_response"))
    return OpenAICompatibleClient(load_solver_config(config))


# 对配置好的 solver 做一次轻量连通性检查。
def preflight_solver(config: Mapping[str, Any]) -> dict[str, Any]:
    solver = dict(config.get("solver", {}))
    # 构造 solver 后发一个小请求，提前暴露连通性和鉴权问题。
    client = build_solver_client(config)
    try:
        response = client.complete(
            [
                {"role": "system", "content": "You are a connectivity checker."},
                {"role": "user", "content": 'Return exactly {"final_answer":"ok"}.'},
            ],
            max_tokens=int(solver.get("preflight_max_tokens", 128)),
            temperature=0.0,
        )
    # preflight 失败不抛异常，而是返回结构化错误信息。
    except Exception as exc:
        return {
            "ok": False,
            "model": solver.get("model", "unknown"),
            "base_url": solver.get("api_base") or solver.get("base_url") or solver.get("default_base_url"),
            "error": str(exc),
        }
    return {
        "ok": True,
        "model": solver.get("model", "unknown"),
        "base_url": solver.get("api_base") or solver.get("base_url") or solver.get("default_base_url"),
        "response": response,
    }


# 从测试 prompt 的 Choices 行中取第一个候选项。
def _first_choice_from_prompt(prompt: str) -> str:
    marker = "Choices:"
    if marker not in prompt:
        return "unknown"
    tail = prompt.split(marker, 1)[1].strip().splitlines()[0]
    first = tail.split(",", 1)[0].strip()
    return first or "unknown"


# 整理 chat messages，使其兼容 reference solver 请求格式。
def _prepare_messages_for_reference(
    messages: list[dict[str, str]],
    model: str,
) -> list[dict[str, str]]:
    # 对每条消息内容执行截断，保护上游 API 的上下文长度。
    call_messages = [{"role": msg["role"], "content": _truncate(str(msg.get("content", "")))} for msg in messages]
    # GPT-OSS 没有 system 消息时补充 reasoning 设置。
    if _is_gpt_oss(model) and not any(msg.get("role") == "system" for msg in call_messages):
        call_messages.insert(0, {"role": "system", "content": "Reasoning: medium"})
    return call_messages


# 判断模型名是否属于 GPT-OSS 系列。
def _is_gpt_oss(model: str) -> bool:
    return "gpt-oss" in model.lower()


# 截断超长 prompt，同时保留开头和结尾上下文。
def _truncate(prompt: str) -> str:
    # 未超过限制时直接返回原 prompt。
    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt
    # 超长时保留头尾，兼顾任务说明和最近上下文。
    half = MAX_PROMPT_CHARS // 2
    truncated = prompt[:half] + "\n\n... [TRUNCATED] ...\n\n" + prompt[-half:]
    logger.warning("Prompt truncated: %d -> %d chars", len(prompt), len(truncated))
    return truncated


# 尽量从 GPT-OSS harmony 格式内容中解析最终回复。
def _parse_harmony_response(raw_content: str) -> str:
    try:
        # 优先使用 openai_harmony 官方解析器处理 harmony 格式。
        from openai_harmony import HarmonyEncodingName, Role, load_harmony_encoding

        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        tokens = enc.encode(raw_content, allowed_special="all")
        parsed = enc.parse_messages_from_completion_tokens(
            tokens,
            role=Role.ASSISTANT,
            strict=False,
        )
        # 优先返回 assistant 的 final channel。
        for msg in parsed:
            if msg.channel == "final":
                text = "".join(c.text for c in msg.content if hasattr(c, "text"))
                if text.strip():
                    return text
        if parsed:
            text = "".join(c.text for c in parsed[-1].content if hasattr(c, "text"))
            if text.strip():
                return text
    # 解析失败时退回原始内容，保证调用链不中断。
    except Exception:
        pass
    return raw_content
