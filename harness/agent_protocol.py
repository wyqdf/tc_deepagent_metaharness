# agent_protocol.py
# 定义候选 agent 必须遵守的记忆系统协议，并提供动态加载、
# 接口校验、LLM 调用适配和 JSON 答案抽取等通用工具。

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


LLMCallable = Callable[[str], str]


# 候选 memory agent 必须实现的最小协议。
@runtime_checkable
class AgentMemorySystem(Protocol):
    # 对单条输入进行预测，并返回答案和附加信息。
    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    # 从一批带标签训练样本或反馈结果中更新 memory。
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    # 将当前 memory 状态序列化成字符串，便于保存到文件。
    def get_state(self) -> str:
        raise NotImplementedError

    # 从字符串状态恢复 memory。
    def set_state(self, state: str) -> None:
        raise NotImplementedError


# 候选 agent 的基础父类，提供 LLM 调用记录和通用接口占位。
class BaseAgentMemory:
    # 保存 LLM 回调，并为每个线程维护最近一次 prompt 信息。
    def __init__(self, llm: LLMCallable):
        self._llm = llm
        self._prompt_local = threading.local()

    # 调用底层 LLM，同时记录最近一次 prompt 的长度、哈希和原文。
    def call_llm(self, prompt: str) -> str:
        self._prompt_local.last_prompt_len = len(prompt)
        self._prompt_local.last_prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        self._prompt_local.last_prompt_text = prompt
        return self._llm(prompt)

    # 返回最近一次 LLM 调用的 prompt 统计信息，用于 trace 和日志。
    def get_last_prompt_info(self) -> dict[str, Any]:
        return {
            "prompt_len": getattr(self._prompt_local, "last_prompt_len", None),
            "prompt_hash": getattr(self._prompt_local, "last_prompt_hash", None),
            "prompt_text": getattr(self._prompt_local, "last_prompt_text", None),
        }

    # 默认用序列化状态长度近似衡量 memory 上下文大小。
    def get_context_length(self) -> int:
        return len(self.get_state())

    # 使用类名作为 agent 名称。
    @property
    def name(self) -> str:
        return self.__class__.__name__

    # 子类必须实现具体预测逻辑。
    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    # 子类必须实现 memory 更新逻辑。
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    # 子类必须实现状态导出逻辑。
    def get_state(self) -> str:
        raise NotImplementedError

    # 子类必须实现状态恢复逻辑。
    def set_state(self, state: str) -> None:
        raise NotImplementedError


# 从模型输出中提取指定 JSON 字段，兼容纯 JSON、代码块 JSON 和嵌入式 JSON。
def extract_json_field(text: str, field: str, default: str = "") -> str:
    # 先尝试把整段输出当成 JSON。
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return str(data.get(field, default))

    # 再尝试解析 Markdown fenced JSON 代码块。
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return str(data.get(field, default))

    # 再从自由文本中找第一个平衡 JSON 对象。
    candidate = _first_json_object(text)
    if candidate:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return str(data.get(field, default))

    # 最后兜底：用正则提取最后一个 "field": "value"。
    matches = re.findall(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', text)
    return matches[-1] if matches else default


# 动态导入候选 agent 文件，并实例化其中第一个 BaseAgentMemory 子类。
def load_agent_memory(path_or_name: str | Path, llm: LLMCallable, project_root: str | Path | None = None) -> Any:
    path = resolve_agent_path(path_or_name, project_root=project_root)
    module_name = f"_tc_agent_{path.stem}_{hashlib.md5(str(path).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import agent module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # 找到候选文件中真正的 memory 子类，并进行接口校验。
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is BaseAgentMemory:
            continue
        if issubclass(obj, BaseAgentMemory):
            memory = obj(llm=llm)
            validate_agent_memory(memory)
            return memory
    raise ValueError(f"No BaseAgentMemory subclass found in {path}")


# 检查候选 memory 是否实现 predict、learn_from_batch、get_state、set_state。
def validate_agent_memory(memory: Any) -> Any:
    missing = [
        name
        for name in ("predict", "learn_from_batch", "get_state", "set_state")
        if not callable(getattr(memory, name, None))
    ]
    if missing:
        raise TypeError(f"Agent memory missing required methods: {', '.join(missing)}")
    return memory


# 将 agent 名称、相对路径或绝对路径解析为实际存在的 .py 文件。
def resolve_agent_path(path_or_name: str | Path, project_root: str | Path | None = None) -> Path:
    raw = Path(path_or_name)
    root = Path(project_root) if project_root is not None else Path.cwd()
    candidates: list[Path]
    if raw.suffix == ".py":
        candidates = [raw, root / raw]
    elif raw.parts and raw.parts[0] == "agents":
        candidates = [raw.with_suffix(".py"), root / raw.with_suffix(".py")]
    else:
        candidates = [root / "agents" / f"{raw.name}.py"]

    # 按候选路径顺序查找第一个真实存在的文件。
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Agent path not found: {path_or_name}")


# 把 solver 客户端适配成简单的 Callable[[str], str] 形式。
def llm_client_to_callable(llm: Any) -> LLMCallable:
    def _call(prompt: str) -> str:
        if callable(llm) and not hasattr(llm, "complete"):
            return str(llm(prompt))
        return str(llm.complete([{"role": "user", "content": prompt}]))

    return _call


# 在自由文本中寻找第一个括号平衡的 JSON 对象字符串。
def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False

        # 从每个左花括号开始扫描，维护字符串状态和括号深度。
        for pos in range(start, len(text)):
            char = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return re.sub(r",\s*([\]}])", r"\1", text[start : pos + 1])
        start = text.find("{", start + 1)
    return None
