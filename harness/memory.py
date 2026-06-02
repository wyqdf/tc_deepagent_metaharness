# Meta-Harness 记忆数据结构与通用解析工具。
#
# 本模块定义文本分类实验中通用的 Example、RetrievedExample、Prediction 等数据结构，
# 并提供标签抽取、JSON 解析、tokenize、候选标签格式化等基础工具。
# 这些类型主要服务于旧版 compact memory system、prompt 构造和 baseline agent。

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


# LLM 输入样本的统一结构。
@dataclass
class Example:
    # 样本 ID，通常来自 JSONL 中的 id；没有时由加载器自动生成。
    id: str
    # 模型实际看到的文本，可能是原始文本，也可能是 data.py 包装后的完整 prompt。
    text: str
    # 标准答案标签或目标答案。
    label: str
    # 可选候选标签列表；生成式任务通常为空列表。
    choices: list[str]
    # 附加信息，例如原始问题、数据集名、来源路径等。
    metadata: dict[str, Any] = field(default_factory=dict)


# 一次预测时从 memory 中检索到的训练样本。
@dataclass
class RetrievedExample:
    # 被检索样本的 ID。
    id: str
    # 被检索样本的文本。
    text: str
    # 被检索样本的标准答案。
    label: str
    # 检索分数，数值含义由具体系统决定。
    score: float
    # 检索项的额外信息。
    metadata: dict[str, Any] = field(default_factory=dict)


# 一次预测的结构化结果。
@dataclass
class Prediction:
    # 系统抽取出的最终预测标签或答案。
    label: str
    # LLM 原始输出，方便调试和复核。
    raw_output: str
    # 实际发给 LLM 的 chat messages。
    prompt: list[dict[str, str]]
    # 本次预测使用到的检索样本。
    retrieved: list[RetrievedExample] = field(default_factory=list)
    # 额外诊断信息，例如解析细节、策略名等。
    metadata: dict[str, Any] = field(default_factory=dict)


# 旧版 compact memory system 需要实现的接口。
class MemorySystem(Protocol):
    # 系统名，用于日志、leaderboard 和结果文件。
    name: str

    # 从训练样本中构建或更新记忆状态。
    def learn(self, examples: list[Example], llm: Any) -> None:
        raise NotImplementedError

    # 对单条样本做预测，并返回结构化预测记录。
    def predict(self, example: Example, llm: Any) -> Prediction:
        raise NotImplementedError


# 将文本切成小写英文/数字 token。
def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


# 将候选标签列表格式化成逗号分隔字符串。
def format_choices(choices: Sequence[str]) -> str:
    return ", ".join(str(choice) for choice in choices)


# 尽量从普通文本、Markdown 代码块或花括号片段中解析 JSON。
def safe_json_loads(text: str) -> Any | None:
    stripped = text.strip()

    # 依次尝试：完整文本、fenced JSON、首尾花括号片段。
    for candidate in (stripped, _extract_fenced_json(stripped), _extract_braced_json(stripped)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


# 从模型输出中抽取预测标签。
def extract_label(text: str, choices: Sequence[str]) -> str:
    # 优先解析 JSON 字段，兼容 final_answer / label / answer。
    data = safe_json_loads(text)
    if isinstance(data, dict):
        value = str(data.get("final_answer") or data.get("label") or data.get("answer") or "")
        if not choices and value.strip():
            return value.strip()
        for choice in choices:
            if value.strip().lower() == str(choice).strip().lower():
                return str(choice)
        if value.strip():
            return value.strip()

    # JSON 解析失败时，退回到候选项文本包含匹配。
    lowered = text.lower()
    for choice in choices:
        if str(choice).lower() in lowered:
            return str(choice)
    return "unknown"


# 提取 Markdown fenced code block 中的 JSON 内容。
def _extract_fenced_json(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    return match.group(1) if match else None


# 提取文本中从第一个 { 到最后一个 } 的 JSON 候选片段。
def _extract_braced_json(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")

    # 没有完整花括号包围时不尝试解析。
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]
