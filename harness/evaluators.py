"""
文件分析：结果比对的评分裁判器。
主要作用：负责接收模型基于具体 Task 的各类长篇大论 JSON 输出，基于严格的判断逻辑提取真正的答案字段，输出规范化的布尔信号（正误判断），为性能指标统计垫边。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable


TRANSFER_TASKS = {
    "AGNews",
    "GoEmotions",
    "Banking77",
    "FinancialPhraseBank",
    "SciCite",
    "TweetEval_hate",
    "Amazon5",
    "SciTail",
}

#提取最终答案的函数，适用于不同格式的文本响应，尝试多种解析方法以确保提取到正确的答案。
def extract_final_answer(response_text: str) -> str:
    """Extract final_answer exactly like the reference evaluator."""
    # Keep trying progressively looser parses until final_answer is recovered.
    if not response_text:
        return ""

    text = response_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        payload = []
        in_json = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```json"):
                in_json = True
                continue
            if stripped.startswith("```") and in_json:
                break
            if in_json:
                payload.append(line)
        if payload:
            text = "\n".join(payload)
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:].strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "final_answer" in data:
            return str(data["final_answer"])
    except json.JSONDecodeError:
        pass

    candidate = _extract_braced_json(text)
    if candidate:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(data, dict) and "final_answer" in data:
                return str(data["final_answer"])

    return response_text

#获取评估器函数，根据任务名称返回相应的评估函数，这些评估函数根据不同的任务类型进行正确性判断和指标计算。
def get_evaluator(task: str) -> Callable[..., bool | dict[str, Any]]:
    # Pick the task-specific scorer and normalize its return shape.
    canonical = canonical_task_name(task)
    if canonical == "FiNER":
        return lambda pred, target, **_: eval_finer(pred, target)
    if canonical == "USPTO":

        def _eval_uspto(pred: str, target: str, **_: Any) -> dict[str, Any]:
            raw = eval_uspto(pred, target)
            return {
                "was_correct": raw["correct"],
                "metrics": {"jaccard_similarity": raw["jaccard_similarity"]},
            }

        return _eval_uspto
    if canonical == "Symptom2Disease":
        return lambda pred, target, **_: eval_symptom2disease(pred, target)
    if canonical == "LawBench":

        def _eval_lawbench(pred: str, target: str, **_: Any) -> dict[str, Any]:
            raw = eval_lawbench(pred, target)
            return {
                "was_correct": raw["correct"],
                "metrics": {
                    "f1": raw["f1"],
                    "precision": raw["precision"],
                    "recall": raw["recall"],
                    "tp": raw["tp"],
                    "fp": raw["fp"],
                    "fn": raw["fn"],
                },
            }

        return _eval_lawbench
    if canonical == "AEGIS":
        return lambda pred, target, **_: eval_aegis(pred, target)
    if canonical in TRANSFER_TASKS:
        return lambda pred, target, **_: eval_classification(pred, target)
    return lambda pred, target, **_: eval_classification(pred, target)

#将评估结果进行统一处理，无论是简单的布尔值还是包含指标的字典，都转换为一个统一的格式，方便后续的分析和比较。
def unpack_eval_result(raw: bool | dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    # Convert either a bare boolean or a metrics dict into a uniform tuple.
    if isinstance(raw, dict):
        return bool(raw["was_correct"]), dict(raw.get("metrics", {}))
    return bool(raw), {}


def canonical_task_name(task: str) -> str:
    # Use one canonical name per dataset family, regardless of alias.
    mapping = {
        "uspto": "USPTO",
        "symptom2disease": "Symptom2Disease",
        "lawbench": "LawBench",
        "finer": "FiNER",
        "aegis": "AEGIS",
        "aegis2": "AEGIS",
    }
    return mapping.get(task.lower(), task)


def eval_finer(prediction: str, target: str) -> bool:
    # FiNER is exact-match on the extracted final answer.
    return extract_final_answer(prediction).lower().strip() == target.lower().strip()


def eval_uspto(prediction: str, target: str) -> dict[str, Any]:
    # Compare reactant sets exactly and also report Jaccard overlap.
    def parse_reactants(smiles: str) -> set[str]:
        return {part.strip().lower() for part in smiles.strip().split(".") if part.strip()}

    pred_set = parse_reactants(extract_final_answer(prediction))
    target_set = parse_reactants(target)
    if not pred_set and not target_set:
        jaccard = 1.0
    elif not pred_set or not target_set:
        jaccard = 0.0
    else:
        jaccard = len(pred_set & target_set) / len(pred_set | target_set)
    return {"correct": pred_set == target_set, "jaccard_similarity": jaccard}


def eval_symptom2disease(prediction: str, target: str) -> bool:
    # Pull the diagnosis out of the structured answer and compare normalized text.
    text = extract_final_answer(prediction)
    match = re.search(r"\[DIAGNOSIS\](.*?)\[/DIAGNOSIS\]", text, re.I | re.S)
    if match:
        text = match.group(1).strip()
    else:
        match = re.search(r"(?:diagnosis|final diagnosis|conclusion)[:：]\s*([^\n]+)", text, re.I)
        if match:
            text = match.group(1).strip()

    return _normalize_plain(text) == _normalize_plain(target)


def eval_lawbench(prediction: str, target: str) -> dict[str, Any]:
    # Score LawBench as exact charge-set match plus precision/recall/F1.
    pred_charges = _parse_lawbench_charges(prediction)
    true_charges = _parse_lawbench_charges(target)
    tp = len(pred_charges & true_charges)
    fp = len(pred_charges - true_charges)
    fn = len(true_charges - pred_charges)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "correct": pred_charges == true_charges,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def eval_aegis(prediction: str, target: str) -> bool:
    # Collapse the answer to safe/unsafe and compare directly.
    pred = extract_final_answer(prediction).lower()
    if "unsafe" in pred:
        pred = "unsafe"
    elif "safe" in pred:
        pred = "safe"
    else:
        pred = pred.strip()
    return pred == target.lower().strip()


def eval_classification(prediction: str, target: str) -> bool:
    # Plain label classification uses normalized exact match.
    text = extract_final_answer(prediction).lower().strip().replace("_", " ")
    target_text = target.lower().strip().replace("_", " ")
    return _normalize_plain(text) == _normalize_plain(target_text)


def _parse_lawbench_charges(text: str) -> set[str]:
    # Extract a normalized set of charge labels from the LawBench answer format.
    text = extract_final_answer(text).strip()
    match = re.search(r"\[罪名\](.*?)(?:<eoa>|$)", text)
    if match:
        text = match.group(1).strip()
    elif "罪名:" in text:
        text = text.split("罪名:")[-1]
    text = re.sub(r"<eoa>.*", "", text).strip()
    for sep in [";", "；", ",", "，", "、"]:
        if sep in text:
            return {part.strip() for part in text.split(sep) if part.strip()}
    return {text} if text else set()


def _normalize_plain(value: str) -> str:
    # Lowercase, collapse whitespace, and trim trailing punctuation.
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    return re.sub(r"[.!?]+$", "", value)


def _extract_braced_json(text: str) -> str | None:
    # Best-effort brace matching for a JSON object embedded in text.
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    return match.group() if match else None
