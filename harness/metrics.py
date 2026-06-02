"""
文件分析：数学与指标统计库。
主要作用：负责纯粹的量化评估数学统计。根据多条判断日志来最终计算输出各个数据验证测试任务下的微平均F1、宏平均F1、总Accuracy等关键比赛成绩。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def normalize_label(value: str, choices: Sequence[str]) -> str:
    # Snap a model output back to one of the allowed labels when possible.
    raw = str(value).strip()
    raw_lower = raw.lower()
    for choice in choices:
        if raw_lower == str(choice).strip().lower():
            return str(choice)
    for choice in choices:
        if str(choice).strip().lower() in raw_lower:
            return str(choice)
    return raw


def accuracy(predictions: Sequence[Mapping[str, Any]]) -> float:
    # Compute plain accuracy from scored rows.
    if not predictions:
        return 0.0
    correct = 0
    for row in predictions:
        if "was_correct" in row:
            correct += int(bool(row["was_correct"]))
        elif str(row.get("prediction", "")).strip() == str(row.get("label", "")).strip():
            correct += 1
    return correct / len(predictions)


def per_dataset_scores(results: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    # Group rows by dataset and compute accuracy per group.
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in results:
        dataset = str(row.get("dataset", "unknown"))
        grouped.setdefault(dataset, []).append(row)
    return {dataset: accuracy(rows) for dataset, rows in grouped.items()}


def macro_average(scores: Mapping[str, float]) -> float:
    # Average dataset scores equally, regardless of dataset size.
    if not scores:
        return 0.0
    return sum(float(score) for score in scores.values()) / len(scores)


def compute_micro_f1(predictions: Sequence[Mapping[str, Any]]) -> float:
    # Aggregate all TP/FP/FN counts before computing a single micro-F1.
    total_tp = total_fp = total_fn = 0
    has_data = False
    for row in predictions:
        metrics = row.get("metrics", {})
        if "tp" in metrics:
            total_tp += int(metrics["tp"])
            total_fp += int(metrics["fp"])
            total_fn += int(metrics["fn"])
            has_data = True
    if not has_data:
        return 0.0
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def make_result(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Build the summary block stored in benchmark outputs and logs.
    correct = sum(1 for row in predictions if bool(row.get("was_correct", False)))
    result = {
        "accuracy": correct / len(predictions) if predictions else 0.0,
        "correct": correct,
        "total": len(predictions),
    }
    f1_values = [
        row["metrics"]["f1"]
        for row in predictions
        if row.get("metrics", {}).get("f1") is not None
    ]
    if f1_values:
        result["avg_f1"] = sum(float(value) for value in f1_values) / len(f1_values)
        result["micro_f1"] = compute_micro_f1(predictions)
    else:
        result["avg_f1"] = None
        result["micro_f1"] = None
    return result
