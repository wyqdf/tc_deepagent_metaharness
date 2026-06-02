"""Benchmark runner for text classification memory systems.

文件分析：候选基准验证与评测分发器。
主要作用：动态加载系统目录下自动写好的测试代码（代理 Memory 系统），调度外部评测验证逻辑，实现对其代码准确性、损耗长进行量化评分。
"""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from harness.data import load_examples
from harness.memory import Example, MemorySystem
from harness.evaluators import get_evaluator, unpack_eval_result
from harness.metrics import make_result, macro_average, per_dataset_scores


@dataclass
class DatasetSpec:
    name: str
    train_path: str
    val_path: str
    test_path: str


@dataclass
class BenchmarkResult:
    system_name: str
    split: str
    per_dataset: dict[str, float]
    average: float
    summary: dict[str, Any] = field(default_factory=dict)
    traces: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        # Prefer the explicit summary count, fall back to the trace count.
        return int(self.summary.get("total", len(self.traces)))

    @property
    def all_failed(self) -> bool:
        return self.total == 0 and bool(self.errors)


def load_jsonl(path: str | Path, task: str | None = None) -> list[Example]:
    # Shared loader used by both the compact and official-like benchmarks.
    return load_examples(path, task=task)


def import_memory_system(
    path: str | Path,
    factory: str = "build_memory_system",
    config: dict[str, Any] | None = None,
) -> MemorySystem:
    # Dynamically import a candidate file and call its builder factory.
    module_path = Path(path)
    if not module_path.is_absolute():
        module_path = Path.cwd() / module_path
    module_name = f"_tcharness_candidate_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import memory system from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    builder = getattr(module, factory)
    return builder(config or {})


def evaluate_system(
    system_path: str | Path,
    datasets: Sequence[DatasetSpec],
    split: str,
    llm: Any,
    config: dict[str, Any] | None = None,
    max_workers: int = 1,
) -> BenchmarkResult:
    # Run train-then-evaluate for each dataset and aggregate the scores.
    system = import_memory_system(system_path, config=config)
    traces: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        train_examples = load_jsonl(dataset.train_path, task=dataset.name)
        eval_examples = load_jsonl(_split_path(dataset, split), task=dataset.name)
        system.learn(train_examples, llm)
        dataset_rows, dataset_errors = _evaluate_examples(
            system=system,
            examples=eval_examples,
            dataset_name=dataset.name,
            llm=llm,
            max_workers=max_workers,
        )
        traces.extend(dataset_rows)
        score_rows.extend(dataset_rows)
        errors.extend(dataset_errors)

    per_dataset = per_dataset_scores(score_rows)
    summary = make_result(score_rows)
    return BenchmarkResult(
        system_name=getattr(system, "name", Path(system_path).stem),
        split=split,
        per_dataset=per_dataset,
        average=macro_average(per_dataset),
        summary=summary,
        traces=traces,
        errors=errors,
    )


def _evaluate_examples(
    system: MemorySystem,
    examples: Sequence[Example],
    dataset_name: str,
    llm: Any,
    max_workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Evaluate examples in order while collecting per-example failures separately.
    if not examples:
        return [], []
    worker_count = max(1, min(int(max_workers), len(examples)))
    ordered: list[dict[str, Any] | None] = [None] * len(examples)
    errors: list[dict[str, Any]] = []

    if worker_count == 1:
        for idx, example in enumerate(examples):
            try:
                ordered[idx] = _predict_one(system, example, dataset_name, llm)
            except Exception as exc:
                errors.append({"dataset": dataset_name, "id": example.id, "error": str(exc)})
        return [row for row in ordered if row is not None], errors

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_predict_one, system, example, dataset_name, llm): (idx, example)
            for idx, example in enumerate(examples)
        }
        for future in as_completed(futures):
            idx, example = futures[future]
            try:
                ordered[idx] = future.result()
            except Exception as exc:
                errors.append({"dataset": dataset_name, "id": example.id, "error": str(exc)})
    return [row for row in ordered if row is not None], errors


def _predict_one(
    system: MemorySystem,
    example: Example,
    dataset_name: str,
    llm: Any,
) -> dict[str, Any]:
    # Run one prediction, score it, and keep the raw trace fields.
    prediction = system.predict(example, llm)
    evaluator = get_evaluator(dataset_name)
    raw_eval = evaluator(prediction.label, example.label)
    was_correct, extra_metrics = unpack_eval_result(raw_eval)
    row = {
        "dataset": dataset_name,
        "id": example.id,
        "prediction": prediction.label,
        "label": example.label,
        "target": example.label,
        "was_correct": was_correct,
        "raw_output": prediction.raw_output,
        "prompt": prediction.prompt,
        "retrieved": [item.__dict__ for item in prediction.retrieved],
        "metadata": prediction.metadata,
    }
    if extra_metrics:
        row["metrics"] = extra_metrics
    return row


def _split_path(dataset: DatasetSpec, split: str) -> str:
    # Resolve the file path for the requested split.
    if split == "train":
        return dataset.train_path
    if split == "val":
        return dataset.val_path
    if split == "test":
        return dataset.test_path
    raise ValueError(f"Unknown split: {split}")
