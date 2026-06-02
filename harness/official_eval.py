# official_eval.py
# 本模块实现 official-like 的离线评测流程：加载候选 memory agent、
# 在 train split 上训练 memory，在 val/test split 上预测并记录分数、trace 和状态文件。

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from harness.agent_protocol import llm_client_to_callable, load_agent_memory
from harness.data import load_examples
from harness.eval import DatasetSpec, import_memory_system
from harness.evaluators import get_evaluator, unpack_eval_result
from harness.metrics import make_result, macro_average, per_dataset_scores


# 一次 official-like 评测的汇总结果，包含各数据集分数、平均分、trace 和错误信息。
@dataclass
class OfficialBenchmarkResult:
    system_name: str
    split: str
    per_dataset: dict[str, float]
    average: float
    summary: dict[str, Any] = field(default_factory=dict)
    traces: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    # 返回本次评测的样本总数；优先使用 summary 中的 total。
    @property
    def total(self) -> int:
        return int(self.summary.get("total", len(self.traces)))

    # 判断本次评测是否完全失败，即没有有效样本且存在错误记录。
    @property
    def all_failed(self) -> bool:
        return self.total == 0 and bool(self.errors)


# 把旧版 compact MemorySystem 适配成 official inner-loop 需要的接口形状。
class OfficialMemoryAdapter:

    # 保存原始 memory system、LLM 句柄和线程本地 prompt 记录。
    def __init__(self, system: Any, llm: Any):
        self.system = system
        self.llm = llm
        self._prompt_local = threading.local()

    # 返回被包装系统的名称。
    @property
    def name(self) -> str:
        return str(getattr(self.system, "name", self.system.__class__.__name__))

    # 把 harness 的 batch row 转成 official example，再交给旧版 system.learn。
    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        examples = [
            _official_row_to_example(
                row,
                row_id=f"train-{idx}",
                text_key="input",
                label_key="ground_truth",
            )
            for idx, row in enumerate(batch_results)
        ]
        self.system.learn(examples, self.llm)

    # 把输入转成 official example，调用旧版 system.predict，并记录 prompt 元数据。
    def predict(self, input_text: str) -> tuple[str, dict[str, Any]]:
        example = _official_row_to_example(
            {"input": input_text, "target": ""},
            row_id="eval",
            text_key="input",
            label_key="target",
        )
        prediction = self.system.predict(example, self.llm)
        prompt_text = _prompt_to_text(prediction.prompt)
        self._prompt_local.last_prompt_len = len(prompt_text)
        self._prompt_local.last_prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:8]
        self._prompt_local.last_prompt_text = prompt_text
        metadata = dict(prediction.metadata)
        metadata.setdefault("raw_output", prediction.raw_output)
        metadata.setdefault("retrieved", [item.__dict__ for item in prediction.retrieved])
        return prediction.label, metadata

    # 返回最近一次预测时使用的 prompt 长度、哈希和原文。
    def get_last_prompt_info(self) -> dict[str, Any]:
        return {
            "prompt_len": getattr(self._prompt_local, "last_prompt_len", None),
            "prompt_hash": getattr(self._prompt_local, "last_prompt_hash", None),
            "prompt_text": getattr(self._prompt_local, "last_prompt_text", None),
        }

    # 如果底层系统支持导出状态，则直接委托给底层系统。
    def get_state(self) -> str:
        examples = [getattr(example, "__dict__", {}) for example in getattr(self.system, "examples", [])]
        return json.dumps({"examples": examples}, ensure_ascii=False)

    # 如果底层系统支持恢复状态，则直接委托给底层系统。
    def set_state(self, state: str) -> None:
        data = json.loads(state)
        if "examples" in data and hasattr(self.system, "examples"):
            self.system.examples = [
                _official_row_to_example(row, f"state-{idx}", "text", "label")
                for idx, row in enumerate(data["examples"])
            ]


# official-like 总评测入口：按数据集加载 memory、训练/恢复状态、并发预测并汇总分数。
def evaluate_system_official(
    system_path: str | Path,
    datasets: Sequence[DatasetSpec],
    split: str,
    llm: Any,
    config: Mapping[str, Any] | None = None,
    max_workers: int = 32,
) -> OfficialBenchmarkResult:
    eval_config = dict(config or {})
    seed = int(eval_config.get("seed", 42))
    mode = str(eval_config.get("mode", "offline"))
    num_epochs = int(eval_config.get("num_epochs", 1))
    batch_size = int(eval_config.get("batch_size", 1))
    combined_eval = bool(eval_config.get("combined_eval", False))
    output_root = Path(str(eval_config["output_root"])) if eval_config.get("output_root") else None
    model_short = str(eval_config.get("model_short", "model"))
    write_artifacts = bool(eval_config.get("write_artifacts", output_root is not None))

    traces: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    system_name = Path(system_path).stem

    for dataset in datasets:
        memory = _load_official_memory(system_path, llm, eval_config)
        system_name = Path(system_path).stem
        try:
            train_examples, val_examples, test_examples = load_dataset_splits_3way_from_specs(
                dataset,
                seed=seed,
                limits=dict(eval_config.get("dataset_limits", {})),
            )
            evaluator = get_evaluator(dataset.name)
            if mode != "offline":
                raise ValueError("official_eval currently supports offline mode")
            logs_dir = (
                output_root / "logs" / dataset.name / system_name / model_short
                if output_root is not None
                else None
            )
            artifact_dir = (
                output_root
                / ("results" if split == "test" else "logs")
                / dataset.name
                / system_name
                / model_short
                if output_root is not None
                else None
            )
            # test 阶段不重新训练，直接读取 val 阶段保存的 memory 状态。
            if split == "test":
                _load_saved_memory_for_test(memory, logs_dir, dataset.name, system_name)
            # val/train 阶段先做 offline 训练，再保存可复用 memory。
            else:
                _offline_train(
                    memory=memory,
                    examples=train_examples,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    log_path=logs_dir / "log.jsonl" if logs_dir and write_artifacts else None,
                )
                if split == "val" and logs_dir and write_artifacts:
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    (logs_dir / "memory.json").write_text(memory.get_state(), encoding="utf-8")
            if split == "val":
                eval_examples = val_examples
            elif split == "test":
                eval_examples = test_examples
            elif split == "combined":
                eval_examples = val_examples + test_examples
            else:
                raise ValueError(f"Unknown split: {split}")
            if combined_eval and split in {"val", "test"}:
                combined_rows = evaluate_memory_official(
                    memory=memory,
                    examples=val_examples + test_examples,
                    check_answer=evaluator,
                    dataset_name=dataset.name,
                    max_workers=max_workers,
                )
                rows = (
                    combined_rows[: len(val_examples)]
                    if split == "val"
                    else combined_rows[len(val_examples) :]
                )
            else:
                rows = evaluate_memory_official(
                    memory=memory,
                    examples=eval_examples,
                    check_answer=evaluator,
                    dataset_name=dataset.name,
                    max_workers=max_workers,
                )
            traces.extend(rows)
            if artifact_dir and write_artifacts:
                _write_split_artifacts(
                    artifact_dir=artifact_dir,
                    split=split,
                    rows=rows,
                    dataset_name=dataset.name,
                    memory_name=system_name,
                    model_short=model_short,
                )
        except Exception as exc:
            errors.append({"dataset": dataset.name, "error": str(exc)})

    per_dataset = per_dataset_scores(traces)
    summary = make_result(traces)
    return OfficialBenchmarkResult(
        system_name=system_name,
        split=split,
        per_dataset=per_dataset,
        average=macro_average(per_dataset),
        summary=summary,
        traces=traces,
        errors=errors,
    )


# 按 DatasetSpec 加载每个数据集的 train/val/test 三个 split。
def load_dataset_splits_3way_from_specs(
    dataset: DatasetSpec,
    seed: int = 42,
    limits: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    limit_row = dict(limits or {}).get(dataset.name, {})
    train = _official_rows(load_examples(dataset.train_path, task=dataset.name))
    val = _official_rows(load_examples(dataset.val_path, task=dataset.name))
    test = _official_rows(load_examples(dataset.test_path, task=dataset.name))
    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(val)
    random.Random(seed).shuffle(test)
    return (
        train[: int(limit_row.get("num_train", len(train)))],
        val[: int(limit_row.get("num_val", len(val)))],
        test[: int(limit_row.get("num_test", len(test)))],
    )


# 评估一个已经构造好的 memory 实例，返回当前 split 的逐样本 trace 和错误列表。
def evaluate_memory_official(
    memory: Any,
    examples: Sequence[dict[str, Any]],
    check_answer: Callable[..., bool | dict[str, Any]],
    dataset_name: str,
    max_workers: int = 32,
) -> list[dict[str, Any]]:
    if not examples:
        return []

    # 单样本预测函数：调用 memory.predict、判分、记录 prompt 与耗时。
    # 内部函数用于并发执行单条样本预测。
    def predict_one(idx: int, ex: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        pred, metadata = memory.predict(ex["input"])
        prompt_info = memory.get_last_prompt_info()
        prompt_len = prompt_info.get("prompt_len") or 0
        prompt_text = prompt_info.get("prompt_text") or ""
        context_len = max(0, prompt_len - len(ex["input"])) if prompt_len else 0
        raw = check_answer(pred, ex["target"], **_get_eval_kwargs(ex))
        ok, metrics = unpack_eval_result(raw)
        result = {
            "dataset": dataset_name,
            "id": f"{dataset_name}-{idx}",
            "prediction": pred,
            "target": ex["target"],
            "label": ex["target"],
            "was_correct": ok,
            "prompt_len": prompt_len,
            "context_len": context_len,
            "prompt_text": prompt_text,
            "prompt": prompt_text,
            "metadata": metadata,
            "raw_output": metadata.get("full_response", metadata.get("raw_output", "")),
        }
        if metrics:
            result["metrics"] = metrics
        return idx, result

    results: list[dict[str, Any] | None] = [None] * len(examples)
    worker_count = max(1, min(int(max_workers), len(examples)))
    # 多线程路径用于并发评测，提高大数据集速度。
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(predict_one, idx, ex): idx for idx, ex in enumerate(examples)}
        for future in as_completed(futures):
            try:
                idx, result = future.result()
            except Exception as exc:
                idx = futures[future]
                result = _failed_eval_row(idx, examples[idx], dataset_name, exc)
            results[idx] = result
    return [row for row in results if row is not None]


# 把官方评测的 trace 和 error 汇总成 OfficialBenchmarkResult。
def make_result_official(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return make_result(predictions)


# 构造一条预测失败时的 trace 记录，保证错误也能进入日志。
def _failed_eval_row(
    idx: int,
    ex: Mapping[str, Any],
    dataset_name: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "id": f"{dataset_name}-{idx}",
        "prediction": "",
        "target": ex.get("target", ""),
        "label": ex.get("target", ""),
        "was_correct": False,
        "prompt_len": 0,
        "context_len": 0,
        "prompt_text": "",
        "prompt": "",
        "metadata": {"error": str(exc), "error_type": exc.__class__.__name__},
        "raw_output": "",
        "error": str(exc),
        "error_type": exc.__class__.__name__,
    }


# offline 训练阶段：把 train split 按 epoch 和 batch 喂给 memory.learn_from_batch。
def _offline_train(
    memory: Any,
    examples: Sequence[dict[str, Any]],
    batch_size: int,
    num_epochs: int,
    log_path: Path | None = None,
) -> None:
    start = time.time()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        _append_jsonl(log_path, {"type": "meta", "mode": "offline", "num_epochs": num_epochs})
    for _epoch in range(num_epochs):
        for batch_start in range(0, len(examples), batch_size):
            batch = examples[batch_start : batch_start + batch_size]
            batch_results = []
            for ex in batch:
                row = {
                    "input": ex["input"],
                    "prediction": ex["target"],
                    "ground_truth": ex["target"],
                    "was_correct": True,
                }
                for key, value in ex.items():
                    if key not in ("input", "target") and key not in row:
                        row[key] = value
                batch_results.append(row)
            train_start = time.time()
            # 候选 memory 的训练/记忆写入发生在这里。
            memory.learn_from_batch(batch_results)
            if log_path is not None:
                _append_jsonl(
                    log_path,
                    {
                        "type": "train_batch",
                        "t": round(time.time() - start, 2),
                        "epoch": _epoch,
                        "step": batch_start,
                        "batch_size": len(batch),
                        "train_ms": int((time.time() - train_start) * 1000),
                    },
                )


# 把 Example 列表转换成 official-like 评测行。
def _official_rows(examples: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        row = {"input": example.text, "target": example.label}
        metadata = getattr(example, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            for key, value in metadata.items():
                if key == "raw_input" and key not in row:
                    row[key] = value
        rows.append(row)
    return rows


# 根据 agent 文件加载 memory；优先使用新协议，失败时回退旧版 compact 协议适配。
def _load_official_memory(system_path: str | Path, llm: Any, eval_config: Mapping[str, Any]) -> Any:
    # 优先加载实现 BaseAgentMemory 协议的新候选。
    try:
        memory = load_agent_memory(
            system_path,
            llm=llm_client_to_callable(llm),
            project_root=eval_config.get("project_root"),
        )
        return memory
    except Exception:
        if bool(eval_config.get("agent_protocol_only", False)):
            raise
        system = import_memory_system(system_path, config=dict(eval_config.get("system_config", {})))
        # 旧协议系统需要通过 adapter 暴露 learn/predict/state 接口。
        return OfficialMemoryAdapter(system, llm)


# test 阶段加载 val 阶段保存的 memory.json。
def _load_saved_memory_for_test(memory: Any, artifact_dir: Path | None, dataset_name: str, system_name: str) -> None:
    if artifact_dir is None:
        return
    memory_path = artifact_dir / "memory.json"
    # 如果没有 val 阶段保存的 memory，test 无法复用训练状态。
    if not memory_path.exists():
        raise FileNotFoundError(
            f"No saved memory state for test: {dataset_name}/{system_name}. Run val first: {memory_path}"
        )
    # 把 memory.json 恢复到新建的 memory 实例中。
    memory.set_state(memory_path.read_text(encoding="utf-8"))


# 把某个数据集某个 split 的聚合指标、逐样本日志和 memory 状态写到磁盘。
def _write_split_artifacts(
    artifact_dir: Path,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    dataset_name: str,
    memory_name: str,
    model_short: str,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    correct = sum(1 for row in rows if row.get("was_correct"))
    total = len(rows)
    avg_context = int(
        sum(int(row.get("context_len", row.get("prompt_len", 0)) or 0) for row in rows) / total
    ) if total else 0
    # split.json 保存聚合分数和上下文长度等摘要。
    payload = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "error_count": sum(1 for row in rows if row.get("error")),
        "dataset": dataset_name,
        "memory": memory_name,
        "model": model_short,
        "mode": "offline",
        "memory_context_chars": avg_context,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (artifact_dir / f"{split}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log_path = artifact_dir / "log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "type": "eval_step",
                        "step": index,
                        "input_preview": str(row.get("prompt_text") or row.get("prompt") or "")[:200],
                        "pred": row.get("prediction"),
                        "tgt": row.get("target"),
                        "ok": row.get("was_correct"),
                        "metrics": row.get("metrics", {}),
                        "error": row.get("error"),
                        "error_type": row.get("error_type"),
                        "prompt_len": row.get("prompt_len"),
                        "prompt_hash": row.get("prompt_hash"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# 向 JSONL 文件追加一条 JSON 记录。
def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


# 把 harness row 转成旧版 official adapter 需要的 Example 对象。
def _official_row_to_example(
    row: Mapping[str, Any],
    row_id: str,
    text_key: str,
    label_key: str,
) -> Any:
    from harness.memory import Example

    # 保留原 row 的 metadata，并补充内部 row_id。
    metadata = dict(row.get("metadata", {})) if isinstance(row.get("metadata"), dict) else {}
    for key, value in row.items():
        if key not in {text_key, label_key, "metadata"}:
            metadata.setdefault(key, value)
    return Example(
        id=str(row.get("id") or row_id),
        text=str(row.get(text_key, "")),
        label=str(row.get(label_key, "")),
        choices=[str(choice) for choice in row.get("choices", [])],
        metadata=metadata,
    )


# 根据 evaluator 函数签名决定是否传入 choices 和 metadata。
def _get_eval_kwargs(ex: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = {key: value for key, value in ex.items() if key not in ("input", "target")}
    if "raw_input" in ex:
        kwargs["input_nums"] = ex["raw_input"]
    return kwargs


# 把 chat messages 或普通 prompt 统一渲染成纯文本。
def _prompt_to_text(prompt: Any) -> str:
    # chat messages 需要把 role/content 展平成文本用于长度统计。
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, Mapping):
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(message))
        return "\n\n".join(parts)
    return str(prompt)
