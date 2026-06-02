# Meta-Harness 外层循环模块。
#
# 本模块负责评估 baseline、调用 proposer 生成候选 agent、验证和评估候选、
# 维护 leaderboard/frontier，并在最后执行 test 评测。

from __future__ import annotations

import ast
import os
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from harness.agent_protocol import load_agent_memory
from harness.eval import DatasetSpec, evaluate_system
from harness.llm import build_solver_client, preflight_solver
from harness.official_eval import evaluate_system_official
from harness.prompts import build_proposer_prompt
from harness.proposer import DeepAgentProposer
from harness.store import RunStore


@dataclass
# 原始配置和解析后的项目根目录。
class HarnessConfig:
    raw: dict[str, Any]
    project_root: Path
# 读取 YAML 实验配置文件。
def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
    
    
# 从配置文件启动一次完整的 Meta-Harness 实验。
#
# 流程包括 solver 预检查、baseline 验证集评估、候选生成与评估、
# frontier 更新，以及最终 test 评测。
def run_harness(config_path: str | Path) -> dict[str, Any]:
    # 解析配置路径，并建立本次实验的运行目录。
    config_path = Path(config_path)
    project_root = config_path.resolve().parents[1]
    raw = load_config(config_path)
    harness_config = HarnessConfig(raw=raw, project_root=project_root)
    store = RunStore(project_root / "runs", raw["run"]["name"])
    store.create()
    store.log_event("run_started", {"config_path": str(config_path)})
    store.snapshot_config(raw)

    # 可选预检查：先确认 solver 可调用，避免长时间运行后才失败。
    if _preflight_enabled(raw):
        store.log_event("solver_preflight_started", {})
        preflight = preflight_solver(raw)
        if not preflight["ok"]:
            store.log_event("solver_preflight_failed", preflight)
            raise RuntimeError(f"Solver preflight failed: {preflight['error']}")
        store.log_event("solver_preflight_finished", preflight)

    # 准备数据集、solver、并发数和后续评测配置。
    datasets = _dataset_specs(harness_config)
    solver = build_solver_client(raw)
    max_workers = _max_workers(raw)
    leaderboard: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []

    eval_config = _official_eval_config(raw, project_root=project_root, run_dir=store.path)
    baseline_rows: list[dict[str, Any]] = []
    # 先评估 baseline，作为后续候选 agent 的对照分数。
    for baseline_name in _baseline_names(raw):
        baseline_path = _agent_path(project_root, baseline_name)
        store.log_event(
            "baseline_eval_started",
            {"system": baseline_name, "path": str(baseline_path), "split": "val", "max_workers": max_workers},
        )
        baseline_result = _evaluate(
            baseline_path,
            datasets,
            "val",
            solver,
            eval_config,
            max_workers=max_workers,
        )
        _stop_if_eval_failed(store, baseline_result, baseline_name)
        row = _leaderboard_row(
            baseline_name,
            str(baseline_path),
            baseline_result.average,
            _average_context_length(store.path, baseline_name, _model_short_name(raw), datasets, "logs"),
        )
        leaderboard.append(row)
        baseline_rows.append(_benchmark_to_dict(baseline_result))
        history.append({"name": baseline_name, "path": str(baseline_path), "result": baseline_result})
        _write_evolution_summary(store.path, 0, baseline_name, baseline_result, row, "baseline", "baseline")
        store.log_event(
            "baseline_eval_finished",
            {"system": baseline_name, "average": baseline_result.average, "summary": baseline_result.summary},
        )
        _print_score(baseline_name, "val", baseline_result)
    best = max(leaderboard, key=lambda item: float(item["average"]))
    store.write_json("baseline_scores.json", baseline_rows)
    store.write_leaderboard(leaderboard)
    frontier = _write_official_frontier(store.path, leaderboard, _model_short_name(raw), datasets)
    store.save_frontier({"best": best, "leaderboard": leaderboard})

    # 创建 DeepAgent proposer；它只负责生成候选代码，不负责打分。
    rounds = int(raw.get("run", {}).get("rounds", 1))
    proposer_config = dict(raw.get("proposer", {}))
    proposer = DeepAgentProposer(
        model=str(proposer_config.get("model", "claude-opus-4-6")),
        output_dir=project_root / raw["candidate"]["output_dir"],
        env=_proposer_env(proposer_config),
        dry_run=bool(proposer_config.get("dry_run", False)),
        root_dir=project_root,
        allowed_run_dir=store.path,
        max_retries=int(proposer_config.get("max_retries", 2)),
        retry_sleep_seconds=float(proposer_config.get("retry_sleep_seconds", 10.0)),
    )

    # 外层进化循环：每轮生成、验证、评估候选，并更新排行榜。
    for iteration in range(1, rounds + 1):
        iter_dir = store.iter_dir(iteration)
        store.log_event("iteration_started", {"iteration": iteration, "rounds": rounds})
        # 把当前 leaderboard、history、frontier 和源码摘要打包给 proposer。
        task_prompt = _build_official_like_proposer_context(
            project_root=project_root,
            config=raw,
            run_dir=store.path,
            iteration=iteration,
            datasets=datasets,
            leaderboard=leaderboard,
            history=history,
        )
        store.write_text(f"{iter_dir.name}/proposer_prompt.md", task_prompt)
        store.log_event("proposer_started", {"iteration": iteration, "model": proposer.model})
        # DeepAgent 写候选文件，并通过 pending_eval.json 告诉 loop 评测哪些候选。
        proposals = proposer.propose_official(
            task_prompt,
            iteration,
            store.path / "pending_eval.json",
            trace_path=iter_dir / "proposer_tool_calls.jsonl",
            response_path=iter_dir / "proposer_response.md",
            messages_path=iter_dir / "proposer_messages.json",
        )
        store.log_event(
            "proposer_finished",
            {"iteration": iteration, "candidates": [proposal.name for proposal in proposals]},
        )
        _write_proposer_audit(
            store.path,
            iter_dir,
            task_prompt,
            proposals,
            iter_dir / "proposer_tool_calls.jsonl",
            raw,
        )
        # 先审计和过滤候选；导入失败或疑似数据集硬编码的候选不会进入评测。
        proposal_decisions = _validate_proposals_with_reasons(proposals)
        valid_proposals = [row["proposal"] for row in proposal_decisions if row["valid"]]
        for row in proposal_decisions:
            if row["valid"]:
                continue
            proposal = row["proposal"]
            payload = {
                "iteration": iteration,
                "candidate": proposal.name,
                "path": proposal.path,
                "reasons": row["reasons"],
            }
            store.log_event("candidate_rejected", payload)
            store.append_jsonl(f"{iter_dir.name}/candidate_rejections.jsonl", payload)
        store.write_json(
            f"{iter_dir.name}/pending_eval.json",
            _load_json_file(store.path / "pending_eval.json"),
        )
        # 合法候选逐个跑验证集，并把分数写回 leaderboard/history。
        for proposal in valid_proposals:
            candidate_path = Path(proposal.path)
            iter_candidate = iter_dir / Path(proposal.path).name
            if candidate_path.resolve() != iter_candidate.resolve():
                shutil.copyfile(candidate_path, iter_candidate)
            store.write_text(f"{iter_dir.name}/{proposal.name}_hypothesis.md", proposal.hypothesis)
            store.write_json(f"{iter_dir.name}/{proposal.name}_manifest.json", proposal.manifest)
            store.log_event(
                "candidate_eval_started",
                {"iteration": iteration, "candidate": proposal.name, "path": str(candidate_path)},
            )
            candidate_result = _evaluate(
                candidate_path,
                datasets,
                "val",
                solver,
                eval_config,
                max_workers=max_workers,
            )
            _stop_if_eval_failed(store, candidate_result, proposal.name)
            store.log_event(
                "candidate_eval_finished",
                {
                    "iteration": iteration,
                    "candidate": proposal.name,
                    "average": candidate_result.average,
                    "summary": candidate_result.summary,
                },
            )
            _print_score(proposal.name, "val", candidate_result, iteration=iteration)
            store.write_json(f"{iter_dir.name}/{proposal.name}_scores.json", _benchmark_to_dict(candidate_result))
            row = _leaderboard_row(
                proposal.name,
                str(candidate_path),
                candidate_result.average,
                _average_context_length(store.path, proposal.name, _model_short_name(raw), datasets, "logs"),
            )
            previous_best = best
            leaderboard.append(row)
            history.append({"name": proposal.name, "path": str(candidate_path), "result": candidate_result})
            best = max(leaderboard, key=lambda item: float(item["average"]))
            _write_evolution_summary(
                store.path,
                iteration,
                proposal.name,
                candidate_result,
                previous_best,
                str(proposal.manifest.get("axis", "?")),
                proposal.hypothesis,
                proposal.manifest.get("components", []),
            )
            store.append_jsonl(
                "evolution.jsonl",
                {"iteration": iteration, "candidate": proposal.name, "average": candidate_result.average},
            )
        # 本轮结束后刷新 leaderboard/frontier，供下一轮 proposer 继续分析。
        store.write_leaderboard(leaderboard)
        frontier = _write_official_frontier(store.path, leaderboard, _model_short_name(raw), datasets)
        store.save_frontier({"best": best, "leaderboard": leaderboard})
        store.log_event(
            "iteration_finished",
            {"iteration": iteration, "candidates": [proposal.name for proposal in valid_proposals], "best": best},
        )

    # 所有进化轮次结束后，只选择代表性系统进入最终 test。
    final_results: dict[str, Any] = {}
    final_systems = _final_test_systems(_baseline_names(raw), frontier, leaderboard)
    store.log_event("final_test_started", {"systems": list(final_systems), "split": "test"})
    for system_name, system_path in final_systems.items():
        final_result = _evaluate(
            system_path,
            datasets,
            "test",
            solver,
            eval_config,
            max_workers=max_workers,
        )
        _stop_if_eval_failed(store, final_result, str(system_path))
        final_results[system_name] = _benchmark_to_dict(final_result)
        _print_score(system_name, "test", final_result)
    store.write_json("final_test_scores.json", final_results)
    result = {"best": best, "final_test": final_results, "run_dir": str(store.path), "frontier": frontier}
    store.log_event("run_finished", {"best": best, "final_test_systems": list(final_results)})
    return result
# 从配置中解析各数据集的 train/val/test 路径。
def _dataset_specs(config: HarnessConfig) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for row in config.raw.get("datasets", []):
        train_path = Path(str(row["train"]))
        val_path = Path(str(row["val"]))
        test_path = Path(str(row["test"]))
        specs.append(
            DatasetSpec(
                name=str(row["name"]),
                train_path=str(train_path if train_path.is_absolute() else config.project_root / train_path),
                val_path=str(val_path if val_path.is_absolute() else config.project_root / val_path),
                test_path=str(test_path if test_path.is_absolute() else config.project_root / test_path),
            )
        )
    return specs


# 构造一条统一格式的 leaderboard 记录。
def _leaderboard_row(name: str, path: str, average: float, context_length: int = 0) -> dict[str, Any]:
    return {"name": name, "path": path, "average": average, "context_length": int(context_length)}
# 返回配置指定的 baseline 名称；未配置时使用默认 baseline。
def _baseline_names(config: Mapping[str, Any]) -> list[str]:
    candidate = dict(config.get("candidate", {}))
    if candidate.get("baselines"):
        return [Path(str(name)).stem for name in candidate["baselines"]]
    if candidate.get("baseline"):
        return [Path(str(candidate["baseline"])).stem]
    return ["no_memory", "fewshot_all"]


# 把 agent 名称或路径解析为绝对 Python 文件路径。
def _agent_path(project_root: Path, name_or_path: str | Path) -> Path:
    raw = Path(str(name_or_path))
    if raw.suffix == ".py":
        path = raw if raw.is_absolute() else project_root / raw
    elif raw.parts and raw.parts[0] == "agents":
        path = project_root / raw.with_suffix(".py")
    else:
        path = project_root / "agents" / f"{raw.name}.py"
    return path.resolve()


# 提取 solver 模型名中适合用作目录名的短名称。
def _model_short_name(config: Mapping[str, Any]) -> str:
    return str(dict(config.get("solver", {})).get("model", "model")).split("/")[-1].lower()


# 从配置中读取评测并发数。
def _max_workers(config: Mapping[str, Any]) -> int:
    evaluation = dict(config.get("evaluation", {}))
    inner_loop = dict(config.get("inner_loop", {}))
    benchmark = dict(config.get("benchmark", {}))
    return int(
        evaluation.get(
            "max_workers",
            inner_loop.get("max_workers", benchmark.get("concurrency", 1)),
        )
    )
# 构造传给 official-like evaluator 的评测配置。
def _official_eval_config(
    config: Mapping[str, Any],
    project_root: Path | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    evaluation = dict(config.get("evaluation", {}))
    dataset_cfg = dict(config.get("dataset", {}))
    inner_loop = dict(config.get("inner_loop", {}))
    run_cfg = dict(config.get("run", {}))
    mode = str(evaluation.get("mode", inner_loop.get("mode", "offline")))
    result = {
        "official": bool(evaluation.get("official", True)),
        "seed": int(evaluation.get("seed", inner_loop.get("seed", run_cfg.get("seed", 42)))),
        "mode": mode,
        "num_epochs": int(evaluation.get("num_epochs", inner_loop.get("num_epochs", 1))),
        "batch_size": int(evaluation.get("batch_size", inner_loop.get("batch_size", 1))),
        "combined_eval": bool(evaluation.get("combined_eval", False)),
        "dataset_limits": dataset_cfg.get("overrides", evaluation.get("dataset_limits", {})),
        "agent_protocol_only": bool(evaluation.get("agent_protocol_only", True)),
    }
    if project_root is not None:
        result["project_root"] = str(project_root)
    if run_dir is not None:
        result["output_root"] = str(run_dir)
        result["model_short"] = _model_short_name(config)
        result["write_artifacts"] = True
    return result
# 使用指定 evaluator 在某个 split 上评估一个 agent。
def _evaluate(
    system_path: str | Path,
    datasets: list[DatasetSpec],
    split: str,
    solver: Any,
    eval_config: Mapping[str, Any],
    max_workers: int,
) -> Any:
    if bool(eval_config.get("official", True)):
        return evaluate_system_official(
            system_path,
            datasets,
            split,
            solver,
            eval_config,
            max_workers=max_workers,
        )
    return evaluate_system(system_path, datasets, split, solver, {}, max_workers=max_workers)


# 判断是否启用 solver 预检查。
def _preflight_enabled(config: Mapping[str, Any]) -> bool:
    return bool(dict(config.get("preflight", {})).get("enabled", True))


# 把评测结果对象转换为可写入 JSON 的字典。
def _benchmark_to_dict(result: Any) -> dict[str, Any]:
    return {
        "system_name": result.system_name,
        "split": result.split,
        "per_dataset": result.per_dataset,
        "average": result.average,
        "summary": result.summary,
        "traces": result.traces,
        "errors": result.errors,
    }
# 在控制台打印一行简洁的系统得分。
def _print_score(name: str, split: str, result: Any, iteration: int | None = None) -> None:
    prefix = f"[iter {iteration:03d}] " if iteration is not None else ""
    per_dataset = " ".join(
        f"{dataset}={score:.1%}" for dataset, score in sorted(result.per_dataset.items())
    )
    summary = result.summary or {}
    correct = summary.get("correct", "?")
    total = summary.get("total", "?")
    line = (
        f"{prefix}{name} {split}: avg={result.average:.1%} "
        f"correct={correct}/{total}"
    )
    if per_dataset:
        line += f" | {per_dataset}"
    if summary.get("micro_f1") is not None:
        line += f" | micro_f1={float(summary['micro_f1']):.3f}"
    print(line, flush=True)
# 如果评测没有产生任何有效样本，则终止运行。
def _stop_if_eval_failed(store: RunStore, result: Any, name: str) -> None:
    if not getattr(result, "all_failed", False):
        return
    error_preview = result.errors[0]["error"] if result.errors else "unknown error"
    store.log_event(
        "eval_failed",
        {
            "system": name,
            "errors": len(result.errors),
            "first_error": error_preview,
        },
    )
    raise RuntimeError(
        f"Evaluation for {name} produced zero scored examples. First error: {error_preview}"
    )
# 生成 proposer prompt 中的简短运行状态说明。
def _render_official_task_prompt(run_dir: Path, iteration: int, num_datasets: int) -> str:
    pending_eval = run_dir / "pending_eval.json"
    frontier_val = run_dir / "frontier_val.json"
    evolution_summary = run_dir / "evolution_summary.jsonl"
    return (
        f"Run iteration {iteration} of the evolution loop. There are {num_datasets} datasets.\n\n"
        "## Run directories\n"
        f"All logs and results for this run are under `{run_dir}/`.\n"
        f"- `{evolution_summary}` — past results\n"
        f"- `{frontier_val}` — frontier\n"
        f"- `{run_dir / 'reports'}/` — post-eval reports\n"
        f"- Write pending_eval.json to: `{pending_eval}`"
    )
# 构造传给 DeepAgent proposer 的完整上下文 prompt。
def _build_official_like_proposer_context(
    project_root: Path,
    config: Mapping[str, Any],
    run_dir: Path,
    iteration: int,
    datasets: list[DatasetSpec],
    leaderboard: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> str:
    task_prompt = _render_official_task_prompt(run_dir, iteration, len(datasets))
    # candidate_contract 明确候选数量、写入位置和必须遵守的 agent 协议。
    candidate_contract = {
        "iteration": iteration,
        "write_candidates_under": str(project_root / config["candidate"]["output_dir"]),
        "pending_eval_path": str(run_dir / "pending_eval.json"),
        "required_protocol": "BaseAgentMemory subclass with predict(input), learn_from_batch(batch_results), get_state(), set_state()",
        "candidate_count": 2,
        "existing_agent_files": _existing_agent_files(project_root / config["candidate"]["output_dir"]),
        "allowed_writes": [
            str(project_root / config["candidate"]["output_dir"] / "*.py"),
            str(run_dir / "pending_eval.json"),
            str(run_dir / "reports" / "*.md"),
        ],
        "write_policy": (
            "Only write candidate files, pending_eval.json, and optional reports. "
            f"Candidate file names must be unique and should start with iter{iteration:03d}_ to avoid overwriting existing agents."
        ),
    }
    # frontier 优先读已有文件；不存在时用当前 leaderboard 的 best 兜底。
    frontier = _load_json_file(run_dir / "frontier_val.json")
    if not frontier:
        frontier = {"best": max(leaderboard, key=lambda row: float(row["average"])) if leaderboard else None}
    # 最终 prompt = 运行状态 + 历史成绩 + 源码摘要 + 候选输出约束。
    return (
        task_prompt
        + "\n\n"
        + build_proposer_prompt(
            frontier=frontier,
            leaderboard=leaderboard,
            recent_traces=_proposer_trace_excerpts(history),
            candidate_contract=candidate_contract,
            source_files=_collect_proposer_source_files(project_root, config, leaderboard),
            dataset_specs=_dataset_context(datasets),
            result_summaries=_result_summaries(history),
            artifact_paths=_artifact_paths(run_dir),
        )
    )


# 列出候选输出目录中已有的 Python agent 文件。
def _existing_agent_files(agent_dir: Path) -> list[str]:
    if not agent_dir.exists():
        return []
    return sorted(path.name for path in agent_dir.glob("*.py") if path.is_file())


# 过滤出通过验证的候选 proposal。
def _validate_proposals(proposals: list[Any]) -> list[Any]:
    return [row["proposal"] for row in _validate_proposals_with_reasons(proposals) if row["valid"]]
# 验证候选 proposal，并保留拒绝原因用于审计。
def _validate_proposals_with_reasons(proposals: list[Any]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for proposal in proposals:
        reasons: list[str] = []
        # 第一层：候选必须能被导入并实例化。
        import_error = _candidate_import_error(proposal.path)
        if import_error:
            reasons.append(import_error)
        # 第二层：候选源码不能包含数据集特化的运行时硬编码。
        reasons.extend(_dataset_specific_hint_reasons(Path(proposal.path)))
        decisions.append({"proposal": proposal, "valid": not reasons, "reasons": reasons})
    return decisions
# 尝试导入候选 agent；失败时返回错误说明。
def _candidate_import_error(path: str | Path) -> str | None:
    try:
        load_agent_memory(path, llm=lambda prompt: '{"final_answer": "ok"}')
    except Exception as exc:
        return f"import_failed:{exc.__class__.__name__}: {exc}"
    return None


# 保留旧版候选验证逻辑以兼容历史代码。
def _validate_proposals_legacy(proposals: list[Any]) -> list[Any]:
    valid = []
    for proposal in proposals:
        system = _try_import_candidate(proposal.path)
        if system is not None and not _has_dataset_specific_hints(Path(proposal.path)):
            valid.append(proposal)
    return valid


# 安全尝试导入并实例化候选 agent。
def _try_import_candidate(path: str | Path) -> Any | None:
    try:
        return load_agent_memory(path, llm=lambda prompt: '{"final_answer": "ok"}')
    except Exception:
        return None


# 写入每个数据集的分数摘要和逐样本评测日志。
def _write_official_result_files(
    run_dir: Path,
    memory_name: str,
    model_short: str,
    result: Any,
) -> None:
    for dataset_name, score in result.per_dataset.items():
        # 每个数据集单独写聚合分数和逐样本日志。
        rows = [row for row in result.traces if row.get("dataset") == dataset_name]
        correct = sum(1 for row in rows if row.get("was_correct"))
        result_dir = run_dir / "logs" / dataset_name / memory_name / model_short
        result_dir.mkdir(parents=True, exist_ok=True)
        # split.json 存聚合指标；log.jsonl 存每条样本的输入、输出和判分细节。
        payload = {
            "accuracy": score,
            "correct": correct,
            "total": len(rows),
            "memory_context_chars": _average_prompt_chars(rows),
            "system_name": memory_name,
            "dataset": dataset_name,
            "model": model_short,
            "split": result.split,
        }
        (result_dir / f"{result.split}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (result_dir / "log.jsonl").open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                handle.write(
                    json.dumps(
                        {
                            "type": "eval_step",
                            "step": index,
                            "input_preview": _prompt_to_text(row.get("prompt", []))[:200],
                            "pred": row.get("prediction"),
                            "tgt": row.get("target") or row.get("label"),
                            "ok": row.get("was_correct"),
                            "metrics": row.get("metrics", {}),
                            "prompt_len": len(_prompt_to_text(row.get("prompt", []))),
                            "prompt_text": _prompt_to_text(row.get("prompt", [])),
                            "raw_output": row.get("raw_output", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
# 写入 Pareto frontier 和各数据集验证集最优系统。
def _write_official_frontier(
    run_dir: Path,
    leaderboard: list[dict[str, Any]],
    model_short: str,
    datasets: list[DatasetSpec],
) -> dict[str, Any]:
    frontier: dict[str, Any] = {}
    # 全局 Pareto 同时考虑验证准确率和上下文长度。
    pareto = _compute_pareto_frontier(leaderboard)
    frontier["_pareto"] = pareto

    # 每个数据集单独选 val 最优系统；同分时偏向上下文更短的系统。
    for dataset in datasets:
        best_entry = None
        for row in leaderboard:
            path = run_dir / "logs" / dataset.name / row["name"] / model_short / "val.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            entry = {
                "best_system": row["name"],
                "best_path": row["path"],
                "val_accuracy": data.get("accuracy", 0.0),
                "ctx_len": data.get("memory_context_chars", 0),
                "model": model_short,
            }
            if best_entry is None or (
                float(entry["val_accuracy"]),
                -int(entry["ctx_len"]),
            ) > (
                float(best_entry["val_accuracy"]),
                -int(best_entry["ctx_len"]),
            ):
                best_entry = entry
        if best_entry:
            frontier[dataset.name] = best_entry
    (run_dir / "frontier_val.json").write_text(
        json.dumps(frontier, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return frontier
# 向 evolution_summary.jsonl 追加一条进化记录。
def _write_evolution_summary(
    run_dir: Path,
    iteration: int,
    system_name: str,
    result: Any,
    previous_best: Mapping[str, Any],
    axis: str,
    hypothesis: str,
    components: Any = None,
) -> None:
    previous = float(previous_best.get("average", 0.0)) if previous_best else 0.0
    avg_val = result.average * 100
    row = {
        "iteration": iteration,
        "system": system_name,
        "avg_val": round(avg_val, 1),
        "axis": axis,
        "hypothesis": hypothesis,
        "delta": round(avg_val - previous * 100, 1) if previous else None,
        "outcome": f"{avg_val:.1f}% ({avg_val - previous * 100:+.1f})" if previous else f"{avg_val:.1f}%",
    }
    if components:
        row["components"] = components
    with (run_dir / "evolution_summary.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# 计算一组 trace 记录中的平均 prompt 字符数。
def _average_prompt_chars(rows: list[dict[str, Any]]) -> int:
    lengths = [len(_prompt_to_text(row.get("prompt", []))) for row in rows]
    return int(sum(lengths) / len(lengths)) if lengths else 0


# 读取并平均某个系统在各数据集上的验证集上下文长度。
def _average_context_length(
    run_dir: Path,
    system_name: str,
    model_short: str,
    datasets: list[DatasetSpec],
    base: str = "logs",
) -> int:
    values: list[int] = []
    for dataset in datasets:
        path = run_dir / base / dataset.name / system_name / model_short / "val.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        value = int(data.get("memory_context_chars", 0) or 0)
        if value > 0:
            values.append(value)
    return int(sum(values) / len(values)) if values else 0
# 保留没有被“更高准确率且更短上下文”支配的系统。
def _compute_pareto_frontier(leaderboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 先按准确率降序、上下文长度升序排序。
    sorted_rows = sorted(
        leaderboard,
        key=lambda row: (-float(row.get("average", 0.0)), int(row.get("context_length", 0) or 0)),
    )
    pareto: list[dict[str, Any]] = []
    min_context = float("inf")
    # 只保留上下文长度刷新当前最小值的系统，过滤被支配方案。
    for row in sorted_rows:
        context_length = int(row.get("context_length", 0) or 0)
        if context_length <= min_context:
            pareto.append(
                {
                    "system": row["name"],
                    "path": row["path"],
                    "val_accuracy": float(row["average"]),
                    "ctx_len": context_length,
                }
            )
            min_context = context_length
    return pareto
# 选择 baseline、Pareto 系统和各数据集最优系统进入 test。
def _final_test_systems(
    baseline_names: list[str],
    frontier: Mapping[str, Any],
    leaderboard: list[dict[str, Any]],
) -> dict[str, Path]:
    # test 集只跑 baseline、Pareto 系统和各数据集 best，避免无意义全量重跑。
    paths_by_name = {str(row["name"]): Path(str(row["path"])) for row in leaderboard}
    selected: dict[str, Path] = {}
    for name in baseline_names:
        if name in paths_by_name:
            selected[name] = paths_by_name[name]
    for entry in frontier.get("_pareto", []):
        if isinstance(entry, Mapping):
            name = str(entry.get("system", ""))
            if name in paths_by_name:
                selected[name] = paths_by_name[name]
    for value in frontier.values():
        if isinstance(value, Mapping) and value.get("best_system"):
            name = str(value["best_system"])
            if name in paths_by_name:
                selected[name] = paths_by_name[name]
    return selected


# 旧版检测：粗略判断候选是否包含数据集特化硬编码。
def _has_dataset_specific_hints(path: Path) -> bool:
    if not path.exists():
        return True
    lowered = path.read_text(encoding="utf-8", errors="replace").lower()
    forbidden = [
        "uspto",
        "symptom2disease",
        "lawbench",
        "crime_prediction",
        "symptom_diagnosis",
        "罪名映射",
    ]
    return any(token.lower() in lowered for token in forbidden)


# 读取 JSON 文件；文件不存在时返回空字典。
def _load_json_file(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
# 构造 DeepAgent proposer 调用模型时使用的环境变量。
def _proposer_env(config: Mapping[str, Any]) -> dict[str, str]:
    base_url_env = str(config.get("base_url_env", "ANTHROPIC_BASE_URL"))
    auth_token_env = str(config.get("auth_token_env", "ANTHROPIC_AUTH_TOKEN"))
    timeout_env = str(config.get("timeout_ms_env", "API_TIMEOUT_MS"))
    env = dict(os.environ)
    env["ANTHROPIC_BASE_URL"] = os.environ.get(
        base_url_env,
        str(config.get("base_url") or config.get("default_base_url") or "https://doro.lol"),
    )
    env["ANTHROPIC_AUTH_TOKEN"] = os.environ.get(auth_token_env, str(config.get("auth_token", "")))
    if config.get("timeout_ms"):
        env[timeout_env] = str(config["timeout_ms"])
    return env
# 生成供 proposer 使用的数据集路径摘要。
def _dataset_context(datasets: list[DatasetSpec]) -> list[dict[str, str]]:
    return [
        {
            "name": dataset.name,
            "train_path": dataset.train_path,
            "val_path": dataset.val_path,
            "test_path": dataset.test_path,
        }
        for dataset in datasets
    ]
# 生成历史评测结果的简要索引信息。
def _result_summaries(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in history:
        result = item["result"]
        rows.append(
            {
                "name": item["name"],
                "path": item["path"],
                "split": result.split,
                "log_hint": "logs/<dataset>/<memory>/<model>/log.jsonl",
                "val_hint": "logs/<dataset>/<memory>/<model>/val.json",
            }
        )
    return rows
# 生成 proposer 应检查的评测 trace 文件提示。
def _proposer_trace_excerpts(
    history: list[dict[str, Any]],
    max_per_result: int = 12,
    prompt_chars: int = 3000,
    raw_chars: int = 2000,
) -> list[dict[str, Any]]:
    excerpts: list[dict[str, Any]] = []
    for item in history:
        result = item["result"]
        seen_datasets: set[str] = set()
        for row in result.traces:
            dataset = str(row.get("dataset", "unknown"))
            if dataset in seen_datasets:
                continue
            seen_datasets.add(dataset)
            excerpts.append(
                {
                    "system": item["name"],
                    "dataset": dataset,
                    "read_trace_file": f"logs/{dataset}/{item['name']}/<model>/log.jsonl",
                }
            )
    return excerpts


# 压缩 retrieved examples，保留少量关键信息。
def _compact_retrieved(items: Any, max_items: int = 5) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return compact
    for item in items[:max_items]:
        if not isinstance(item, Mapping):
            continue
        compact.append(
            {
                "id": item.get("id"),
                "label": item.get("label"),
                "score": item.get("score"),
                "text_excerpt": _truncate_text(str(item.get("text", "")), 600),
            }
        )
    return compact


# 把 chat messages 格式的 prompt 渲染成纯文本。
def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            if isinstance(message, Mapping):
                parts.append(f"{message.get('role', 'unknown')}: {message.get('content', '')}")
            else:
                parts.append(str(message))
        return "\n\n".join(parts)
    return str(prompt)
# 列出当前 run 目录下可供 proposer 检查的产物文件。
def _artifact_paths(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    paths: list[str] = []
    for pattern in [
        "config.yaml",
        "baseline_scores.json",
        "leaderboard.csv",
        "frontier.json",
        "evolution.jsonl",
        "iter_*/scores.json",
        "iter_*/candidate.py",
        "iter_*/manifest.json",
        "iter_*/proposer_prompt.md",
        "iter_*/proposer_response.md",
        "iter_*/proposer_messages.json",
        "iter_*/proposer_tool_calls.jsonl",
    ]:
        paths.extend(str(path) for path in sorted(run_dir.glob(pattern)) if path.exists())
    return paths
# 写入 proposer 的任务、候选输出和工具调用审计记录。
def _write_proposer_audit(
    run_dir: Path,
    iter_dir: Path,
    prompt_text: str,
    proposals: list[Any],
    trace_path: Path,
    config: Mapping[str, Any],
) -> Path:
    # 读取 proposer 的 backend trace，用于复盘它实际读写/执行了什么。
    trace_events: list[dict[str, Any]] = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            try:
                trace_events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # 只保留真实工具调用事件，作为审计记录的核心内容。
    inspected = []
    for event in trace_events:
        if event.get("event") != "backend_call":
            continue
        inspected.append(
            {
                "tool": event.get("tool"),
                "file_path": event.get("file_path") or event.get("args", {}).get("file_path"),
                "path": event.get("path") or event.get("args", {}).get("path"),
                "command": event.get("command") or event.get("args", {}).get("command"),
                "pattern": event.get("pattern") or event.get("args", {}).get("pattern"),
                "result_preview": event.get("result_preview", ""),
            }
        )

    # 汇总本轮任务要求、候选文件、实际检查行为和配置快照。
    audit = {
        "iteration": int(iter_dir.name.split("_")[-1]),
        "workflow": {
            "post_eval_reports_checked": "reports" in prompt_text,
            "must_read_state_files": [
                "evolution_summary.jsonl",
                "frontier_val.json",
                "config.yaml",
                "logs/<dataset>/<agent>/<model>/log.jsonl",
            ],
            "should_inspect": [
                "leaderboard.csv",
                "relevant val.json",
                "current baseline/top/Pareto source files",
            ],
            "prototype_required": "Prototype — MANDATORY" in prompt_text,
        },
        "candidate_files": [proposal.path for proposal in proposals],
        "inspected_actions": inspected,
        "inspected_state_files": [
            entry for entry in inspected if isinstance(entry.get("file_path"), str) or isinstance(entry.get("path"), str)
        ],
        "blocked_paths": [
            path
            for path in [
                str(run_dir.parent / "runs"),
                str(run_dir.parent / "tc_deepagent_metaharness_backup_pre_clean_"),
            ]
            if path
        ],
        "config_snapshot": {
            "run_name": dict(config.get("run", {})).get("name"),
            "rounds": dict(config.get("run", {})).get("rounds"),
            "proposer_model": dict(config.get("proposer", {})).get("model"),
        },
    }
    audit_path = iter_dir / "proposer_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_path
# 收集放入 proposer 上下文的源码片段。
def _collect_proposer_source_files(
    project_root: Path,
    config: Mapping[str, Any],
    leaderboard: list[dict[str, Any]],
) -> dict[str, str]:
    context_cfg = dict(config.get("proposer_context", {}))
    max_file_chars = int(context_cfg.get("max_file_chars", 80000))
    max_total_chars = int(context_cfg.get("max_total_source_chars", 320000))

    # 默认提供最小必要源码：协议、LLM 封装和 baseline agent。
    paths = [
        project_root / "harness" / "__init__.py",
        project_root / "harness" / "agent_protocol.py",
        project_root / "harness" / "llm.py",
        project_root / "agents" / "no_memory.py",
        project_root / "agents" / "fewshot_all.py",
    ]
    # 加入已上榜系统源码，让 proposer 可以继承或局部改造已有好方案。
    for row in leaderboard:
        path = Path(str(row.get("path", "")))
        if path.exists():
            paths.append(path)

    paths.extend(_configured_external_source_files(context_cfg))
    if bool(context_cfg.get("include_collected_harness_sources", False)):
        paths.extend(_discover_collected_harness_sources(context_cfg))

    source_files: dict[str, str] = {}
    total_chars = 0
    seen: set[Path] = set()
    # 去重、读取并截断源码，控制 proposer prompt 的总长度。
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists() or not resolved.is_file():
            continue
        seen.add(resolved)
        label = _source_label(resolved, project_root)
        if total_chars >= max_total_chars:
            content = _omitted_source_preview(resolved)
        elif resolved.name == "SKILL.md" and ".claude/skills/meta-harness" in str(resolved):
            content = _omitted_source_preview(resolved)
        else:
            content = _read_text_for_context(resolved, max_file_chars)
            if total_chars + len(content) > max_total_chars:
                remaining = max_total_chars - total_chars
                content = _truncate_text(content, remaining)
            total_chars += len(content)
        source_files[label] = content
    return source_files
# 读取配置中显式指定的额外源码文件。
def _configured_external_source_files(context_cfg: Mapping[str, Any]) -> list[Path]:
    files: list[Path] = []
    for value in context_cfg.get("external_source_files", []):
        files.append(Path(str(value)))
    for value in context_cfg.get("collected_source_files", []):
        files.append(Path(str(value)))
    if bool(context_cfg.get("include_reference_source", False)):
        for value in context_cfg.get("reference_source_files", []):
            files.append(Path(str(value)))
    return files
# 从配置的外部目录中发现可参考的 harness 源码。
def _discover_collected_harness_sources(context_cfg: Mapping[str, Any]) -> list[Path]:
    roots = [Path(str(root)) for root in context_cfg.get("harness_source_roots", [])]
    globs = list(context_cfg.get("harness_source_globs", [])) or [
        "memory_system.py",
        "llm.py",
        "agents/*.py",
        "systems/*.py",
        ".claude/skills/meta-harness/SKILL.md",
    ]
    max_files = int(context_cfg.get("max_collected_source_files", 500))
    discovered: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for pattern in globs:
            for path in sorted(root.glob(str(pattern))):
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                discovered.append(resolved)
                if len(discovered) >= max_files:
                    return discovered
    return discovered


# 把源码路径转换为 prompt 中使用的稳定标签。
def _source_label(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)
# 读取文本文件用于 proposer 上下文，并兼容编码异常。
def _read_text_for_context(path: Path, max_chars: int) -> str:
    try:
        return _truncate_text(path.read_text(encoding="utf-8"), max_chars)
    except UnicodeDecodeError:
        return _truncate_text(path.read_text(encoding="utf-8", errors="replace"), max_chars)
# 返回源码省略提示，并说明可直接检查的完整文件路径。
def _omitted_source_preview(path: Path) -> str:
    return (
        f"[inline preview omitted because max_total_source_chars was reached]\n"
        f"DeepAgent may inspect the full file directly at: {path}\n"
    )
# 按字符数限制截断长文本，保留开头和结尾。
def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n\n... [TRUNCATED] ...\n\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    tail_text = text[-tail:] if tail else ""
    return text[:head] + marker + tail_text


# 判断候选源码中是否包含禁止的数据集特化字符串。
def _has_dataset_specific_hints(path: Path) -> bool:
    return bool(_dataset_specific_hint_reasons(path))
# 找出运行时字符串中疑似数据集特化硬编码的原因。
def _dataset_specific_hint_reasons(path: Path) -> list[str]:
    if not path.exists():
        return ["missing_file"]
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        # 用 AST 检查运行时字符串，避免把普通注释误判为硬编码。
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax_error:{exc}"]
    _annotate_ast_parents(tree)
    # 这些数据集标识不允许出现在候选的运行时字符串中。
    forbidden = (
        "uspto",
        "symptom2disease",
        "lawbench",
        "crime_prediction",
        "symptom_diagnosis",
    )
    reasons: list[str] = []
    # 只检查字符串常量，并跳过模块/类/函数 docstring。
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if _is_docstring_constant(node):
            continue
        lowered = node.value.lower()
        for token in forbidden:
            if token in lowered:
                reasons.append(f"runtime_string_contains:{token}:line{getattr(node, 'lineno', '?')}")
    return sorted(set(reasons))


# 为 AST 节点补充父节点引用，方便后续判断 docstring。
def _annotate_ast_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "_parent", parent)


# 判断一个字符串常量是否是模块、类或函数 docstring。
def _is_docstring_constant(node: ast.Constant) -> bool:
    parent = getattr(node, "_parent", None)
    if not isinstance(parent, ast.Expr) or parent.value is not node:
        return False
    grandparent = getattr(parent, "_parent", None)
    body = getattr(grandparent, "body", None)
    return bool(body and body[0] is parent)
