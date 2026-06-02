"""Continue an existing official-like run for later proposer rounds."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.eval import DatasetSpec
from harness.llm import build_solver_client, preflight_solver
from harness.loop import (
    HarnessConfig,
    _average_context_length,
    _benchmark_to_dict,
    _build_official_like_proposer_context,
    _dataset_specs,
    _evaluate,
    _leaderboard_row,
    _load_json_file,
    _max_workers,
    _model_short_name,
    _official_eval_config,
    _print_score,
    _proposer_env,
    _stop_if_eval_failed,
    _validate_proposals_with_reasons,
    _write_evolution_summary,
    _write_official_frontier,
    _write_proposer_audit,
    load_config,
)
from harness.proposer import DeepAgentProposer
from harness.store import RunStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oss120b_deepagent_opus46_5rounds_clean_official_workflow.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--start-iteration", type=int, default=6)
    parser.add_argument("--end-iteration", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--allow-existing-iter", action="store_true")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    config_path = _resolve_path(project_root, args.config)
    run_dir = _resolve_run_dir(project_root, args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    if args.start_iteration > args.end_iteration:
        raise SystemExit("--start-iteration must be <= --end-iteration")

    raw = load_config(config_path)
    raw.setdefault("run", {})["rounds"] = args.end_iteration
    raw["run"]["continued_from_iteration"] = args.start_iteration - 1
    if args.max_workers is not None:
        raw.setdefault("evaluation", {})["max_workers"] = args.max_workers

    store = RunStore(run_dir.parent, run_dir.name)
    store.create()
    _snapshot_continuation_config(store, raw, args.start_iteration, args.end_iteration)
    store.log_event(
        "extension_started",
        {
            "config_path": str(config_path),
            "start_iteration": args.start_iteration,
            "end_iteration": args.end_iteration,
            "test_scope": "new_candidates_only",
        },
    )

    if not args.skip_preflight:
        store.log_event("solver_preflight_started", {})
        preflight = preflight_solver(raw)
        if not preflight["ok"]:
            store.log_event("solver_preflight_failed", preflight)
            raise RuntimeError(f"Solver preflight failed: {preflight['error']}")
        store.log_event("solver_preflight_finished", preflight)

    harness_config = HarnessConfig(raw=raw, project_root=project_root)
    datasets = _dataset_specs(harness_config)
    model_short = _model_short_name(raw)
    max_workers = _max_workers(raw)
    solver = build_solver_client(raw)
    eval_config = _official_eval_config(raw, project_root=project_root, run_dir=store.path)
    leaderboard = _load_leaderboard(store.path)
    history = _history_from_existing_val_files(store.path, leaderboard, datasets, model_short)
    if not leaderboard:
        raise SystemExit(f"No leaderboard rows found under {store.path}")
    best = max(leaderboard, key=lambda item: float(item["average"]))
    frontier = _write_official_frontier(store.path, leaderboard, model_short, datasets)
    store.save_frontier({"best": best, "leaderboard": leaderboard})

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

    new_candidates: list[dict[str, str]] = []
    for iteration in range(args.start_iteration, args.end_iteration + 1):
        iter_dir = store.path / f"iter_{iteration:03d}"
        if iter_dir.exists() and not args.allow_existing_iter:
            raise SystemExit(f"Refusing to reuse existing iteration directory: {iter_dir}")
        iter_dir.mkdir(parents=True, exist_ok=True)
        store.log_event(
            "iteration_started",
            {"iteration": iteration, "end_iteration": args.end_iteration, "mode": "extension"},
        )
        reused_pending = _find_existing_pending_eval(store.path, iter_dir, iteration) if args.allow_existing_iter else None
        proposals = proposer.load_pending_eval(reused_pending, iteration) if reused_pending is not None else None
        if proposals is not None and _iteration_has_scores(iter_dir, proposals):
            _add_candidates_once(new_candidates, proposals)
            store.log_event(
                "iteration_reused",
                {
                    "iteration": iteration,
                    "reason": "existing_val_scores_found",
                    "candidates": [proposal.name for proposal in proposals],
                },
            )
            store.log_event(
                "iteration_finished",
                {"iteration": iteration, "candidates": [proposal.name for proposal in proposals], "best": best, "reused": True},
            )
            continue

        task_prompt = _load_or_build_task_prompt(
            project_root=project_root,
            store=store,
            iter_dir=iter_dir,
            config=raw,
            iteration=iteration,
            datasets=datasets,
            leaderboard=leaderboard,
            history=history,
        )
        if proposals is None:
            store.log_event("proposer_started", {"iteration": iteration, "model": proposer.model})
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
        else:
            if reused_pending != store.path / "pending_eval.json":
                shutil.copyfile(reused_pending, store.path / "pending_eval.json")
            store.log_event(
                "proposer_reused",
                {
                    "iteration": iteration,
                    "pending_eval": str(reused_pending),
                    "candidates": [proposal.name for proposal in proposals],
                },
            )
            if (iter_dir / "proposer_tool_calls.jsonl").exists() and not (iter_dir / "proposer_audit.json").exists():
                _write_proposer_audit(
                    store.path,
                    iter_dir,
                    task_prompt,
                    proposals,
                    iter_dir / "proposer_tool_calls.jsonl",
                    raw,
                )

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
        store.write_json(f"{iter_dir.name}/pending_eval.json", _load_json_file(store.path / "pending_eval.json"))

        for proposal in valid_proposals:
            candidate_path = Path(proposal.path)
            iter_candidate = iter_dir / candidate_path.name
            if candidate_path.resolve() != iter_candidate.resolve():
                shutil.copyfile(candidate_path, iter_candidate)
            store.write_text(f"{iter_dir.name}/{proposal.name}_hypothesis.md", proposal.hypothesis)
            store.write_json(f"{iter_dir.name}/{proposal.name}_manifest.json", proposal.manifest)
            store.log_event(
                "candidate_eval_started",
                {"iteration": iteration, "candidate": proposal.name, "path": str(candidate_path), "split": "val"},
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
                _average_context_length(store.path, proposal.name, model_short, datasets, "logs"),
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
            store.append_jsonl(
                f"extension_{args.start_iteration:03d}_{args.end_iteration:03d}_new_candidates.jsonl",
                {"iteration": iteration, "candidate": proposal.name, "path": str(candidate_path)},
            )
            _add_candidates_once(new_candidates, [proposal])

        store.write_leaderboard(leaderboard)
        frontier = _write_official_frontier(store.path, leaderboard, model_short, datasets)
        store.save_frontier({"best": best, "leaderboard": leaderboard})
        store.log_event(
            "iteration_finished",
            {"iteration": iteration, "candidates": [proposal.name for proposal in valid_proposals], "best": best},
        )

    test_results = _test_new_candidates_only(
        store=store,
        datasets=datasets,
        solver=solver,
        eval_config=eval_config,
        max_workers=max_workers,
        candidates=new_candidates,
        start_iteration=args.start_iteration,
        end_iteration=args.end_iteration,
    )
    store.log_event(
        "extension_finished",
        {
            "start_iteration": args.start_iteration,
            "end_iteration": args.end_iteration,
            "new_candidates": [row["name"] for row in new_candidates],
            "tested_candidates": list(test_results),
        },
    )
    return 0


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _resolve_run_dir(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "runs":
        return project_root / path
    return project_root / "runs" / path


def _snapshot_continuation_config(store: RunStore, raw: Mapping[str, Any], start: int, end: int) -> None:
    existing = store.path / "config.yaml"
    backup = store.path / f"config_before_extension_{start:03d}_{end:03d}.yaml"
    if existing.exists() and not backup.exists():
        shutil.copy2(existing, backup)
    store.write_text("config.yaml", yaml.safe_dump(dict(raw), sort_keys=False))
    store.write_json(
        f"extension_{start:03d}_{end:03d}_metadata.json",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "start_iteration": start,
            "end_iteration": end,
            "test_scope": "new_candidates_only",
        },
    )


def _load_leaderboard(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "leaderboard.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "name": row["name"],
                    "path": row["path"],
                    "average": float(row["average"]),
                    "context_length": int(float(row.get("context_length") or 0)),
                }
            )
    return rows


def _history_from_existing_val_files(
    run_dir: Path,
    leaderboard: list[dict[str, Any]],
    datasets: list[DatasetSpec],
    model_short: str,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for row in leaderboard:
        per_dataset: dict[str, float] = {}
        traces: list[dict[str, Any]] = []
        for dataset in datasets:
            val_path = run_dir / "logs" / dataset.name / row["name"] / model_short / "val.json"
            if not val_path.exists():
                continue
            data = json.loads(val_path.read_text(encoding="utf-8"))
            per_dataset[dataset.name] = float(data.get("accuracy", 0.0))
            traces.append({"dataset": dataset.name})
        history.append(
            {
                "name": row["name"],
                "path": row["path"],
                "result": SimpleNamespace(
                    split="val",
                    per_dataset=per_dataset,
                    average=float(row["average"]),
                    summary={},
                    traces=traces,
                    errors=[],
                ),
            }
        )
    return history


def _load_or_build_task_prompt(
    project_root: Path,
    store: RunStore,
    iter_dir: Path,
    config: Mapping[str, Any],
    iteration: int,
    datasets: list[DatasetSpec],
    leaderboard: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> str:
    prompt_path = iter_dir / "proposer_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    task_prompt = _build_official_like_proposer_context(
        project_root=project_root,
        config=config,
        run_dir=store.path,
        iteration=iteration,
        datasets=datasets,
        leaderboard=leaderboard,
        history=history,
    )
    store.write_text(f"{iter_dir.name}/proposer_prompt.md", task_prompt)
    return task_prompt


def _find_existing_pending_eval(run_dir: Path, iter_dir: Path, iteration: int) -> Path | None:
    for path in [iter_dir / "pending_eval.json", run_dir / "pending_eval.json"]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if int(data.get("iteration", -1)) == iteration:
            return path
    return None


def _iteration_has_scores(iter_dir: Path, proposals: list[Any]) -> bool:
    return bool(proposals) and all((iter_dir / f"{proposal.name}_scores.json").exists() for proposal in proposals)


def _add_candidates_once(target: list[dict[str, str]], proposals: list[Any]) -> None:
    seen = {row["name"] for row in target}
    for proposal in proposals:
        if proposal.name in seen:
            continue
        target.append({"name": proposal.name, "path": str(Path(proposal.path))})
        seen.add(proposal.name)


def _test_new_candidates_only(
    store: RunStore,
    datasets: list[DatasetSpec],
    solver: Any,
    eval_config: Mapping[str, Any],
    max_workers: int,
    candidates: list[dict[str, str]],
    start_iteration: int,
    end_iteration: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    json_name = f"extension_{start_iteration:03d}_{end_iteration:03d}_test_scores.json"
    csv_name = f"extension_{start_iteration:03d}_{end_iteration:03d}_test_scores.csv"
    store.log_event(
        "extension_test_started",
        {"systems": [row["name"] for row in candidates], "split": "test", "test_scope": "new_candidates_only"},
    )
    for row in candidates:
        system_name = row["name"]
        system_path = Path(row["path"])
        store.log_event(
            "candidate_test_started",
            {"candidate": system_name, "path": str(system_path), "split": "test"},
        )
        result = _evaluate(system_path, datasets, "test", solver, eval_config, max_workers=max_workers)
        _stop_if_eval_failed(store, result, system_name)
        payload = _benchmark_to_dict(result)
        results[system_name] = payload
        store.write_json(json_name, results)
        _write_test_csv(store.path / csv_name, results)
        _print_score(system_name, "test", result)
        store.log_event(
            "candidate_test_finished",
            {
                "candidate": system_name,
                "average": result.average,
                "summary": result.summary,
                "errors": result.errors,
            },
        )
    store.log_event("extension_test_finished", {"systems": list(results)})
    return results


def _write_test_csv(path: Path, results: Mapping[str, Any]) -> None:
    datasets = sorted({name for result in results.values() for name in result.get("per_dataset", {})})
    fieldnames = ["candidate", "macro_avg", "correct", "total", *datasets, "micro_f1"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate, result in sorted(
            results.items(),
            key=lambda item: float(item[1].get("average", 0.0)),
            reverse=True,
        ):
            summary = result.get("summary", {})
            row: dict[str, Any] = {
                "candidate": candidate,
                "macro_avg": result.get("average", 0.0),
                "correct": summary.get("correct", ""),
                "total": summary.get("total", ""),
                "micro_f1": summary.get("micro_f1", ""),
            }
            for dataset in datasets:
                row[dataset] = result.get("per_dataset", {}).get(dataset, "")
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
