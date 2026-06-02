"""Retest generated candidate agents on the official-like test split."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.eval import DatasetSpec
from harness.llm import build_solver_client, preflight_solver
from harness.loop import (
    HarnessConfig,
    _benchmark_to_dict,
    _dataset_specs,
    _model_short_name,
    _official_eval_config,
    load_config,
)
from harness.official_eval import evaluate_system_official


DEFAULT_CANDIDATES = [
    "iter001_label_coverage_bm25",
    "iter001_confusion_aware_bm25",
    "iter002_label_grouped_prompt",
    "iter002_error_weighted_bm25",
    "iter003_contrastive_retrieval",
    "iter003_two_pass_verify",
    "iter004_error_notes_retrieval",
    "iter004_knn_label_focus",
    "iter005_confusion_verify",
    "iter005_multi_strategy_vote",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/oss120b_deepagent_opus46_5rounds_clean_official_workflow.yaml")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--candidates", nargs="*", default=DEFAULT_CANDIDATES)
    args = parser.parse_args()

    project_root = Path.cwd()
    config_path = project_root / args.config
    source_run = project_root / args.source_run
    run_dir = project_root / "runs" / args.run_name
    if run_dir.exists():
        raise SystemExit(f"Refusing to reuse existing run dir: {run_dir}")
    run_dir.mkdir(parents=True)

    raw = load_config(config_path)
    raw.setdefault("evaluation", {})["max_workers"] = args.max_workers
    model_short = _model_short_name(raw)
    datasets = _dataset_specs(HarnessConfig(raw=raw, project_root=project_root))
    live_log = run_dir / "live.log"
    scores_path = run_dir / "candidate_test_scores.json"
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    _log(live_log, "retest_started", {"source_run": str(source_run), "candidates": args.candidates, "max_workers": args.max_workers})

    preflight = preflight_solver(raw)
    _log(live_log, "solver_preflight_finished", preflight)
    if not preflight["ok"]:
        raise SystemExit(f"Solver preflight failed: {preflight}")

    _copy_memory_snapshots(source_run, run_dir, args.candidates, datasets, model_short)
    solver = build_solver_client(raw)
    eval_config = _official_eval_config(raw, project_root=project_root, run_dir=run_dir)
    eval_config["write_artifacts"] = True
    results: dict[str, Any] = {}

    for name in args.candidates:
        system_path = project_root / "agents" / f"{name}.py"
        _log(live_log, "candidate_test_started", {"candidate": name, "path": str(system_path)})
        result = evaluate_system_official(
            system_path,
            datasets,
            "test",
            solver,
            eval_config,
            max_workers=args.max_workers,
        )
        payload = _benchmark_to_dict(result)
        results[name] = payload
        scores_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        _print_score(name, result)
        _log(
            live_log,
            "candidate_test_finished",
            {
                "candidate": name,
                "average": result.average,
                "summary": result.summary,
                "errors": result.errors,
                "error_count": result.summary.get("error_count", 0) if result.summary else 0,
            },
        )
        if result.errors or (result.summary and result.summary.get("error_count", 0)):
            _log(live_log, "retest_stopped_on_errors", {"candidate": name})
            return 2

    _log(live_log, "retest_finished", {"candidates": list(results)})
    return 0


def _copy_memory_snapshots(
    source_run: Path,
    run_dir: Path,
    candidates: list[str],
    datasets: list[DatasetSpec],
    model_short: str,
) -> None:
    for name in candidates:
        for dataset in datasets:
            src = source_run / "logs" / dataset.name / name / model_short / "memory.json"
            if not src.exists():
                raise FileNotFoundError(f"Missing source memory snapshot: {src}")
            dst = run_dir / "logs" / dataset.name / name / model_short / "memory.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _log(path: Path, message: str, data: dict[str, Any]) -> None:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "data": data,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_score(name: str, result: Any) -> None:
    per_dataset = " ".join(f"{dataset}={score:.1%}" for dataset, score in sorted(result.per_dataset.items()))
    summary = result.summary or {}
    correct = summary.get("correct", "?")
    total = summary.get("total", "?")
    micro_f1 = summary.get("micro_f1")
    line = f"{name} test: avg={result.average:.1%} correct={correct}/{total}"
    if per_dataset:
        line += f" | {per_dataset}"
    if micro_f1 is not None:
        line += f" | micro_f1={micro_f1:.3f}"
    print(line, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
