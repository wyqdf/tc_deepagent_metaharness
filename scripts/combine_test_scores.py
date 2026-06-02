"""Combine baseline and candidate test scores into one CSV.

This script only reads existing run artifacts. It does not call any models or
run benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DATASETS = ["LawBench", "Symptom2Disease", "USPTO"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_leaderboard(path: Path) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["name"]] = {
                "val_macro_avg": float(row["average"]),
                "context_length": int(float(row["context_length"])),
            }
    return rows


def summarize_result_files(run_dir: Path, name: str, model: str) -> dict[str, Any]:
    per_dataset: dict[str, float] = {}
    correct = 0
    total = 0
    error_count = 0
    for dataset in DATASETS:
        path = run_dir / "results" / dataset / name / model / "test.json"
        data = load_json(path)
        per_dataset[dataset] = float(data["accuracy"])
        correct += int(data.get("correct", 0))
        total += int(data.get("total", 0))
        error_count += int(data.get("error_count", 0))
    return {
        "system_name": name,
        "split": "test",
        "per_dataset": per_dataset,
        "average": sum(per_dataset.values()) / len(per_dataset),
        "summary": {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
            "micro_f1": "",
            "error_count": error_count,
        },
        "errors": [],
    }


def row_for_result(
    *,
    name: str,
    phase: str,
    source: str,
    result: dict[str, Any],
    leaderboard: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    per_dataset = result.get("per_dataset", {})
    summary = result.get("summary", {})
    iteration_match = re.match(r"iter(\d+)_", name)
    iteration = int(iteration_match.group(1)) if iteration_match else ""
    val = leaderboard.get(name, {})
    return {
        "name": name,
        "phase": phase,
        "iteration": iteration,
        "test_source": source,
        "val_macro_avg": val.get("val_macro_avg", ""),
        "context_length": val.get("context_length", ""),
        "test_macro_avg": result.get("average", ""),
        "test_correct": summary.get("correct", ""),
        "test_total": summary.get("total", ""),
        "LawBench": per_dataset.get("LawBench", ""),
        "Symptom2Disease": per_dataset.get("Symptom2Disease", ""),
        "USPTO": per_dataset.get("USPTO", ""),
        "micro_f1": summary.get("micro_f1", ""),
        "error_count": summary.get("error_count", len(result.get("errors", []) or [])),
    }


def add_score_json_rows(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    phase: str,
    source: str,
    leaderboard: dict[str, dict[str, float | int]],
) -> None:
    if not path.exists():
        return
    scores = load_json(path)
    for name, result in scores.items():
        rows.append(
            row_for_result(
                name=name,
                phase=phase,
                source=source,
                result=result,
                leaderboard=leaderboard,
            )
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "name",
        "phase",
        "iteration",
        "test_source",
        "val_macro_avg",
        "context_length",
        "test_macro_avg",
        "test_correct",
        "test_total",
        "LawBench",
        "Symptom2Disease",
        "USPTO",
        "micro_f1",
        "error_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--first5-json", required=True, type=Path)
    parser.add_argument("--extension-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gpt-oss-120b")
    args = parser.parse_args()

    leaderboard = load_leaderboard(args.run_dir / "leaderboard.csv")
    rows: list[dict[str, Any]] = []

    for baseline in ["no_memory", "fewshot_all"]:
        rows.append(
            row_for_result(
                name=baseline,
                phase="baseline",
                source="stored_clean_run_results",
                result=summarize_result_files(args.run_dir, baseline, args.model),
                leaderboard=leaderboard,
            )
        )

    add_score_json_rows(
        rows,
        path=args.first5_json,
        phase="candidate_iter001_005",
        source="candidate10_retest_keyUG_concurrency6",
        leaderboard=leaderboard,
    )
    add_score_json_rows(
        rows,
        path=args.extension_json,
        phase="candidate_iter006_010",
        source="extension_006_010_new_candidates_only",
        leaderboard=leaderboard,
    )

    rows.sort(key=lambda row: (-(row["test_macro_avg"] or 0.0), str(row["name"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)

    print(args.output)
    print(f"rows={len(rows)}")
    for row in rows[:10]:
        print(
            "{name}: test={test:.1%} correct={correct}/{total} "
            "LawBench={law:.1%} Symptom2Disease={sym:.1%} USPTO={uspto:.1%}".format(
                name=row["name"],
                test=float(row["test_macro_avg"]),
                correct=row["test_correct"],
                total=row["test_total"],
                law=float(row["LawBench"]),
                sym=float(row["Symptom2Disease"]),
                uspto=float(row["USPTO"]),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
