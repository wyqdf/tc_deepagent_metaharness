"""Write val/test scores in harness loop order.

This script reads existing artifacts only. It does not call models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ["LawBench", "Symptom2Disease", "USPTO"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(result: dict[str, Any], dataset: str) -> float | str:
    return result.get("per_dataset", {}).get(dataset, "")


def summarize_test_files(run_dir: Path, name: str, model: str) -> dict[str, Any]:
    per_dataset: dict[str, float] = {}
    correct = 0
    total = 0
    errors = 0
    for dataset in DATASETS:
        path = run_dir / "results" / dataset / name / model / "test.json"
        data = load_json(path)
        per_dataset[dataset] = float(data["accuracy"])
        correct += int(data.get("correct", 0))
        total += int(data.get("total", 0))
        errors += int(data.get("error_count", 0))
    return {
        "per_dataset": per_dataset,
        "average": sum(per_dataset.values()) / len(per_dataset),
        "summary": {
            "correct": correct,
            "total": total,
            "micro_f1": "",
            "error_count": errors,
        },
        "errors": [],
    }


def load_test_scores(first5_json: Path, extension_json: Path) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for path in [first5_json, extension_json]:
        if path.exists():
            scores.update(load_json(path))
    return scores


def candidate_order(run_dir: Path) -> list[tuple[int, str]]:
    ordered: list[tuple[int, str]] = []
    for iter_dir in sorted(run_dir.glob("iter_*")):
        pending_path = iter_dir / "pending_eval.json"
        if not pending_path.exists():
            continue
        data = load_json(pending_path)
        iteration = int(data.get("iteration") or iter_dir.name.split("_")[-1])
        for item in data.get("candidates", []):
            ordered.append((iteration, item["name"]))
    return ordered


def row(
    *,
    phase: str,
    iteration: int | str,
    name: str,
    val_result: dict[str, Any] | None,
    test_result: dict[str, Any] | None,
) -> dict[str, Any]:
    val_summary = (val_result or {}).get("summary", {})
    test_summary = (test_result or {}).get("summary", {})
    return {
        "phase": phase,
        "iteration": iteration,
        "name": name,
        "val_macro_avg": (val_result or {}).get("average", ""),
        "val_correct": val_summary.get("correct", ""),
        "val_total": val_summary.get("total", ""),
        "val_LawBench": metric(val_result or {}, "LawBench"),
        "val_Symptom2Disease": metric(val_result or {}, "Symptom2Disease"),
        "val_USPTO": metric(val_result or {}, "USPTO"),
        "val_micro_f1": val_summary.get("micro_f1", ""),
        "test_macro_avg": (test_result or {}).get("average", ""),
        "test_correct": test_summary.get("correct", ""),
        "test_total": test_summary.get("total", ""),
        "test_LawBench": metric(test_result or {}, "LawBench"),
        "test_Symptom2Disease": metric(test_result or {}, "Symptom2Disease"),
        "test_USPTO": metric(test_result or {}, "USPTO"),
        "test_micro_f1": test_summary.get("micro_f1", ""),
    }


def percent(value: Any) -> str:
    if value == "":
        return "-"
    return f"{float(value):.1%}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--first5-json", required=True, type=Path)
    parser.add_argument("--extension-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gpt-oss-120b")
    args = parser.parse_args()

    test_scores = load_test_scores(args.first5_json, args.extension_json)
    rows: list[dict[str, Any]] = []

    for baseline_result in load_json(args.run_dir / "baseline_scores.json"):
        name = baseline_result["system_name"]
        rows.append(
            row(
                phase="baseline",
                iteration="",
                name=name,
                val_result=baseline_result,
                test_result=summarize_test_files(args.run_dir, name, args.model),
            )
        )

    for iteration, name in candidate_order(args.run_dir):
        val_path = args.run_dir / f"iter_{iteration:03d}" / f"{name}_scores.json"
        rows.append(
            row(
                phase="candidate",
                iteration=iteration,
                name=name,
                val_result=load_json(val_path) if val_path.exists() else None,
                test_result=test_scores.get(name),
            )
        )

    fieldnames = [
        "phase",
        "iteration",
        "name",
        "val_macro_avg",
        "val_correct",
        "val_total",
        "val_LawBench",
        "val_Symptom2Disease",
        "val_USPTO",
        "val_micro_f1",
        "test_macro_avg",
        "test_correct",
        "test_total",
        "test_LawBench",
        "test_Symptom2Disease",
        "test_USPTO",
        "test_micro_f1",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(args.output)
    print(f"rows={len(rows)}")
    for item in rows:
        print(
            "{prefix}{name}: val={val} ({vc}/{vt}) test={test} ({tc}/{tt})".format(
                prefix=f"iter{int(item['iteration']):03d} " if item["iteration"] != "" else "baseline ",
                name=item["name"],
                val=percent(item["val_macro_avg"]),
                vc=item["val_correct"],
                vt=item["val_total"],
                test=percent(item["test_macro_avg"]),
                tc=item["test_correct"],
                tt=item["test_total"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
