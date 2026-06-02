"""Print compact candidate retest scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores_json", help="Path to candidate_test_scores.json")
    args = parser.parse_args()

    path = Path(args.scores_json)
    if not path.exists():
        print(f"No scores yet: {path}")
        return 0

    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        print(f"No completed candidates yet: {path}")
        return 0

    for name, result in data.items():
        average = result.get("average", 0.0)
        per_dataset = result.get("per_dataset", {})
        summary = result.get("summary", {})
        parts = [
            f"{name}: avg={average:.1%}",
            f"correct={summary.get('correct', '?')}/{summary.get('total', '?')}",
        ]
        for dataset in sorted(per_dataset):
            parts.append(f"{dataset}={per_dataset[dataset]:.1%}")
        if summary.get("micro_f1") is not None:
            parts.append(f"micro_f1={summary['micro_f1']:.3f}")
        if summary.get("error_count"):
            parts.append(f"errors={summary['error_count']}")
        print(" | ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
