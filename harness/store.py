"""
文件分析：文件系统缓存与落盘日志归档站。
主要作用：对接磁盘执行一系列追踪存储策略。如保存跑在路上的排行榜历史、快照每一次演进时的排行榜结果，并对每次尝试迭代下来的海量数据做妥善记录管理。
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


# Persist run artifacts, logs, leaderboard, and frontier snapshots.
class RunStore:
    def __init__(self, root: str | Path, run_name: str):
        self.root = Path(root)
        self.run_name = run_name
        self.path = self.root / run_name

    # Create the run directory before any artifacts are written.
    def create(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    # Save the exact config used for this run.
    def snapshot_config(self, config: Mapping[str, Any]) -> Path:
        return self.write_text("config.yaml", yaml.safe_dump(dict(config), sort_keys=False))

    # Create the per-iteration workspace under runs/.
    def iter_dir(self, iteration: int) -> Path:
        path = self.path / f"iter_{iteration:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Write a plain text artifact at a run-relative path.
    def write_text(self, relative_path: str | Path, text: str) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    # Write a JSON artifact with stable pretty formatting.
    def write_json(self, relative_path: str | Path, data: Any) -> Path:
        return self.write_text(relative_path, json.dumps(data, indent=2, ensure_ascii=False))

    # Append one JSON event to a log file.
    def append_jsonl(self, relative_path: str | Path, data: Mapping[str, Any]) -> Path:
        path = self.path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(data), ensure_ascii=False) + "\n")
        return path

    # Persist the current leaderboard as CSV.
    def write_leaderboard(self, rows: Sequence[Mapping[str, Any]]) -> Path:
        path = self.path / "leaderboard.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["name", "average"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        return path

    def load_frontier(self) -> dict[str, Any]:
        path = self.path / "frontier.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    # Persist the current Pareto/frontier snapshot.
    def save_frontier(self, frontier: Mapping[str, Any]) -> Path:
        return self.write_json("frontier.json", dict(frontier))

    # Emit one timestamped run event to live.log and latest.log.
    def log_event(self, message: str, data: Mapping[str, Any] | None = None) -> Path:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "data": dict(data or {}),
        }
        line = json.dumps(payload, ensure_ascii=False)
        live_path = self.path / "live.log"
        latest_path = self.path / "latest.log"
        live_path.parent.mkdir(parents=True, exist_ok=True)
        with live_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        latest_path.write_text(line + "\n", encoding="utf-8")
        return live_path
