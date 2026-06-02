"""Tests for filesystem run store."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

import json

from harness.store import RunStore


def test_run_store_writes_core_artifacts(tmp_path):
    store = RunStore(tmp_path, "demo_run")

    run_dir = store.create()
    config_path = store.snapshot_config({"run": {"name": "demo_run"}})
    text_path = store.write_text("iter_001/prompt.md", "hello")
    json_path = store.write_json("iter_001/scores.json", {"average": 1.0})
    jsonl_path = store.append_jsonl("evolution.jsonl", {"iteration": 1})
    leaderboard_path = store.write_leaderboard([{"name": "baseline", "average": 1.0}])
    frontier_path = store.save_frontier({"best": "baseline"})

    assert run_dir.exists()
    assert config_path.exists()
    assert text_path.read_text(encoding="utf-8") == "hello"
    assert json.loads(json_path.read_text(encoding="utf-8"))["average"] == 1.0
    assert jsonl_path.read_text(encoding="utf-8").strip() == '{"iteration": 1}'
    assert "baseline" in leaderboard_path.read_text(encoding="utf-8")
    assert frontier_path.exists()
    assert store.load_frontier() == {"best": "baseline"}


def test_iter_dir_is_zero_padded(tmp_path):
    store = RunStore(tmp_path, "demo")

    assert store.iter_dir(2).name == "iter_002"


def test_log_event_appends_live_log_and_overwrites_latest(tmp_path):
    store = RunStore(tmp_path, "demo")
    store.create()

    live_path = store.log_event("started", {"step": 1})
    store.log_event("finished", {"step": 2})

    live_lines = live_path.read_text(encoding="utf-8").splitlines()
    latest = (store.path / "latest.log").read_text(encoding="utf-8")
    assert len(live_lines) == 2
    assert '"message": "started"' in live_lines[0]
    assert '"message": "finished"' in latest
