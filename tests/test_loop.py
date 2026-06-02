"""Tests for the outer harness loop."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

import json
from pathlib import Path

import yaml

from harness.loop import (
    _build_official_like_proposer_context,
    _collect_proposer_source_files,
    _compute_pareto_frontier,
    _dataset_specific_hint_reasons,
    _final_test_systems,
    _max_workers,
    _preflight_enabled,
    _print_score,
    _validate_proposals_with_reasons,
    _proposer_env,
    _stop_if_eval_failed,
    _write_proposer_audit,
    run_harness,
)


def test_run_harness_one_round_dry_loop(tmp_path):
    project = tmp_path
    (project / "configs").mkdir()
    (project / "data/demo").mkdir(parents=True)
    (project / "agents").mkdir(parents=True)
    (project / "runs").mkdir()
    for agent_name in ["no_memory", "fewshot_all"]:
        source = (Path.cwd() / "agents" / f"{agent_name}.py").read_text(encoding="utf-8")
        (project / "agents" / f"{agent_name}.py").write_text(source, encoding="utf-8")

    for split in ["train", "val", "test"]:
        (project / f"data/demo/{split}.jsonl").write_text(
            json.dumps({"id": split, "text": "alpha", "label": "A", "choices": ["A", "B"]}) + "\n",
            encoding="utf-8",
        )

    config = {
        "run": {"name": "dry_loop", "rounds": 1},
        "solver": {"model": "stub", "stub": True, "stub_response": '{"final_answer": "A"}'},
        "proposer": {"model": "claude-opus-4-6", "dry_run": True},
        "candidate": {
            "baselines": ["no_memory", "fewshot_all"],
            "output_dir": "agents",
        },
        "datasets": [
            {
                "name": "demo",
                "train": "data/demo/train.jsonl",
                "val": "data/demo/val.jsonl",
                "test": "data/demo/test.jsonl",
            }
        ],
    }
    config_path = project / "configs/experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_harness(config_path)

    assert result["best"]["name"] in {"no_memory", "fewshot_all", "dry_run_candidate_001_1", "dry_run_candidate_001_2"}
    assert (project / "runs/dry_loop/frontier.json").exists()
    assert (project / "runs/dry_loop/frontier_val.json").exists()
    assert (project / "runs/dry_loop/pending_eval.json").exists()
    assert (project / "runs/dry_loop/iter_001/dry_run_candidate_001_1.py").exists()
    assert (project / "runs/dry_loop/iter_001/dry_run_candidate_001_2.py").exists()
    assert (project / "runs/dry_loop/logs/demo/no_memory/stub/memory.json").exists()
    assert (project / "runs/dry_loop/results/demo/no_memory/stub/test.json").exists()


def test_proposer_env_uses_doro_auth_token_from_config(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    env = _proposer_env(
        {
            "base_url": "https://doro.lol",
            "auth_token": "sk-test",
            "base_url_env": "ANTHROPIC_BASE_URL",
            "auth_token_env": "ANTHROPIC_AUTH_TOKEN",
        }
    )

    assert env["ANTHROPIC_BASE_URL"] == "https://doro.lol"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-test"


def test_max_workers_prefers_evaluation_then_inner_loop_then_benchmark():
    assert _max_workers({"evaluation": {"max_workers": 7}}) == 7
    assert _max_workers({"inner_loop": {"max_workers": 5}}) == 5
    assert _max_workers({"benchmark": {"concurrency": 3}}) == 3
    assert _max_workers({}) == 1


def test_preflight_enabled_defaults_true_and_can_disable():
    assert _preflight_enabled({}) is True
    assert _preflight_enabled({"preflight": {"enabled": False}}) is False


def test_print_score_shows_accuracy_like_reference(capsys):
    class Result:
        average = 0.5
        per_dataset = {"demo": 0.5}
        summary = {"correct": 1, "total": 2, "micro_f1": None}

    _print_score("candidate", "val", Result(), iteration=1)
    captured = capsys.readouterr()

    assert "[iter 001] candidate val: avg=50.0% correct=1/2" in captured.out
    assert "demo=50.0%" in captured.out


def test_stop_if_eval_failed_raises_and_logs(tmp_path):
    from harness.eval import BenchmarkResult
    from harness.store import RunStore

    store = RunStore(tmp_path, "demo")
    store.create()
    result = BenchmarkResult(
        system_name="bad",
        split="val",
        per_dataset={},
        average=0.0,
        summary={"total": 0},
        errors=[{"error": "403"}],
    )

    import pytest

    with pytest.raises(RuntimeError):
        _stop_if_eval_failed(store, result, "bad")
    assert "eval_failed" in (store.path / "latest.log").read_text(encoding="utf-8")


def test_collect_proposer_source_files_includes_core_and_leaderboard_sources():
    project_root = Path.cwd()
    files = _collect_proposer_source_files(
        project_root,
        {"proposer_context": {"max_total_source_chars": 200000}},
        [{"path": str(project_root / "agents/fewshot_all.py")}],
    )

    assert "harness/agent_protocol.py" in files
    assert "harness/llm.py" in files
    assert "agents/fewshot_all.py" in files
    assert "class BaseAgentMemory" in files["harness/agent_protocol.py"]
    assert "class FewShotAll" in files["agents/fewshot_all.py"]
    assert not any("reference_examples" in path for path in files)
    assert "harness/data.py" not in files
    assert "harness/evaluators.py" not in files
    assert "harness/official_eval.py" not in files


def test_collect_proposer_source_files_can_include_collected_harness_sources(tmp_path):
    project_root = Path.cwd()
    source_root = tmp_path / "external_harness"
    skill_dir = source_root / ".claude/skills/meta-harness"
    agents_dir = source_root / "agents"
    skill_dir.mkdir(parents=True)
    agents_dir.mkdir()
    (source_root / "meta_harness.py").write_text("# external meta harness\n", encoding="utf-8")
    (source_root / "benchmark.py").write_text("# external benchmark\n", encoding="utf-8")
    (source_root / "memory_system.py").write_text("# external memory system interface\n", encoding="utf-8")
    (source_root / "llm.py").write_text("# external llm interface\n", encoding="utf-8")
    (source_root / "README.md").write_text("# external readme\n", encoding="utf-8")
    (source_root / "config.yaml").write_text("run: demo\n", encoding="utf-8")
    (skill_dir / "SKILL.md").write_text("# external proposer skill\n", encoding="utf-8")
    (agents_dir / "fewshot.py").write_text("# external fewshot agent\n", encoding="utf-8")

    disabled = _collect_proposer_source_files(
        project_root,
        {
            "proposer_context": {
                "include_collected_harness_sources": False,
                "harness_source_roots": [str(source_root)],
            }
        },
        [],
    )
    enabled = _collect_proposer_source_files(
        project_root,
        {
            "proposer_context": {
                "include_collected_harness_sources": True,
                "harness_source_roots": [str(source_root)],
                "max_total_source_chars": 300000,
            }
        },
        [],
    )

    assert not any(str(source_root) in path for path in disabled)
    assert str(source_root / "memory_system.py") in enabled
    assert str(source_root / "llm.py") in enabled
    assert str(skill_dir / "SKILL.md") in enabled
    assert str(agents_dir / "fewshot.py") in enabled
    assert "inline preview omitted" in enabled[str(skill_dir / "SKILL.md")]
    assert str(source_root / "meta_harness.py") not in enabled
    assert str(source_root / "benchmark.py") not in enabled
    assert str(source_root / "README.md") not in enabled
    assert str(source_root / "config.yaml") not in enabled


def test_collected_harness_sources_do_not_include_deepagent_example_sources(tmp_path):
    project_root = Path.cwd()
    task_harness_root = tmp_path / "reference_examples/text_classification"
    deepagent_example_root = tmp_path / "deepagents/examples/better-harness"
    (task_harness_root / "agents").mkdir(parents=True)
    (deepagent_example_root / "better_harness").mkdir(parents=True)
    (task_harness_root / "memory_system.py").write_text("# task memory interface\n", encoding="utf-8")
    (task_harness_root / "agents/fewshot.py").write_text("# task candidate\n", encoding="utf-8")
    (deepagent_example_root / "better_harness/core.py").write_text("# proposer tool example\n", encoding="utf-8")

    files = _collect_proposer_source_files(
        project_root,
        {
            "proposer_context": {
                "include_collected_harness_sources": True,
                "harness_source_roots": [str(task_harness_root)],
                "max_total_source_chars": 300000,
            }
        },
        [],
    )

    assert str(task_harness_root / "memory_system.py") in files
    assert str(task_harness_root / "agents/fewshot.py") in files
    assert not any("deepagents/examples/better-harness" in path for path in files)


def test_collect_proposer_source_files_keeps_candidate_paths_after_inline_budget(tmp_path):
    project_root = Path.cwd()
    source_root = tmp_path / "external_harness"
    source_root.mkdir()
    (source_root / "memory_system.py").write_text("x" * 500, encoding="utf-8")
    (source_root / "llm.py").write_text("# llm still indexed\n", encoding="utf-8")

    files = _collect_proposer_source_files(
        project_root,
        {
            "proposer_context": {
                "include_collected_harness_sources": True,
                "harness_source_roots": [str(source_root)],
                "harness_source_globs": ["memory_system.py", "llm.py"],
                "max_total_source_chars": 10,
            }
        },
        [],
    )

    assert str(source_root / "memory_system.py") in files
    assert str(source_root / "llm.py") in files
    assert "inline preview omitted" in files[str(source_root / "llm.py")]


def test_build_official_like_proposer_context_contains_same_information_types(tmp_path):
    from harness.eval import BenchmarkResult, DatasetSpec

    project_root = Path.cwd()
    run_dir = tmp_path / "runs/demo"
    run_dir.mkdir(parents=True)
    (run_dir / "frontier_val.json").write_text(
        json.dumps({"USPTO": {"best_system": "baseline", "val_accuracy": 0.1}}),
        encoding="utf-8",
    )
    (run_dir / "evolution_summary.jsonl").write_text(
        json.dumps({"iteration": 0, "system": "baseline"}) + "\n",
        encoding="utf-8",
    )
    result = BenchmarkResult(
        system_name="baseline",
        split="val",
        per_dataset={"USPTO": 0.0},
        average=0.0,
        summary={"correct": 0, "total": 1},
        traces=[
            {
                "dataset": "USPTO",
                "id": "x1",
                "prediction": "bad",
                "target": "gold",
                "was_correct": False,
                "prompt": [{"role": "user", "content": "classify this"}],
                "retrieved": [{"id": "r1", "text": "prior", "label": "gold", "score": 1.0}],
            }
        ],
    )
    prompt = _build_official_like_proposer_context(
        project_root=project_root,
        config={
            "candidate": {
                "output_dir": "agents",
            },
            "proposer_context": {
                "max_total_source_chars": 120000,
            },
        },
        run_dir=run_dir,
        iteration=1,
        datasets=[
            DatasetSpec(
                "USPTO",
                "/tmp/train.jsonl",
                "/tmp/val.jsonl",
                "/tmp/test.jsonl",
            )
        ],
        leaderboard=[{"name": "fewshot_all", "path": str(project_root / "agents/fewshot_all.py"), "average": 0.0}],
        history=[{"name": "fewshot_all", "path": str(project_root / "agents/fewshot_all.py"), "result": result}],
    )

    assert "Candidate contract:" in prompt
    assert "allowed_writes" in prompt
    assert "agents" in prompt
    assert "State file paths to inspect:" in prompt
    assert "frontier_val.json" in prompt
    assert "Result file index:" in prompt
    assert "Trace file index:" in prompt
    assert "log.jsonl" in prompt
    assert "classify this" not in prompt
    assert '"text": "prior"' not in prompt
    assert '"retrieved"' not in prompt
    assert "per_dataset" not in prompt
    assert "Source file index:" in prompt
    assert "harness/agent_protocol.py" in prompt
    assert "class MemorySystem" not in prompt
    assert "harness/official_eval.py" not in prompt
    assert "harness/evaluators.py" not in prompt
    assert "reference_examples/text_classification/data" not in prompt


def test_pareto_frontier_prefers_accuracy_and_lower_context():
    rows = [
        {"name": "a", "path": "agents/a.py", "average": 0.70, "context_length": 100},
        {"name": "b", "path": "agents/b.py", "average": 0.80, "context_length": 200},
        {"name": "c", "path": "agents/c.py", "average": 0.75, "context_length": 50},
        {"name": "d", "path": "agents/d.py", "average": 0.60, "context_length": 500},
    ]

    pareto = _compute_pareto_frontier(rows)

    assert [entry["system"] for entry in pareto] == ["b", "c"]


def test_final_test_systems_include_baselines_pareto_and_dataset_bests():
    leaderboard = [
        {"name": "no_memory", "path": "agents/no_memory.py", "average": 0.1},
        {"name": "fewshot_all", "path": "agents/fewshot_all.py", "average": 0.2},
        {"name": "candidate_a", "path": "agents/candidate_a.py", "average": 0.3},
        {"name": "candidate_b", "path": "agents/candidate_b.py", "average": 0.4},
    ]
    frontier = {
        "_pareto": [{"system": "candidate_a"}],
        "USPTO": {"best_system": "candidate_b"},
    }

    systems = _final_test_systems(["no_memory", "fewshot_all"], frontier, leaderboard)

    assert list(systems) == ["no_memory", "fewshot_all", "candidate_a", "candidate_b"]


def test_dataset_hint_guard_ignores_docstrings_but_flags_runtime_strings(tmp_path):
    docstring_only = tmp_path / "docstring_only.py"
    docstring_only.write_text(
        '"""Mentions USPTO and LawBench only as explanatory text."""\n'
        "from harness.agent_protocol import BaseAgentMemory\n\n"
        "class Candidate(BaseAgentMemory):\n"
        "    def predict(self, input): return 'A', {}\n"
        "    def learn_from_batch(self, batch_results): pass\n"
        "    def get_state(self): return '{}'\n"
        "    def set_state(self, state): pass\n",
        encoding="utf-8",
    )
    runtime_hint = tmp_path / "runtime_hint.py"
    runtime_hint.write_text(
        "from harness.agent_protocol import BaseAgentMemory\n\n"
        "DATASET = 'USPTO'\n"
        "class Candidate(BaseAgentMemory):\n"
        "    def predict(self, input): return DATASET, {}\n"
        "    def learn_from_batch(self, batch_results): pass\n"
        "    def get_state(self): return '{}'\n"
        "    def set_state(self, state): pass\n",
        encoding="utf-8",
    )

    assert _dataset_specific_hint_reasons(docstring_only) == []
    assert _dataset_specific_hint_reasons(runtime_hint)


def test_validate_proposals_reports_rejection_reasons(tmp_path):
    from harness.proposer import CandidateProposal

    candidate = tmp_path / "runtime_hint.py"
    candidate.write_text(
        "from harness.agent_protocol import BaseAgentMemory\n\n"
        "DATASET = 'LawBench'\n"
        "class Candidate(BaseAgentMemory):\n"
        "    def predict(self, input): return DATASET, {}\n"
        "    def learn_from_batch(self, batch_results): pass\n"
        "    def get_state(self): return '{}'\n"
        "    def set_state(self, state): pass\n",
        encoding="utf-8",
    )
    proposal = CandidateProposal(
        name="runtime_hint",
        path=str(candidate),
        hypothesis="test",
        source_code=candidate.read_text(encoding="utf-8"),
    )

    decisions = _validate_proposals_with_reasons([proposal])

    assert decisions == [
        {
            "proposal": proposal,
            "valid": False,
            "reasons": ["runtime_string_contains:lawbench:line3"],
        }
    ]


def test_write_proposer_audit_records_required_fields(tmp_path):
    from harness.proposer import CandidateProposal

    run_dir = tmp_path / "runs/demo"
    iter_dir = run_dir / "iter_001"
    iter_dir.mkdir(parents=True)
    trace = iter_dir / "proposer_tool_calls.jsonl"
    trace.write_text(
        json.dumps(
            {
                "event": "backend_call",
                "tool": "read",
                "file_path": "logs/USPTO/no_memory/stub/log.jsonl",
                "args": {"file_path": "logs/USPTO/no_memory/stub/log.jsonl"},
                "result_preview": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proposal = CandidateProposal(
        name="candidate",
        path="agents/candidate.py",
        hypothesis="test",
        source_code="from harness.agent_protocol import BaseAgentMemory\n",
    )

    audit_path = _write_proposer_audit(
        run_dir=run_dir,
        iter_dir=iter_dir,
        prompt_text="WORKFLOW\nStep 2: Prototype — MANDATORY\n",
        proposals=[proposal],
        trace_path=trace,
        config={"run": {"name": "demo", "rounds": 5}, "proposer": {"model": "claude-opus-4-6"}},
    )

    data = json.loads(audit_path.read_text(encoding="utf-8"))
    assert data["workflow"]["prototype_required"] is True
    assert "evolution_summary.jsonl" in data["workflow"]["must_read_state_files"]
    assert data["candidate_files"] == ["agents/candidate.py"]
    assert data["inspected_actions"][0]["file_path"] == "logs/USPTO/no_memory/stub/log.jsonl"
