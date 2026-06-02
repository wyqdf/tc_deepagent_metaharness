"""Tests for proposer candidate writing."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

import json

from harness.agent_protocol import load_agent_memory
from harness.proposer import CandidateProposal, DeepAgentProposer, _LoggingLocalShellBackend


def test_write_candidate_keeps_file_under_output_dir(tmp_path):
    proposer = DeepAgentProposer("claude-opus-4-6", tmp_path, dry_run=True)
    proposal = CandidateProposal(
        name="demo_candidate",
        path="demo_candidate.py",
        hypothesis="test",
        source_code=(
            "from harness.agent_protocol import BaseAgentMemory\n\n"
            "class DemoCandidate(BaseAgentMemory):\n"
            "    def predict(self, input): return 'A', {}\n"
            "    def learn_from_batch(self, batch_results): pass\n"
            "    def get_state(self): return '{}'\n"
            "    def set_state(self, state): pass\n"
        ),
        manifest={"iteration": 1},
    )

    path = proposer.write_candidate(proposal)

    assert path.parent == tmp_path
    assert path.name == "demo_candidate.py"
    assert load_agent_memory(path, llm=lambda prompt: '{"final_answer": "A"}').predict("x")[0] == "A"


def test_dry_run_propose_returns_importable_candidate(tmp_path):
    proposer = DeepAgentProposer("claude-opus-4-6", tmp_path, dry_run=True)

    proposal = proposer.propose("context", iteration=2)
    path = proposer.write_candidate(proposal)

    assert proposal.name == "dry_run_candidate_002"
    assert load_agent_memory(path, llm=lambda prompt: '{"final_answer": "A"}').predict("x")[0] == "A"


def test_proposer_keeps_explicit_root_dir(tmp_path):
    root = tmp_path / "project"
    proposer = DeepAgentProposer("claude-opus-4-6", tmp_path, dry_run=True, root_dir=root)

    assert proposer.root_dir == root


def test_official_dry_run_writes_pending_eval_with_two_candidates(tmp_path):
    root = tmp_path
    output_dir = root / "agents"
    pending = root / "runs/demo/pending_eval.json"
    trace = root / "runs/demo/iter_001/proposer_tool_calls.jsonl"
    response = root / "runs/demo/iter_001/proposer_response.md"
    messages = root / "runs/demo/iter_001/proposer_messages.json"
    proposer = DeepAgentProposer(
        "claude-opus-4-6",
        output_dir,
        dry_run=True,
        root_dir=root,
    )

    proposals = proposer.propose_official(
        "task",
        iteration=4,
        pending_eval_path=pending,
        trace_path=trace,
        response_path=response,
        messages_path=messages,
    )

    assert [proposal.name for proposal in proposals] == [
        "dry_run_candidate_004_1",
        "dry_run_candidate_004_2",
    ]
    assert pending.exists()
    loaded = proposer.load_pending_eval(pending, iteration=4)
    assert len(loaded) == 2
    assert (output_dir / "dry_run_candidate_004_1.py").exists()
    assert pending.read_text(encoding="utf-8").count("agents/") == 2
    assert "dry_run_proposer" in trace.read_text(encoding="utf-8")
    assert "dry_run_candidate_004_1" in response.read_text(encoding="utf-8")
    assert json.loads(messages.read_text(encoding="utf-8"))["dry_run"] is True


def test_load_pending_eval_accepts_path_field_from_proposer(tmp_path):
    root = tmp_path
    output_dir = root / "agents"
    output_dir.mkdir()
    candidate = output_dir / "iter012_demo.py"
    candidate.write_text(
        "from harness.agent_protocol import BaseAgentMemory\n\n"
        "class DemoCandidate(BaseAgentMemory):\n"
        "    def predict(self, input): return 'A', {}\n"
        "    def learn_from_batch(self, batch_results): pass\n"
        "    def get_state(self): return '{}'\n"
        "    def set_state(self, state): pass\n",
        encoding="utf-8",
    )
    pending = root / "runs/demo/pending_eval.json"
    pending.parent.mkdir(parents=True)
    pending.write_text(
        json.dumps(
            {
                "iteration": 12,
                "candidates": [
                    {
                        "name": "iter012_demo",
                        "path": "agents/iter012_demo.py",
                        "hypothesis": "path-only regression candidate",
                        "axis": "exploration",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    proposer = DeepAgentProposer("claude-opus-4-6", output_dir, dry_run=True, root_dir=root)

    proposals = proposer.load_pending_eval(pending, iteration=12)

    assert len(proposals) == 1
    assert proposals[0].path == str(candidate)
    assert proposals[0].manifest["file"] == "agents/iter012_demo.py"
    assert proposals[0].manifest["path"] == "agents/iter012_demo.py"


def test_logging_backend_records_file_and_shell_actions(tmp_path):
    trace = tmp_path / "trace.jsonl"
    backend = _LoggingLocalShellBackend(root_dir=tmp_path, virtual_mode=False, trace_path=trace)

    backend.write("notes.txt", "alpha\nbeta\n")
    backend.read("notes.txt")
    backend.grep("alpha", path="notes.txt")
    backend.execute("printf hello")

    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    tools = [event.get("tool") for event in events if event.get("event") == "backend_call"]

    assert tools == ["write", "read", "grep", "execute"]
    assert events[-1]["result"]["exit_code"] == 0
    assert "hello" in events[-1]["result"]["output_preview"]


def test_deepagent_retry_logs_failed_attempt(tmp_path, monkeypatch):
    trace = tmp_path / "retry_trace.jsonl"
    proposer = DeepAgentProposer(
        "claude-opus-4-6",
        tmp_path / "agents",
        dry_run=False,
        root_dir=tmp_path,
        max_retries=1,
        retry_sleep_seconds=0,
    )
    calls = {"count": 0}

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary")
        return "ok"

    monkeypatch.setattr(proposer, "_run_deepagent", fake_run)

    assert proposer._run_deepagent_with_retries("task", 1, trace_path=trace) == "ok"

    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    event_names = [event.get("event") for event in events]
    assert calls["count"] == 2
    assert event_names.count("proposer_attempt_started") == 2
    assert "proposer_attempt_failed" in event_names
    assert any(event.get("will_retry") is True for event in events)


def test_logging_backend_enforces_proposer_information_boundary(tmp_path):
    trace = tmp_path / "trace.jsonl"
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness/llm.py").write_text("# allowed\n", encoding="utf-8")
    (tmp_path / "harness/official_eval.py").write_text("# denied\n", encoding="utf-8")
    backend = _LoggingLocalShellBackend(root_dir=tmp_path, virtual_mode=False, trace_path=trace)

    allowed = backend.read("harness/llm.py")
    denied = backend.read("harness/official_eval.py")
    listing = backend.glob("*.py", path="harness")
    command = backend.execute("cat harness/official_eval.py")

    assert not getattr(allowed, "error", None)
    assert "Access denied" in denied.error
    assert "Access denied" in command.output
    assert [item["path"] for item in listing.matches] == [str(tmp_path / "harness/llm.py")]
    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert any(event["result"].get("error") for event in events if event.get("tool") == "read")


def test_logging_backend_emits_top_level_audit_fields(tmp_path):
    trace = tmp_path / "trace.jsonl"
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents/demo.py").write_text("print('ok')\n", encoding="utf-8")
    backend = _LoggingLocalShellBackend(root_dir=tmp_path, virtual_mode=False, trace_path=trace)

    backend.read("agents/demo.py")
    backend.execute("printf hello")

    events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    call_events = [event for event in events if event.get("event") == "backend_call"]

    assert any(event.get("file_path") == "agents/demo.py" for event in call_events)
    assert any(event.get("command") == "printf hello" for event in call_events)
    assert all("result_preview" in event for event in call_events)


def test_deepagent_proposer_write_guard_matches_root_policy(tmp_path):
    proposer = DeepAgentProposer("claude-opus-4-6", tmp_path / "agents", dry_run=True, root_dir=tmp_path)

    assert proposer._write_guard("agents/candidate.py", "class X: pass")
    assert proposer._write_guard("agents/candidate.py", "def predict(self, input) -> tuple[str, dict]: pass")
    assert proposer._write_guard("agents/candidate.py", "from __future__ import annotations\nclass X: pass")
    assert proposer._write_guard("/tmp/proposer_proto.py", "print('ok')")
    assert proposer._write_guard("runs/demo/out.txt", "hello")
    assert not proposer._write_guard("harness/llm.py", "class X: pass")
    assert not proposer._write_guard("agents/candidate.py", "import os\nos.remove('agents/other.py')")


def test_deepagent_proposer_allows_only_current_run_dir_when_configured(tmp_path):
    proposer = DeepAgentProposer(
        "claude-opus-4-6",
        tmp_path / "agents",
        dry_run=True,
        root_dir=tmp_path,
        allowed_run_dir=tmp_path / "runs/current",
    )

    assert proposer._write_guard("agents/candidate.py", "class X: pass")
    assert proposer._write_guard("runs/current/pending_eval.json", "{\"ok\": true}")
    assert not proposer._write_guard("runs/old/pending_eval.json", "{\"ok\": true}")


def test_logging_backend_allows_current_run_state_but_blocks_old_runs(tmp_path):
    trace = tmp_path / "trace.jsonl"
    current = tmp_path / "runs/current"
    old = tmp_path / "runs/old"
    backup = tmp_path / "tc_deepagent_metaharness_backup_pre_clean_20260524_demo"
    current.mkdir(parents=True)
    old.mkdir(parents=True)
    backup.mkdir()
    (current / "config.yaml").write_text("current: true\n", encoding="utf-8")
    (old / "config.yaml").write_text("old: true\n", encoding="utf-8")
    (backup / "config.yaml").write_text("backup: true\n", encoding="utf-8")
    (tmp_path / "AGENT.md").write_text("old notes\n", encoding="utf-8")

    backend = _LoggingLocalShellBackend(
        root_dir=tmp_path,
        allowed_run_dir=current,
        virtual_mode=False,
        trace_path=trace,
    )

    allowed = backend.read("runs/current/config.yaml")
    denied_old = backend.read("runs/old/config.yaml")
    denied_backup = backend.read(backup / "config.yaml")
    denied_agent = backend.read("AGENT.md")
    listing = backend.glob("config.yaml", path="runs")

    assert not getattr(allowed, "error", None)
    assert "current: true" in allowed.file_data["content"]
    assert "Access denied" in denied_old.error
    assert "Access denied" in denied_backup.error
    assert "Access denied" in denied_agent.error
    assert [item["path"] for item in listing.matches] == [str(current / "config.yaml")]


def test_logging_backend_write_boundary_allows_current_run_and_agents_only(tmp_path):
    trace = tmp_path / "trace.jsonl"
    root = tmp_path / "project"
    current = root / "runs/current"
    current.mkdir(parents=True)
    (root / "agents").mkdir()
    tmp_proto = tmp_path / f"{tmp_path.name}_proto.py"
    if tmp_proto.exists():
        tmp_proto.unlink()
    backend = _LoggingLocalShellBackend(
        root_dir=root,
        allowed_run_dir=current,
        virtual_mode=False,
        trace_path=trace,
    )

    agent_write = backend.write("agents/candidate.py", "class Candidate: pass\n")
    typed_agent_write = backend.write(
        "agents/typed_candidate.py",
        "def predict(self, input) -> tuple[str, dict]:\n    return 'A', {}\n",
    )
    import_agent_write = backend.write(
        "agents/import_candidate.py",
        "from __future__ import annotations\nfrom collections import Counter\n",
    )
    tmp_write = backend.write(str(tmp_proto), "print('ok')\n")
    current_write = backend.write("runs/current/pending_eval.json", "{}")
    old_write = backend.write("runs/old/pending_eval.json", "{}")
    harness_write = backend.write("harness/llm.py", "# nope\n")

    assert not getattr(agent_write, "error", None)
    assert not getattr(typed_agent_write, "error", None)
    assert not getattr(import_agent_write, "error", None)
    assert not getattr(tmp_write, "error", None)
    assert not getattr(current_write, "error", None)
    assert "Access denied" in old_write.error
    assert "Access denied" in harness_write.error


def test_logging_backend_blocks_shell_mutation_of_agents_and_runs(tmp_path):
    trace = tmp_path / "trace.jsonl"
    current = tmp_path / "runs/current"
    current.mkdir(parents=True)
    (tmp_path / "agents").mkdir()
    read_target = current / "readable.txt"
    read_target.write_text("state\n", encoding="utf-8")
    backend = _LoggingLocalShellBackend(
        root_dir=tmp_path,
        allowed_run_dir=current,
        virtual_mode=False,
        trace_path=trace,
    )

    denied_agent = backend.execute("python3 -c \"open('agents/x.py','w').write('x')\"")
    denied_run = backend.execute("cat > runs/current/pending_eval.json <<'EOF'\n{}\nEOF")
    allowed_tmp = backend.execute("python3 -c \"open('/tmp/proposer_proto.txt','w').write('ok'); print('ok')\"")
    allowed_read_current = backend.execute(
        f"python3 -c \"print(open('{read_target}','r').read())\""
    )

    assert "Access denied" in denied_agent.output
    assert "Access denied" in denied_run.output
    assert allowed_tmp.exit_code == 0
    assert allowed_read_current.exit_code == 0
