"""Tests for CLI entrypoint."""
# Regression checks for the harness contract continue below.


from __future__ import annotations

import pytest

from harness.cli import main


def test_cli_help_returns_success(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    captured = capsys.readouterr()

    assert exc.value.code == 0
    assert "benchmark" in captured.out
    assert "run" in captured.out


def test_cli_benchmark_with_stub_solver(capsys):
    code = main(
        [
            "benchmark",
            "--split",
            "val",
            "--config",
            "configs/experiment.yaml",
            "--stub",
            "--max-workers",
            "2",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "average" in captured.out


def test_cli_preflight_with_stub_solver(capsys):
    code = main(["preflight", "--config", "configs/dry_run.yaml"])
    captured = capsys.readouterr()

    assert code == 0
    assert '"ok": true' in captured.out
