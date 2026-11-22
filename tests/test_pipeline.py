"""Orchestration entry point: failures must stop the run and be reported."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import run_pipeline as rp  # noqa: E402


def test_stage_failure_raises(monkeypatch):
    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 3, "", "boom: API unreachable"))
    with pytest.raises(RuntimeError, match="stage 'ingest' failed"):
        rp.run_stage("ingest")


def test_quality_gate_failure_raises(monkeypatch):
    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    def bad_gate(stage, sample): raise AssertionError("row reconciliation failed")
    monkeypatch.setattr(rp, "gate", bad_gate)
    with pytest.raises(RuntimeError, match="quality gate failed after 'clean'"):
        rp.run_stage("clean")


def test_full_run_gates_pass_on_current_outputs():
    for stage in ("validate", "clean", "transform", "analytics", "features", "train"):
        assert rp.gate(stage, sample=False)
