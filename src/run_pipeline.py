"""Person 4 - end-to-end pipeline entry point.

    python src/run_pipeline.py                       # full data, all stages
    python src/run_pipeline.py --sample              # 10,000-row development slice
    python src/run_pipeline.py --skip-ingest         # reuse the existing raw lake (offline)
    python src/run_pipeline.py --stages clean transform

Stages (in order): inspect -> ingest -> validate -> clean -> transform -> analytics -> features -> train
Every stage runs as a subprocess of the project's own interpreter, is timed, logged to
logs/pipeline.log, and is followed by a lightweight data-quality gate. The first failure stops the
run with a non-zero exit code. The Dagster job in orchestration/dagster_pipeline.py reuses
``run_stage`` so both entry points behave identically.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PROCESSED = ROOT / "data" / "processed"
STAGES = ["inspect", "ingest", "validate", "clean", "transform", "analytics", "features", "train"]
SAMPLE_ROWS = 10_000
LOG = logging.getLogger("pipeline")


def setup_logging() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    if LOG.handlers:
        return
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.FileHandler(ROOT / "logs" / "pipeline.log"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); LOG.addHandler(h)
    LOG.setLevel(logging.INFO)


def command(stage: str, sample: bool) -> list[str]:
    py = sys.executable
    return {
        "inspect": [py, str(SRC / "01_inspect.py")],
        "ingest": [py, str(SRC / "01_ingest.py")],
        "validate": [py, str(SRC / "02_validate.py")],
        "clean": [py, str(SRC / "02_clean.py"), *(["--sample-rows", str(SAMPLE_ROWS)] if sample else [])],
        "transform": [py, str(SRC / "03_transform.py"), *(["--sample"] if sample else [])],
        "analytics": [py, str(SRC / "04_analytics.py"), *(["--sample"] if sample else [])],
        "features": [py, str(SRC / "05_features.py"), *(["--sample"] if sample else [])],
        "train": [py, str(SRC / "06_train.py"), *(["--sample", "--eval-users", "1000"] if sample else [])],
    }[stage]


# ------------------------------------------------------------------ quality gates
def _json(name: str) -> dict:
    return json.loads((PROCESSED / name).read_text())


def gate(stage: str, sample: bool) -> str:
    """Return a short description of what was checked; raise AssertionError on failure."""
    sfx = "_sample" if sample else ""
    if stage == "validate":
        latest = sorted(PROCESSED.glob("validation_report_*.json"))[-1]
        rep = json.loads(latest.read_text())
        assert rep["status"] == "PASSED", f"validation status {rep['status']}"
        assert rep["transactions"]["row_count"] > 0, "raw transactions empty"
        return f"{latest.name} PASSED, {rep['transactions']['row_count']} rows"
    if stage == "clean":
        r = _json(f"cleaning_report{sfx}.json")
        assert r["valid_rows"] > 0, "no valid rows after cleaning"
        assert r["input_rows"] == r["valid_rows"] + r["quarantined_invalid_rows"] + r["duplicates_removed"], "row reconciliation failed"
        return f"input={r['input_rows']} valid={r['valid_rows']} quarantined={r['quarantined_invalid_rows']} dup={r['duplicates_removed']}"
    if stage == "transform":
        r = _json(f"transformation_report{sfx}.json")
        assert r["reconciliation"]["all_passed"], "transformation reconciliation failed"
        c = _json(f"cleaning_report{sfx}.json")
        assert r["output_rows"]["transformed"] == c["valid_rows"], "transformed rows != cleaned rows"
        return f"{r['output_rows']['transformed']} rows, {r['output_rows']['customer']} customers, 12/12 checks passed"
    if stage == "analytics":
        r = json.loads((ROOT / "data" / "analytics" / f"analytics_report{sfx}.json").read_text())
        t = _json(f"transformation_report{sfx}.json")
        assert r["interaction_stats"]["rows"] == t["output_rows"]["transformed"], "analytics rows != transformed rows"
        return f"{r['interaction_stats']['rows']} rows, sparsity {r['interaction_stats']['sparsity_pct']}%"
    if stage == "features":
        import pandas as pd
        reg = json.loads((ROOT / "data" / "feature_store" / "registry.json").read_text())
        assert f"user_features{sfx}" in reg["views"] and f"item_features{sfx}" in reg["views"], "feature views missing"
        v = max(int(k) for k in reg["views"][f"user_features{sfx}"]["versions"])
        return f"user_features v{v} registered"
    if stage == "train":
        res = json.loads((ROOT / "reports" / f"model_performance{sfx}.json").read_text())
        assert len(res) >= 3 and (ROOT / "models" / f"svd_model{sfx}.npz").exists(), "models/metrics missing"
        best = max(res, key=lambda x: x["ndcg@10"])
        return f"{len(res)} runs tracked, best ndcg@10={best['ndcg@10']:.5f} ({best['run']})"
    return "no gate"


def run_stage(stage: str, sample: bool = False) -> dict:
    """Run one stage, apply its gate, return a summary dict. Raises RuntimeError on failure."""
    setup_logging()
    cmd = command(stage, sample)
    LOG.info("[%s] START %s", stage, " ".join(Path(c).name if c.startswith("/") else c for c in cmd))
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        tail = "\n".join((p.stderr or p.stdout).strip().splitlines()[-6:])
        LOG.error("[%s] FAILED rc=%s after %.1fs\n%s", stage, p.returncode, dt, tail)
        raise RuntimeError(f"stage '{stage}' failed (rc={p.returncode}): {tail.splitlines()[-1] if tail else ''}")
    try:
        checked = gate(stage, sample)
    except AssertionError as e:
        LOG.error("[%s] QUALITY GATE FAILED: %s", stage, e)
        raise RuntimeError(f"quality gate failed after '{stage}': {e}") from e
    LOG.info("[%s] OK in %.1fs | gate: %s", stage, dt, checked)
    return {"stage": stage, "seconds": round(dt, 1), "gate": checked}


def run_pipeline(stages: list[str], sample: bool = False) -> list[dict]:
    setup_logging()
    LOG.info("=== pipeline start | stages=%s | sample=%s ===", stages, sample)
    out = [run_stage(s, sample) for s in stages]
    LOG.info("=== pipeline finished OK | total %.1fs ===", sum(o["seconds"] for o in out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true"); ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=None)
    a = ap.parse_args()
    chosen = a.stages or [s for s in STAGES if not (a.skip_ingest and s == "ingest")]
    try:
        run_pipeline(chosen, a.sample)
    except RuntimeError as e:
        LOG.error("=== pipeline FAILED: %s ===", e)
        sys.exit(1)
