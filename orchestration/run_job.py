"""Execute the Dagster job in-process and print per-step status.

    DAGSTER_HOME=$PWD/dagster_home python orchestration/run_job.py [--sample] [--skip-ingest]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("DAGSTER_HOME", str(ROOT / "dagster_home"))
(ROOT / "dagster_home").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "orchestration"))

from dagster import DagsterInstance  # noqa: E402
from dagster_pipeline import recomart_pipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sample", action="store_true"); ap.add_argument("--skip-ingest", action="store_true")
a = ap.parse_args()
cfg = {"ops": {n: {"config": {"sample": a.sample, "skip_ingest": a.skip_ingest}}
               for n in ("inspect", "ingest", "validate", "clean", "transform", "analytics", "features", "train")}}
res = recomart_pipeline.execute_in_process(run_config=cfg, instance=DagsterInstance.get(), raise_on_error=False)
print(f"\nRUN {res.run_id}: {'SUCCESS' if res.success else 'FAILURE'}")
for ev in res.get_step_success_events():
    print("  OK  ", ev.step_key)
for ev in res.get_step_failure_events():
    print("  FAIL", ev.step_key)
sys.exit(0 if res.success else 1)
