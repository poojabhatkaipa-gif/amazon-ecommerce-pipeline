"""Dagster orchestration for the RecoMart pipeline (Person 4).

DAG:  inspect -> ingest -> validate -> clean -> transform -> analytics -> features -> train

* each op calls src/run_pipeline.run_stage (subprocess + quality gate + logging to logs/pipeline.log)
* ingest has a retry policy (3 attempts, 30 s apart) for the REST API
* a run-failure sensor logs failures to logs/pipeline_failures.log (monitoring hook)
* schedule: daily 02:00 for the full pipeline (periodic fetching)

Run in-process (no web server needed):
    DAGSTER_HOME=$PWD/dagster_home python orchestration/run_job.py [--sample] [--skip-ingest]
Web UI (shows the DAG, runs, logs):
    DAGSTER_HOME=$PWD/dagster_home dagster dev -f orchestration/dagster_pipeline.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

from dagster import (Config, Definitions, In, Nothing, OpExecutionContext, Out, RetryPolicy, RunFailureSensorContext,
                     ScheduleDefinition, job, op, run_failure_sensor)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import run_pipeline as rp  # noqa: E402


class PipelineConfig(Config):
    sample: bool = False
    skip_ingest: bool = False


def _stage(context: OpExecutionContext, stage: str, cfg: PipelineConfig) -> None:
    if stage == "ingest" and cfg.skip_ingest:
        context.log.info("ingest skipped (skip_ingest=true): reusing existing raw lake")
        return
    try:
        res = rp.run_stage(stage, cfg.sample)
    except RuntimeError as e:
        context.log.error(str(e)); raise
    context.log.info(f"{stage} OK in {res['seconds']}s | {res['gate']}")


def _make(stage: str, retry=None):
    @op(name=stage, ins={"after": In(Nothing)}, retry_policy=retry, description=f"Pipeline stage: {stage}")
    def _op(context: OpExecutionContext, config: PipelineConfig):
        _stage(context, stage, config)
    return _op


inspect_op = _make("inspect")
ingest_op = _make("ingest", RetryPolicy(max_retries=3, delay=30))
validate_op, clean_op, transform_op = _make("validate"), _make("clean"), _make("transform")
analytics_op, features_op, train_op = _make("analytics"), _make("features"), _make("train")


@op(name="start", out=Out(Nothing))
def start_op(context: OpExecutionContext):
    context.log.info("pipeline started %s", datetime.now(timezone.utc).isoformat(timespec="seconds"))


@job(name="recomart_pipeline", description="Ingest -> validate -> prepare -> transform -> features -> train")
def recomart_pipeline():
    a = inspect_op(after=start_op())
    b = ingest_op(after=a)
    c = validate_op(after=b)
    d = clean_op(after=c)
    e = transform_op(after=d)
    f = analytics_op(after=e)
    g = features_op(after=f)
    train_op(after=g)


daily = ScheduleDefinition(job=recomart_pipeline, cron_schedule="0 2 * * *", name="daily_full_pipeline")


@run_failure_sensor(name="log_pipeline_failures")
def failure_logger(context: RunFailureSensorContext):
    (ROOT / "logs").mkdir(exist_ok=True)
    with open(ROOT / "logs" / "pipeline_failures.log", "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} | run {context.dagster_run.run_id} "
                f"failed: {context.failure_event.message}\n")


defs = Definitions(jobs=[recomart_pipeline], schedules=[daily], sensors=[failure_logger])
