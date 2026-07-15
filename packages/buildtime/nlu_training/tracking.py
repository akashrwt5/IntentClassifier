"""MLflow experiment tracking — local file backend, optional by design.

Roadmap §16: experiments in MLflow with a LOCAL file backend (./mlruns,
gitignored). The dependency is optional: if mlflow is not installed every
hook is a silent no-op, so CI and lean dev environments never pay for it.
Install with `pip install mlflow` (or `make install-dev` once adopted).

Usage (already wired into evaluate):
    from .tracking import log_evaluation
    log_evaluation(report, run_name="post-migration")
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRACKING_URI = (REPO / "mlruns").as_uri()
EXPERIMENT = "nlu-intent-classifier"


def _mlflow():
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def log_evaluation(report: dict, run_name: str = "evaluate",
                   params: dict | None = None) -> bool:
    """Log a unified-evaluate report as one MLflow run. Returns False if
    mlflow is unavailable (silent no-op)."""
    mlflow = _mlflow()
    if mlflow is None:
        return False
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        for lang, m in report["per_language"].items():
            mlflow.log_metric(f"{lang}_holdout_accuracy", m["holdout_accuracy"])
            mlflow.log_metric(f"{lang}_macro_f1", m["macro_f1"])
            mlflow.log_metric(f"{lang}_ece", m["ece"])
        mlflow.log_metric("wrong_action_count", report["wrong_action_count"])
        mlflow.log_metric("oos_recall", report["oos_recall"])
        mlflow.log_metric("gates_passed", int(report["gates_passed"]))
    return True
