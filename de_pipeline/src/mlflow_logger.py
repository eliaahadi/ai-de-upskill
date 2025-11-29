from __future__ import annotations
import os
import mlflow


def maybe_init(experiment: str = "de_pipeline"):
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def log_params(params: dict) -> None:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.log_params(params)


def log_metrics(metrics: dict) -> None:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.log_metrics(metrics)
