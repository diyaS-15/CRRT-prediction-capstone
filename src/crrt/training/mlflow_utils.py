"""Shared MLflow setup for all training scripts.

Uses a local SQLite-backed tracking store (mlflow.db at the repo root) rather
than the plain mlruns/ file store, because SQLite is required for the full
Model Registry API (registering versions, stage transitions) without running
a separate `mlflow server` process. Override MLFLOW_TRACKING_URI to point at
a real tracking server (e.g. one backed by Postgres + S3 artifact storage)
once this moves beyond a single laptop.
"""
import os
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "crrt-prediction")


def init_mlflow(experiment: str = MLFLOW_EXPERIMENT) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
