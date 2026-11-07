"""Task 3 helpers: MLflow setup against DagsHub (or local), reproducibility tags,
and a hand-written per-epoch logging callback."""
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import tensorflow as tf
import yaml
from dotenv import load_dotenv

from .config import ROOT

load_dotenv(ROOT / ".env")   # MLFLOW_TRACKING_URI / USERNAME / PASSWORD live here, never in code


def setup(experiment: str):
    uri = os.getenv("MLFLOW_TRACKING_URI", f"file://{ROOT / 'mlruns'}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def dvc_data_version() -> str:
    """md5 of the tracked dataset folder, read straight from the DVC pointer file."""
    pointer = ROOT / "data" / "raw.dvc"
    if not pointer.exists():
        return "untracked"
    outs = yaml.safe_load(pointer.read_text()).get("outs", [{}])
    return outs[0].get("md5", "unknown")


def repro_tags(seed: int, extra: dict | None = None) -> dict:
    tags = {"git_commit": git_commit(), "dvc_data_version": dvc_data_version(), "seed": str(seed),
            "roll_number": "23L-0594"}
    tags.update(extra or {})
    return tags


class MLflowEpochLogger(tf.keras.callbacks.Callback):
    """Manual instrumentation: logs every metric Keras reports, plus the current LR,
    at a global step so stage 1 and stage 2 curves line up on one axis."""

    def __init__(self, step_offset: int = 0, prefix: str = ""):
        super().__init__()
        self.offset, self.prefix = step_offset, prefix

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        lr = self.model.optimizer.learning_rate
        logs["lr"] = float(lr.numpy() if hasattr(lr, "numpy") else lr)
        mlflow.log_metrics({self.prefix + k: float(v) for k, v in logs.items()},
                           step=self.offset + epoch + 1)


def log_keras_model(model, registered_name: str | None = None):
    """mlflow 2.x uses artifact_path=, 3.x prefers name=. Support both."""
    kw = {"registered_model_name": registered_name} if registered_name else {}
    try:
        return mlflow.tensorflow.log_model(model, name="model", **kw)
    except TypeError:
        return mlflow.tensorflow.log_model(model, artifact_path="model", **kw)


def write_requirements(path: Path):
    out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    path.write_text(out)
    return path
