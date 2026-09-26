"""Experiment tracking with MLflow.

Runs are stored locally at the project root (both ignored by Git):

- ``mlflow.db``: SQLite database with parameters, metrics and tags
- ``mlartifacts/``: files saved with each run (tables, figures, models)

Open the web interface from the project root with::

    mlflow ui --backend-store-uri sqlite:///mlflow.db

and go to http://127.0.0.1:5000
"""

from contextlib import contextmanager
import subprocess

import mlflow

from ms_seg.config import PROJ_ROOT

TRACKING_URI = f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_DIR = PROJ_ROOT / "mlartifacts"


def _git_commit() -> str:
    """Current Git commit, so every run is linked to the exact code that produced it."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _experiment_id(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(name, artifact_location=(ARTIFACT_DIR / name).as_uri())


@contextmanager
def start_run(experiment: str, run_name: str | None = None, params: dict | None = None, tags: dict | None = None):
    """Open an MLflow run, logging its parameters and the Git commit."""
    mlflow.set_tracking_uri(TRACKING_URI)
    with mlflow.start_run(experiment_id=_experiment_id(experiment), run_name=run_name) as run:
        mlflow.set_tags({"git_commit": _git_commit(), **(tags or {})})
        if params:
            mlflow.log_params(params)
        yield run
