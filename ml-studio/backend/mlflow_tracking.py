"""Resolve MLflow tracking URI the same way as the API (env + default sqlite path)."""

from __future__ import annotations

import os
from pathlib import Path


def configure_mlflow_tracking() -> tuple[str, bool]:
    """Return (tracking_uri, is_remote_http)."""
    explicit = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if explicit:
        return explicit, explicit.lower().startswith(("http://", "https://"))

    dag = os.getenv("DAGSHUB_MLFLOW_URI", "").strip()
    token = os.getenv("DAGSHUB_TOKEN", "").strip()
    if dag and token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        return dag, True

    custom = os.getenv("MLFLOW_SQLITE_PATH", "").strip()
    if custom:
        db_path = Path(custom).expanduser().resolve()
    else:
        db_path = (Path(__file__).resolve().parent / "mlruns.db").resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # SQLAlchemy absolute POSIX path: sqlite:////abs/path (three slashes + path starting with /)
    uri = "sqlite:///" + db_path.as_posix()
    return uri, False
