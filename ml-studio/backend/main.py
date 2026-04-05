import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse

import db
from eda import (
    compute_boxplots,
    compute_categorical_stats,
    compute_correlation_matrix,
    compute_eda,
    compute_feature_target,
    compute_health,
    compute_missing_values,
    compute_outliers,
    compute_pairplot,
    compute_scatter,
    compute_skewness,
    compute_target_analysis,
)
from diagnostics import compute_diagnostics
from predict import get_feature_importances, load_model, run_prediction
from train import CLASSIFICATION_MODELS, REGRESSION_MODELS, detect_task_type, load_and_prepare_data, train_all
from tune import run_tuning
from transforms import apply_transforms, preview_transforms, validate_transform_steps
from url_safety import fetch_url_bytes

load_dotenv()

log = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY", "").strip()
_origins_raw = os.getenv("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if _origins_raw == "*" else [o.strip() for o in _origins_raw.split(",") if o.strip()]


def _public_error_detail(exc: Exception) -> str:
    expose = os.getenv("EXPOSE_ERROR_DETAILS", "true").lower() in ("1", "true", "yes")
    return str(exc) if expose else "Request failed"

# ── MLflow tracking URI ───────────────────────────────────────────────────────
# Order: MLFLOW_TRACKING_URI → DagHub → sqlite at a stable path (backend/mlruns.db).
# Relative sqlite:///mlruns.db breaks when the API cwd ≠ where you run `mlflow ui`.


def _configure_mlflow_tracking() -> tuple[str, bool]:
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


_MLFLOW_URI, _MLFLOW_IS_REMOTE = _configure_mlflow_tracking()
mlflow.set_tracking_uri(_MLFLOW_URI)
log.info("MLflow tracking URI: %s", _MLFLOW_URI)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ML Studio API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _api_key_guard(request: Request, call_next):
    if not API_KEY:
        return await call_next(request)
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path == "/health":
        return await call_next(request)
    key = request.headers.get("x-api-key", "")
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        key = key or auth[7:].strip()
    if key != API_KEY:
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_dataset(dataset_id: str) -> dict:
    dataset = db.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    return dataset


def _load_df(dataset: dict) -> pd.DataFrame:
    """Load the active CSV (transformed version if it exists, else original)."""
    path = dataset.get("transformed_path") or dataset["storage_path"]
    return db.download_df(path)


def _load_eda_df(dataset: dict, data_source: str | None) -> pd.DataFrame:
    """Load dataframe for EDA. data_source=original forces the uploaded CSV; otherwise active (transformed if present)."""
    if data_source and str(data_source).lower() == "original":
        return db.download_df(dataset["storage_path"])
    return _load_df(dataset)


# ── Upload / preview ──────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(None), url: str = None):
    try:
        dataset_id = str(uuid.uuid4())[:8]

        if file:
            content = await file.read()
            filename = file.filename or "data.csv"
        elif url:
            content, filename = fetch_url_bytes(url, timeout=30)
        else:
            raise HTTPException(status_code=400, detail="Either file or url must be provided")

        # Quick validation before storing
        df = pd.read_csv(pd.io.common.BytesIO(content) if not isinstance(content, bytes) else __import__("io").BytesIO(content))
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")

        storage_path = db.upload_csv(dataset_id, content, filename)
        db.upsert_dataset(dataset_id, {
            "filename":     filename,
            "storage_path": storage_path,
            "columns":      df.columns.tolist(),
            "shape":        list(df.shape),
            "dtypes":       {col: str(df[col].dtype) for col in df.columns},
            "created_at":   datetime.now().isoformat(),
        })

        return {
            "dataset_id": dataset_id,
            "filename":   filename,
            "shape":      list(df.shape),
            "columns":    df.columns.tolist(),
            "storage":    db.storage_mode(),
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/preview/{dataset_id}")
def preview_dataset(dataset_id: str, limit: int = 10):
    try:
        dataset = _require_dataset(dataset_id)
        df = _load_df(dataset)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return {
            "columns":         df.columns.tolist(),
            "numeric_columns": numeric_cols,
            "dtypes":          {col: str(df[col].dtype) for col in df.columns},
            "preview":         df.head(limit).values.tolist(),
            "shape":           list(df.shape),
            "missing":         {col: int(df[col].isnull().sum()) for col in df.columns},
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


# ── Training ──────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    hyperparams:  dict = Field(default_factory=dict)
    models:       list[str] | None = None
    split_config: dict = Field(default_factory=dict)


@app.post("/train/{dataset_id}")
def train_models(dataset_id: str, target_col: str, task_type: str = None,
                 body: TrainRequest = TrainRequest()):
    try:
        dataset = _require_dataset(dataset_id)
        df = _load_df(dataset)

        if task_type is None:
            task_type = detect_task_type(df[target_col])

        results = train_all(df, target_col, dataset_id, task_type,
                            hyperparams=body.hyperparams,
                            selected_models=body.models or None,
                            split_config=body.split_config or {})

        db.upsert_dataset(dataset_id, {
            **dataset,
            "task_type":    results["task_type"],
            "target_col":   target_col,
            "features":     results["feature_names"],
            "split_config": results["split_config"],
        })

        return results
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


# ── Experiments ───────────────────────────────────────────────────────────────

@app.get("/experiments/{dataset_id}")
def get_experiments(dataset_id: str):
    try:
        _require_dataset(dataset_id)
        client = mlflow.tracking.MlflowClient()
        exp_name = f"dataset-{dataset_id}"
        try:
            exp = client.get_experiment_by_name(exp_name)
            if not exp:
                return {"runs": []}
        except Exception:
            return {"runs": []}

        runs = list(client.search_runs(exp.experiment_id))
        runs.sort(key=lambda r: r.info.start_time or 0, reverse=True)
        result_runs = []
        for r in runs:
            # Numeric hyperparams (skip non-numeric params like "model")
            numeric_params = {}
            for k, v in r.data.params.items():
                if k == "model":
                    continue
                try:
                    numeric_params[k] = float(v)
                except (ValueError, TypeError):
                    numeric_params[k] = v

            run_data: dict = {
                "run_id":     r.info.run_id,
                "model":      r.data.params.get("model", "unknown"),
                "status":     r.info.status,
                "params":     numeric_params,
                "started_at": r.info.start_time,
            }
            for metric in ["rmse", "mae", "r2", "cv_rmse", "accuracy", "f1_score", "roc_auc"]:
                if metric in r.data.metrics:
                    run_data[metric] = r.data.metrics[metric]
            result_runs.append(run_data)

        return {"runs": result_runs}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


# ── Feature importances ───────────────────────────────────────────────────────

@app.get("/diagnostics/{dataset_id}/{model_name}")
def get_model_diagnostics(dataset_id: str, model_name: str):
    """Confusion matrix / ROC / calibration (classification), residuals & learning curve (regression), permutation importance."""
    try:
        dataset = _require_dataset(dataset_id)
        if "target_col" not in dataset:
            raise HTTPException(status_code=400, detail="Train on this dataset before running diagnostics.")

        df = _load_df(dataset)
        pipe = load_model(model_name, dataset_id)
        if pipe is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        from train import split_config_from_dict

        task_type = dataset.get("task_type", "regression")
        sc = split_config_from_dict(dataset.get("split_config") or {})
        X_train, X_test, y_train, y_test, _feature_names, _num_cols, _cat_cols = load_and_prepare_data(
            df, dataset["target_col"], sc, task_type
        )
        names = list(X_test.columns)
        return compute_diagnostics(
            pipe, X_train, X_test, y_train, y_test, task_type, names,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/importances/{dataset_id}/{model_name}")
def get_importances(dataset_id: str, model_name: str):
    try:
        dataset = _require_dataset(dataset_id)
        if "target_col" not in dataset:
            raise HTTPException(status_code=400, detail="Model not yet trained on this dataset")

        df = _load_df(dataset)
        pipe = load_model(model_name, dataset_id)
        if pipe is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        from train import split_config_from_dict
        task_type = dataset.get("task_type", "regression")
        cfg_dict = dataset.get("split_config") or {}
        sc = split_config_from_dict(cfg_dict)
        X_train, X_test, _, _, feature_names, _num_cols, _cat_cols = load_and_prepare_data(
            df, dataset["target_col"], sc, task_type
        )
        return get_feature_importances(pipe, X_train, X_test, feature_names, model_name)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


# ── Predict ───────────────────────────────────────────────────────────────────

@app.post("/predict/{dataset_id}/{model_name}")
def predict(dataset_id: str, model_name: str, data: dict):
    try:
        dataset = _require_dataset(dataset_id)
        if "target_col" not in dataset:
            raise HTTPException(status_code=400, detail="Model not yet trained on this dataset")

        pipe = load_model(model_name, dataset_id)
        if pipe is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        feature_names = dataset["features"]
        return run_prediction(pipe, data, feature_names)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


# ── EDA ───────────────────────────────────────────────────────────────────────

@app.get("/eda/{dataset_id}")
def get_eda(dataset_id: str, data_source: str | None = None):
    try:
        dataset = _require_dataset(dataset_id)
        df = _load_eda_df(dataset, data_source)
        return compute_eda(df)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/eda/{dataset_id}/matrix")
def get_correlation_matrix(dataset_id: str, data_source: str | None = None):
    try:
        df = _load_eda_df(_require_dataset(dataset_id), data_source)
        return compute_correlation_matrix(df)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/eda/{dataset_id}/missing")
def get_missing_values(dataset_id: str, data_source: str | None = None):
    try:
        df = _load_eda_df(_require_dataset(dataset_id), data_source)
        return compute_missing_values(df)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/eda/{dataset_id}/outliers")
def get_outliers(dataset_id: str, data_source: str | None = None):
    try:
        df = _load_eda_df(_require_dataset(dataset_id), data_source)
        return compute_outliers(df)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/eda/{dataset_id}/scatter")
def get_scatter(dataset_id: str, col_x: str, col_y: str, target: str = None, data_source: str | None = None):
    try:
        df = _load_eda_df(_require_dataset(dataset_id), data_source)
        return compute_scatter(df, col_x, col_y, target)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/eda/{dataset_id}/categorical")
def get_categorical(dataset_id: str, data_source: str | None = None):
    try:
        df = _load_eda_df(_require_dataset(dataset_id), data_source)
        return compute_categorical_stats(df)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/eda/{dataset_id}/health")
def get_health(dataset_id: str, data_source: str | None = None):
    try:
        df = _load_eda_df(_require_dataset(dataset_id), data_source)
        return compute_health(df)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/eda/{dataset_id}/target")
def get_target_analysis(dataset_id: str, target_col: str, data_source: str | None = None):
    try:
        df = _load_eda_df(_require_dataset(dataset_id), data_source)
        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{target_col}' not found")
        return compute_target_analysis(df, target_col)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/eda/{dataset_id}/skewness")
def get_skewness(dataset_id: str, data_source: str | None = None):
    try:
        df = _load_eda_df(_require_dataset(dataset_id), data_source)
        return compute_skewness(df)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/eda/{dataset_id}/boxplot")
def get_boxplots(dataset_id: str, columns: str = None, data_source: str | None = None):
    try:
        df   = _load_eda_df(_require_dataset(dataset_id), data_source)
        cols = [c.strip() for c in columns.split(",")] if columns else None
        return compute_boxplots(df, cols)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/eda/{dataset_id}/pairplot")
def get_pairplot(dataset_id: str, columns: str = None, data_source: str | None = None):
    try:
        df   = _load_eda_df(_require_dataset(dataset_id), data_source)
        cols = [c.strip() for c in columns.split(",")] if columns else None
        return compute_pairplot(df, cols)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/eda/{dataset_id}/feature-target")
def get_feature_target(dataset_id: str, target_col: str, data_source: str | None = None):
    try:
        df = _load_eda_df(_require_dataset(dataset_id), data_source)
        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{target_col}' not found")
        return compute_feature_target(df, target_col)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


# ── Transforms ────────────────────────────────────────────────────────────────

@app.post("/transform/{dataset_id}/preview")
def transform_preview(dataset_id: str, body: dict):
    """Return before/after preview rows without saving.
    If from_original=True, preview applies steps to the original upload (same as apply with from_original).
    """
    try:
        dataset = _require_dataset(dataset_id)
        from_original = body.get("from_original", False)
        if from_original:
            df = db.download_df(dataset["storage_path"])
        else:
            df = _load_df(dataset)
        steps = body.get("steps", [])
        existing = dataset.get("transform_steps", [])
        all_steps = steps if from_original else existing + steps
        validate_transform_steps(all_steps)
        return preview_transforms(df, steps, n=body.get("n", 5))
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.post("/transform/{dataset_id}/apply")
def transform_apply(dataset_id: str, body: dict):
    """Apply transforms and save the transformed CSV as a new version.
    If from_original=True, resets to the original CSV and applies all given steps fresh.
    """
    try:
        dataset = _require_dataset(dataset_id)
        steps = body.get("steps", [])
        from_original = body.get("from_original", False)
        existing_steps = dataset.get("transform_steps", [])
        all_steps = steps if from_original else existing_steps + steps
        validate_transform_steps(all_steps)
        # Load from original if requested (used for individual step revert)
        if from_original:
            df = db.download_df(dataset["storage_path"])
        else:
            df = _load_df(dataset)
        transformed = apply_transforms(df, steps)

        # Persist the transformed CSV
        csv_bytes = transformed.to_csv(index=False).encode()
        transformed_path = db.upload_csv(dataset_id, csv_bytes, "transformed.csv")

        from datetime import datetime as _dt

        db.upsert_dataset(dataset_id, {
            **dataset,
            "transformed_path":     transformed_path,
            "columns":              transformed.columns.tolist(),
            "shape":                list(transformed.shape),
            "dtypes":               {col: str(transformed[col].dtype) for col in transformed.columns},
            "transform_steps":      all_steps,
            "transform_applied_at": _dt.now().isoformat(),
        })

        return {
            "shape":         list(transformed.shape),
            "columns":       transformed.columns.tolist(),
            "steps_applied": len(all_steps),
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/transform/{dataset_id}/history")
def transform_history(dataset_id: str):
    """Return the list of transform steps that were last applied."""
    try:
        dataset = _require_dataset(dataset_id)
        return {
            "steps":      dataset.get("transform_steps", []),
            "applied_at": dataset.get("transform_applied_at"),
            "active":     "transformed_path" in dataset,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.post("/transform/{dataset_id}/reset")
def transform_reset(dataset_id: str):
    """Revert to the original uploaded CSV."""
    try:
        dataset = _require_dataset(dataset_id)
        if "transformed_path" in dataset:
            drop_keys = {"transformed_path", "transform_steps", "transform_applied_at"}
            updated = {k: v for k, v in dataset.items() if k not in drop_keys}
            db.upsert_dataset(dataset_id, updated)
        return {"status": "reset to original"}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


# ── Misc ──────────────────────────────────────────────────────────────────────

@app.post("/tune/{dataset_id}")
def tune_model(dataset_id: str, target_col: str, model: str,
               n_trials: int = 30, task_type: str = None, body: TrainRequest = TrainRequest()):
    try:
        dataset = _require_dataset(dataset_id)
        df = _load_df(dataset)
        if task_type is None:
            task_type = detect_task_type(df[target_col])
        result = run_tuning(
            df, target_col, dataset_id, model, task_type,
            n_trials=n_trials,
            split_config=body.split_config or {},
        )
        db.upsert_dataset(dataset_id, {
            **dataset,
            "task_type":    task_type,
            "target_col":   target_col,
            "features":     result["feature_names"],
            "split_config": result["split_config"],
        })
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Request failed")
        raise HTTPException(status_code=400, detail=_public_error_detail(e))


@app.get("/datasets")
def list_datasets():
    return {"datasets": db.list_datasets()}


@app.get("/mlflow-info")
def mlflow_info():
    uri = mlflow.get_tracking_uri()
    is_remote = uri.lower().startswith(("http://", "https://"))
    ui_url = uri if is_remote else "http://localhost:5000"
    local_ui_command = None
    if not is_remote:
        # Quote for POSIX shells; Windows users can paste URI without quotes if needed.
        safe = uri.replace('"', '\\"')
        local_ui_command = f'mlflow ui --backend-store-uri "{safe}" --port 5000'
    return {
        "tracking_uri": uri,
        "ui_url":       ui_url,
        "is_remote":    is_remote,
        "local_ui_command": local_ui_command,
    }


@app.get("/health")
def health():
    return {"status": "ok", "storage": db.storage_mode()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
