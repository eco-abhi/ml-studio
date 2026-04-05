import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np


def load_model(model_name: str, dataset_name: str, run_id: str | None = None):
    """Load sklearn pipeline from MLflow: specific run_id, or latest run for dataset+model."""
    client = mlflow.tracking.MlflowClient()
    exp_name = f"dataset-{dataset_name}"
    try:
        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            return None
    except Exception:
        return None

    if run_id:
        run_id = run_id.strip()
        try:
            run = client.get_run(run_id)
        except Exception:
            return None
        if run.info.experiment_id != exp.experiment_id:
            return None
        if run.data.params.get("model") != model_name:
            return None
        model_uri = f"runs:/{run_id}/model"
        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception:
            return None

    runs = client.search_runs(
        exp.experiment_id,
        filter_string=f"params.model = '{model_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None

    rid = runs[0].info.run_id
    model_uri = f"runs:/{rid}/model"
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        return None


def get_feature_importances(pipe, X_train, X_test, feature_names, model_name):
    """Get feature importances using built-in or coefficient magnitude."""
    inner = pipe.named_steps["model"]

    if hasattr(inner, "feature_importances_"):
        imps = inner.feature_importances_
    else:
        try:
            coefs = inner.coef_
            if coefs.ndim > 1:
                coefs = np.abs(coefs).mean(axis=0)
            else:
                coefs = np.abs(coefs)
            scaler = pipe.named_steps.get("scaler")
            if scaler and hasattr(scaler, "scale_"):
                imps = coefs * scaler.scale_
            else:
                imps = coefs
        except Exception:
            imps = np.ones(len(feature_names)) / len(feature_names)

    pairs = sorted(zip(feature_names, imps.tolist()), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": round(v, 4)} for f, v in pairs]


def run_prediction(pipe, features_dict, feature_names):
    """Run inference on a single sample."""
    X = pd.DataFrame([features_dict])[feature_names]
    pred = pipe.predict(X)[0]

    confidence = None
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(X)[0]
        confidence = round(float(max(probs)), 4)

    return {
        "prediction": round(float(pred), 4) if isinstance(pred, (int, float, np.number)) else pred,
        "confidence": confidence,
    }
