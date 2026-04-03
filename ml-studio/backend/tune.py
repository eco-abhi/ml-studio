"""
Optuna hyperparameter tuning for any model in the ML Studio.

Endpoint usage:
  POST /tune/{dataset_id}?target_col=quality&model=random_forest&n_trials=30&task_type=regression
"""

import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score,
)
from transforms import ML_PIPELINE_SPLIT_COL
from train import (
    SplitConfig,
    build_classification_models,
    build_model_pipeline,
    build_regression_models,
    load_and_prepare_data,
    split_config_from_dict,
)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Search spaces ─────────────────────────────────────────────────────────────

def _suggest(trial: optuna.Trial, model: str, task_type: str) -> dict:
    """Return a param dict suggested by the trial for the given model."""
    if model == "random_forest":
        return {
            "n_estimators":      trial.suggest_int("n_estimators",      50, 400, step=10),
            "max_depth":         trial.suggest_int("max_depth",          2,  20),
            "min_samples_split": trial.suggest_int("min_samples_split",  2,  20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",   1,  10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", "none"]),
        }
    if model == "gradient_boosting":
        return {
            "n_estimators":   trial.suggest_int("n_estimators",   50, 300, step=10),
            "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":      trial.suggest_int("max_depth",       2,   8),
            "subsample":      trial.suggest_float("subsample",     0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if model == "decision_tree":
        return {
            "max_depth":         trial.suggest_int("max_depth",         2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",  1, 10),
            "ccp_alpha":         trial.suggest_float("ccp_alpha",       0.0, 0.02),
        }
    if model == "ridge":
        return {
            "alpha": trial.suggest_float("alpha", 0.001, 100.0, log=True),
        }
    if model == "logistic_regression":
        return {
            "C":        trial.suggest_float("C",        0.001, 100.0, log=True),
            "max_iter": trial.suggest_int("max_iter",   200,   2000,  step=100),
        }
    if model == "ridge_classifier":
        return {
            "alpha": trial.suggest_float("alpha", 0.001, 100.0, log=True),
        }
    if model == "linear_regression":
        return {}  # no hyperparameters to tune
    return {}


# ── Objective ─────────────────────────────────────────────────────────────────

def _make_objective(
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    task_type: str,
    cfg: SplitConfig,
    num_cols: list[str],
    cat_cols: list[str],
):
    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, model_name, task_type)

        hp = {model_name: params}
        if task_type == "classification":
            model_map = build_classification_models(hp)
        else:
            model_map = build_regression_models(hp)

        model = model_map.get(model_name)
        if model is None:
            raise optuna.exceptions.TrialPruned()

        pipe = build_model_pipeline(
            model,
            cfg,
            num_cols,
            cat_cols,
            classification=(task_type == "classification"),
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        if task_type == "regression":
            # minimise RMSE
            return float(np.sqrt(mean_squared_error(y_test, preds)))
        else:
            # maximise accuracy → return negative so Optuna minimises
            return -float(accuracy_score(y_test, preds))

    return objective


# ── Main entry point ──────────────────────────────────────────────────────────

def run_tuning(
    df,
    target_col: str,
    dataset_name: str,
    model_name: str,
    task_type: str,
    n_trials: int = 30,
    split_config: dict | None = None,
) -> dict:
    cfg = split_config_from_dict(split_config)
    X_train, X_test, y_train, y_test, feature_names, num_cols, cat_cols = load_and_prepare_data(
        df, target_col, cfg, task_type
    )

    direction = "minimize"  # RMSE for regression, negative accuracy for classification
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
    obj = _make_objective(
        X_train, y_train, X_test, y_test, model_name, task_type, cfg, num_cols, cat_cols,
    )
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_value  = study.best_value
    if task_type == "classification":
        best_value = -best_value  # convert back to accuracy

    # Rebuild and evaluate the best model
    hp = {model_name: best_params}
    if task_type == "classification":
        model_map = build_classification_models(hp)
    else:
        model_map = build_regression_models(hp)

    model = model_map[model_name]
    best_pipe = build_model_pipeline(
        model, cfg, num_cols, cat_cols, classification=(task_type == "classification"),
    )
    best_pipe.fit(X_train, y_train)
    preds = best_pipe.predict(X_test)

    metrics: dict = {}
    if task_type == "regression":
        metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
            "mae":  round(float(np.mean(np.abs(y_test.values - preds))), 4),
            "r2":   round(float(r2_score(y_test, preds)), 4),
        }
    else:
        acc = round(float(accuracy_score(y_test, preds)), 4)
        f1  = round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4)
        metrics = {"accuracy": acc, "f1_score": f1}

    # Log best run to MLflow
    try:
        mlflow.create_experiment(f"dataset-{dataset_name}")
    except Exception:
        pass
    mlflow.set_experiment(f"dataset-{dataset_name}")
    with mlflow.start_run(run_name=f"{model_name}_optuna"):
        mlflow.set_tag("tuning", "optuna")
        mlflow.set_tag("dataset", dataset_name)
        mlflow.set_tag("task_type", task_type)
        mlflow.log_param("model", model_name)
        mlflow.log_param("n_trials", n_trials)
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(
            best_pipe, artifact_path="model",
            registered_model_name=f"{dataset_name}-{model_name}"
        )

    # Trial history for the chart
    trials = [
        {
            "number":  t.number,
            "value":   round(-t.value if task_type == "classification" else t.value, 4),
            "params":  t.params,
            "state":   t.state.name,
        }
        for t in study.trials
        if t.state.name == "COMPLETE"
    ]

    n_tot = len(X_train) + len(X_test)
    test_frac = (len(X_test) / n_tot) if n_tot else cfg.test_size
    tune_split_cfg = {
        "test_size":    cfg.test_size,
        "random_state": cfg.random_state,
        "shuffle":      cfg.shuffle,
        "stratify":     cfg.stratify,
        "val_strategy": cfg.val_strategy,
        "cv_folds":     cfg.cv_folds,
        "cv_repeats":   cfg.cv_repeats,
        "val_size":     cfg.val_size,
        "scaler":       cfg.scaler,
        "imbalance_method":        cfg.imbalance_method,
        "train_pca_components":    cfg.train_pca_components,
        "categorical_encoding":    cfg.categorical_encoding,
        "max_category_cardinality": cfg.max_category_cardinality,
    }
    if ML_PIPELINE_SPLIT_COL in df.columns:
        tune_split_cfg["holdout_from_transforms"] = True
        tune_split_cfg["test_size_effective"] = round(float(test_frac), 4)

    return {
        "model":         model_name,
        "task_type":     task_type,
        "n_trials":      n_trials,
        "best_params":   best_params,
        "best_value":    round(best_value, 4),
        "metric_name":   "rmse" if task_type == "regression" else "accuracy",
        "metrics":       metrics,
        "trials":        trials,
        "feature_names": feature_names,
        "split_config":  tune_split_cfg,
    }
