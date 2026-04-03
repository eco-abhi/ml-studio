import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from dataclasses import dataclass, fields
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, RepeatedKFold,
    RepeatedStratifiedKFold, LeaveOneOut, ShuffleSplit,
    cross_val_score, cross_validate,
)
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from transforms import ML_PIPELINE_SPLIT_COL
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              accuracy_score, roc_auc_score, f1_score)
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

ALL_REGRESSION_MODELS     = ["linear_regression","ridge","decision_tree","random_forest","gradient_boosting"]
ALL_CLASSIFICATION_MODELS = ["logistic_regression","ridge_classifier","decision_tree","random_forest","gradient_boosting"]

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None
    RandomUnderSampler = None
    ImbPipeline = None


# ── Split config dataclass ─────────────────────────────────────────────────────

@dataclass
class SplitConfig:
    test_size:          float = 0.2
    random_state:       int   = 42
    shuffle:            bool  = True
    stratify:           bool  = True          # auto-disabled for regression
    # Validation strategy: "none" | "kfold" | "stratified_kfold" | "repeated_kfold"
    #                     | "repeated_stratified_kfold" | "shuffle_split" | "loo"
    val_strategy:       str   = "kfold"
    cv_folds:           int   = 5
    cv_repeats:         int   = 3             # for repeated strategies
    val_size:           float = 0.1           # for shuffle_split held-out fraction
    # Scaler: "standard" | "minmax" | "robust" | "none"
    scaler:             str   = "standard"
    # Classification imbalance (applied on training fold features after preprocessing)
    imbalance_method:   str   = "none"         # none | smote | random_under
    # Optional PCA after numeric/categorical preprocessing (0 = disabled)
    train_pca_components: int = 0
    # Low-cardinality object/category columns: numeric_only (drop) | target_encode | one_hot
    categorical_encoding: str = "numeric_only"
    max_category_cardinality: int = 50


def split_config_from_dict(d: dict | None) -> SplitConfig:
    """Build SplitConfig ignoring unknown keys (forward-compatible clients)."""
    if not d:
        return SplitConfig()
    allowed = {f.name for f in fields(SplitConfig)}
    return SplitConfig(**{k: v for k, v in d.items() if k in allowed})


def _make_scaler(name: str):
    return {
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(),
        "robust":   RobustScaler(),
        "none":     None,
    }.get(name, StandardScaler())


def _make_cv(cfg: SplitConfig, classification: bool):
    """Return a sklearn CV splitter (or int) based on config."""
    n = cfg.cv_folds
    r = cfg.cv_repeats
    s = cfg.random_state
    strat = cfg.stratify and classification

    if cfg.val_strategy == "none":
        return None
    if cfg.val_strategy == "kfold":
        return (StratifiedKFold(n_splits=n, shuffle=cfg.shuffle, random_state=s)
                if strat else KFold(n_splits=n, shuffle=cfg.shuffle, random_state=s))
    if cfg.val_strategy == "stratified_kfold":
        return StratifiedKFold(n_splits=n, shuffle=cfg.shuffle, random_state=s)
    if cfg.val_strategy == "repeated_kfold":
        return (RepeatedStratifiedKFold(n_splits=n, n_repeats=r, random_state=s)
                if strat else RepeatedKFold(n_splits=n, n_repeats=r, random_state=s))
    if cfg.val_strategy == "repeated_stratified_kfold":
        return RepeatedStratifiedKFold(n_splits=n, n_repeats=r, random_state=s)
    if cfg.val_strategy == "shuffle_split":
        return ShuffleSplit(n_splits=n, test_size=cfg.val_size, random_state=s)
    if cfg.val_strategy == "loo":
        return LeaveOneOut()
    return KFold(n_splits=n, shuffle=cfg.shuffle, random_state=s)


# ── Hyperparam helpers ─────────────────────────────────────────────────────────

def _hp(hyperparams: dict, model: str, key: str, default):
    return hyperparams.get(model, {}).get(key, default)

def _bool(val) -> bool:
    if isinstance(val, bool): return val
    return str(val).lower() not in ("false", "0", "no", "none", "")

def _none_if_zero(val):
    return None if val == 0 else val

def _str_or_none(val):
    if val is None or (isinstance(val, str) and val.lower() == "none"):
        return None
    return val


# ── Model builders ─────────────────────────────────────────────────────────────

def build_regression_models(hyperparams: dict) -> dict:
    h = lambda model, key, default: _hp(hyperparams, model, key, default)
    return {
        "linear_regression": LinearRegression(
            fit_intercept=_bool(h("linear_regression", "fit_intercept", True)),
        ),
        "ridge": Ridge(
            alpha=         h("ridge", "alpha",         1.0),
            max_iter=      h("ridge", "max_iter",      1000),
            solver=        h("ridge", "solver",        "auto"),
            fit_intercept= _bool(h("ridge", "fit_intercept", True)),
        ),
        "decision_tree": DecisionTreeRegressor(
            max_depth=         _none_if_zero(h("decision_tree", "max_depth",         5)),
            min_samples_split= h("decision_tree", "min_samples_split",               2),
            min_samples_leaf=  h("decision_tree", "min_samples_leaf",                1),
            criterion=         h("decision_tree", "criterion",                       "squared_error"),
            max_features=      _str_or_none(h("decision_tree", "max_features",       "sqrt")),
            ccp_alpha=         h("decision_tree", "ccp_alpha",                       0.0),
            random_state=42,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=      h("random_forest", "n_estimators",                    100),
            max_depth=         _none_if_zero(h("random_forest", "max_depth",         0)),
            min_samples_split= h("random_forest", "min_samples_split",               2),
            min_samples_leaf=  h("random_forest", "min_samples_leaf",                1),
            criterion=         h("random_forest", "criterion",                       "squared_error"),
            max_features=      _str_or_none(h("random_forest", "max_features",       "sqrt")),
            bootstrap=         _bool(h("random_forest", "bootstrap",                 True)),
            max_samples=       (lambda v: v if v < 1.0 else None)(h("random_forest", "max_samples", 1.0)),
            ccp_alpha=         h("random_forest", "ccp_alpha",                       0.0),
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=        h("gradient_boosting", "n_estimators",              100),
            learning_rate=       h("gradient_boosting", "learning_rate",             0.1),
            max_depth=           h("gradient_boosting", "max_depth",                 3),
            subsample=           h("gradient_boosting", "subsample",                 1.0),
            min_samples_split=   h("gradient_boosting", "min_samples_split",         2),
            min_samples_leaf=    h("gradient_boosting", "min_samples_leaf",          1),
            max_features=        _str_or_none(h("gradient_boosting", "max_features", "none")),
            validation_fraction= h("gradient_boosting", "validation_fraction",       0.1),
            n_iter_no_change=    _none_if_zero(h("gradient_boosting", "n_iter_no_change", 0)),
            ccp_alpha=           h("gradient_boosting", "ccp_alpha",                 0.0),
            random_state=42,
        ),
    }


def build_classification_models(hyperparams: dict) -> dict:
    h = lambda model, key, default: _hp(hyperparams, model, key, default)
    penalty = h("logistic_regression", "penalty", "l2")
    return {
        "logistic_regression": LogisticRegression(
            C=           h("logistic_regression", "C",        1.0),
            max_iter=    h("logistic_regression", "max_iter", 1000),
            penalty=     None if penalty == "none" else penalty,
            solver=      h("logistic_regression", "solver",   "lbfgs"),
            tol=         h("logistic_regression", "tol",      1e-4),
            l1_ratio=    (h("logistic_regression", "l1_ratio", 0.5) if penalty == "elasticnet" else None),
            random_state=42,
        ),
        "ridge_classifier": RidgeClassifier(
            alpha=         h("ridge_classifier", "alpha",         1.0),
            max_iter=      h("ridge_classifier", "max_iter",      1000),
            solver=        h("ridge_classifier", "solver",        "auto"),
            fit_intercept= _bool(h("ridge_classifier", "fit_intercept", True)),
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=         _none_if_zero(h("decision_tree", "max_depth",         5)),
            min_samples_split= h("decision_tree", "min_samples_split",               2),
            min_samples_leaf=  h("decision_tree", "min_samples_leaf",                1),
            criterion=         h("decision_tree", "criterion",                       "gini"),
            max_features=      _str_or_none(h("decision_tree", "max_features",       "sqrt")),
            ccp_alpha=         h("decision_tree", "ccp_alpha",                       0.0),
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=      h("random_forest", "n_estimators",                    100),
            max_depth=         _none_if_zero(h("random_forest", "max_depth",         0)),
            min_samples_split= h("random_forest", "min_samples_split",               2),
            min_samples_leaf=  h("random_forest", "min_samples_leaf",                1),
            criterion=         h("random_forest", "criterion",                       "gini"),
            max_features=      _str_or_none(h("random_forest", "max_features",       "sqrt")),
            bootstrap=         _bool(h("random_forest", "bootstrap",                 True)),
            max_samples=       (lambda v: v if v < 1.0 else None)(h("random_forest", "max_samples", 1.0)),
            ccp_alpha=         h("random_forest", "ccp_alpha",                       0.0),
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=        h("gradient_boosting", "n_estimators",              100),
            learning_rate=       h("gradient_boosting", "learning_rate",             0.1),
            max_depth=           h("gradient_boosting", "max_depth",                 3),
            subsample=           h("gradient_boosting", "subsample",                 1.0),
            min_samples_split=   h("gradient_boosting", "min_samples_split",         2),
            min_samples_leaf=    h("gradient_boosting", "min_samples_leaf",          1),
            max_features=        _str_or_none(h("gradient_boosting", "max_features", "none")),
            validation_fraction= h("gradient_boosting", "validation_fraction",       0.1),
            n_iter_no_change=    _none_if_zero(h("gradient_boosting", "n_iter_no_change", 0)),
            ccp_alpha=           h("gradient_boosting", "ccp_alpha",                 0.0),
            random_state=42,
        ),
    }


# Backward-compat exports
REGRESSION_MODELS     = build_regression_models({})
CLASSIFICATION_MODELS = build_classification_models({})


# ── Data preparation ───────────────────────────────────────────────────────────

def detect_task_type(y):
    if y.dtype == float and (y % 1 != 0).any():
        return "regression"
    if len(y.unique()) < 20:
        return "classification"
    return "regression"


def load_and_prepare_data(df, target_col, cfg: SplitConfig, task_type: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    use_pipeline_split = ML_PIPELINE_SPLIT_COL in df.columns

    if use_pipeline_split:
        X_raw = df.drop(columns=[target_col, ML_PIPELINE_SPLIT_COL], errors="ignore")
        y = df[target_col]
        lab = df[ML_PIPELINE_SPLIT_COL]
        df_clean = pd.concat([X_raw, y, lab], axis=1).dropna()
        split_lab = df_clean[ML_PIPELINE_SPLIT_COL].astype(str)
        if not split_lab.isin(["train", "test"]).all():
            raise ValueError(
                f"Column {ML_PIPELINE_SPLIT_COL!r} must be 'train' or 'test' for every row "
                "used in modeling (after dropping rows with missing features, target, or split label)."
            )
        X = df_clean.drop(columns=[target_col, ML_PIPELINE_SPLIT_COL])
        y = df_clean[target_col]
        is_train = split_lab == "train"
        is_test = split_lab == "test"
        if not is_train.any() or not is_test.any():
            raise ValueError(
                "Pipeline train/test split requires at least one row labeled train and one labeled test."
            )
    else:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        df_clean = pd.concat([X, y], axis=1).dropna()
        X = df_clean.drop(target_col, axis=1)
        y = df_clean[target_col]
        is_train = is_test = None

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_candidates = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    enc = (cfg.categorical_encoding or "numeric_only").lower()
    max_card = max(2, int(cfg.max_category_cardinality or 50))

    if enc == "numeric_only":
        X = X[numeric_cols]
        num_cols = [c for c in numeric_cols if c in X.columns]
        cat_cols: list[str] = []
    else:
        cat_cols = [c for c in cat_candidates if X[c].nunique(dropna=False) <= max_card]
        drop_cats = [c for c in cat_candidates if c not in cat_cols]
        X = X.drop(columns=drop_cats, errors="ignore")
        num_cols = [c for c in numeric_cols if c in X.columns]
        cat_cols = [c for c in cat_cols if c in X.columns]

    if use_pipeline_split:
        X_train = X.loc[is_train]
        X_test = X.loc[is_test]
        y_train = y.loc[is_train]
        y_test = y.loc[is_test]
    else:
        _stratify = None
        if cfg.stratify and task_type == "classification" and len(y.unique()) >= 2:
            _stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=   cfg.test_size,
            random_state=cfg.random_state if cfg.shuffle else None,
            shuffle=     cfg.shuffle,
            stratify=    _stratify,
        )
    feature_names = X_train.columns.tolist()
    return X_train, X_test, y_train, y_test, feature_names, num_cols, cat_cols


def _numeric_preprocessor(cfg: SplitConfig):
    scaler = _make_scaler(cfg.scaler)
    if scaler is None:
        return "passthrough"
    return SkPipeline([("scaler", scaler)])


def build_model_pipeline(
    model,
    cfg: SplitConfig,
    num_cols: list[str],
    cat_cols: list[str],
    *,
    classification: bool,
):
    """Assemble preprocessing, optional PCA, imbalance sampling, and estimator."""
    steps: list = []
    enc = (cfg.categorical_encoding or "numeric_only").lower()
    has_cat = bool(cat_cols) and enc != "numeric_only"

    if has_cat:
        num_pipe = _numeric_preprocessor(cfg)
        if enc == "target_encode":
            cat_est = TargetEncoder(random_state=cfg.random_state)
        else:
            _mc = min(100, max(10, int(cfg.max_category_cardinality or 50)))
            try:
                cat_est = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=_mc,
                )
            except TypeError:
                cat_est = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers: list = []
        if num_cols:
            transformers.append(("num", num_pipe, num_cols))
        transformers.append(("cat", cat_est, cat_cols))
        prep = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        steps.append(("prep", prep))
    else:
        scaler = _make_scaler(cfg.scaler)
        if scaler is not None:
            steps.append(("scaler", scaler))

    n_pca = int(cfg.train_pca_components or 0)
    if n_pca > 0:
        steps.append(
            ("pca", PCA(n_components=n_pca, random_state=cfg.random_state)),
        )

    imb = (cfg.imbalance_method or "none").lower()
    if classification and imb == "smote" and SMOTE is not None:
        steps.append(("smote", SMOTE(random_state=cfg.random_state)))
    elif classification and imb == "random_under" and RandomUnderSampler is not None:
        steps.append(("undersample", RandomUnderSampler(random_state=cfg.random_state)))

    steps.append(("model", model))

    uses_sampling = any(s[0] in ("smote", "undersample") for s in steps)
    if uses_sampling and ImbPipeline is not None:
        return ImbPipeline(steps)
    if uses_sampling and ImbPipeline is None:
        steps = [s for s in steps if s[0] not in ("smote", "undersample")]
        if not steps or steps[-1][0] != "model":
            steps.append(("model", model))
        return SkPipeline(steps)
    return SkPipeline(steps)


# ── Training helpers ───────────────────────────────────────────────────────────

def _cv_regression_scores(pipe, X_train, y_train, cfg: SplitConfig):
    """Return dict of cv metric means. Returns {} when val_strategy='none'."""
    cv = _make_cv(cfg, classification=False)
    if cv is None:
        return {}
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
    raw = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, error_score="raise")
    return {
        "cv_rmse": round(-raw["test_rmse"].mean(), 4),
        "cv_mae":  round(-raw["test_mae"].mean(), 4),
        "cv_r2":   round(raw["test_r2"].mean(), 4),
        "cv_std":  round(raw["test_rmse"].std(), 4),
        "cv_folds_run": len(raw["test_rmse"]),
    }


def _cv_classification_scores(pipe, X_train, y_train, cfg: SplitConfig):
    cv = _make_cv(cfg, classification=True)
    if cv is None:
        return {}
    scoring = {"accuracy": "accuracy", "f1": "f1_weighted"}
    raw = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, error_score="raise")
    return {
        "cv_accuracy": round(raw["test_accuracy"].mean(), 4),
        "cv_f1":       round(raw["test_f1"].mean(), 4),
        "cv_std":      round(raw["test_accuracy"].std(), 4),
        "cv_folds_run": len(raw["test_accuracy"]),
    }


# ── Main training functions ────────────────────────────────────────────────────

def train_regression_models(X_train, X_test, y_train, y_test, feature_names,
                             dataset_name, task_type, hyperparams: dict,
                             selected_models: list[str] | None, cfg: SplitConfig,
                             num_cols: list[str], cat_cols: list[str]):
    results = []
    all_models = build_regression_models(hyperparams)
    models = {k: v for k, v in all_models.items()
              if selected_models is None or k in selected_models}

    for name, model in models.items():
        pipe = build_model_pipeline(
            model, cfg, num_cols, cat_cols, classification=False,
        )

        with mlflow.start_run(run_name=name):
            mlflow.set_tag("dataset",        dataset_name)
            mlflow.set_tag("task_type",      task_type)
            mlflow.log_param("model",        name)
            mlflow.log_param("n_features",   len(feature_names))
            mlflow.log_param("n_train",      len(X_train))
            mlflow.log_param("n_test",       len(X_test))
            mlflow.log_param("test_size",    cfg.test_size)
            mlflow.log_param("scaler",       cfg.scaler)
            mlflow.log_param("val_strategy", cfg.val_strategy)
            mlflow.log_param("cv_folds",     cfg.cv_folds)
            mlflow.log_param("random_state", cfg.random_state)
            mlflow.log_param("imbalance_method", cfg.imbalance_method)
            mlflow.log_param("train_pca_components", cfg.train_pca_components)
            mlflow.log_param("categorical_encoding", cfg.categorical_encoding)
            for k, v in hyperparams.get(name, {}).items():
                mlflow.log_param(k, v)

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
            mae  = round(float(mean_absolute_error(y_test, preds)), 4)
            r2   = round(float(r2_score(y_test, preds)), 4)

            cv_scores = _cv_regression_scores(pipe, X_train, y_train, cfg)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae",  mae)
            mlflow.log_metric("r2",   r2)
            for k, v in cv_scores.items():
                if isinstance(v, float): mlflow.log_metric(k, v)

            mlflow.sklearn.log_model(pipe, artifact_path="model",
                                     registered_model_name=f"{dataset_name}-{name}")
            results.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2,
                             "cv_rmse": cv_scores.get("cv_rmse"),
                             **{k: v for k, v in cv_scores.items() if k != "cv_rmse"}})

    return sorted(results, key=lambda x: x["rmse"])


def train_classification_models(X_train, X_test, y_train, y_test, feature_names,
                                 dataset_name, task_type, hyperparams: dict,
                                 selected_models: list[str] | None, cfg: SplitConfig,
                                 num_cols: list[str], cat_cols: list[str]):
    results = []
    all_models = build_classification_models(hyperparams)
    models = {k: v for k, v in all_models.items()
              if selected_models is None or k in selected_models}

    for name, model in models.items():
        pipe = build_model_pipeline(
            model, cfg, num_cols, cat_cols, classification=True,
        )

        with mlflow.start_run(run_name=name):
            mlflow.set_tag("dataset",        dataset_name)
            mlflow.set_tag("task_type",      task_type)
            mlflow.log_param("model",        name)
            mlflow.log_param("n_features",   len(feature_names))
            mlflow.log_param("n_train",      len(X_train))
            mlflow.log_param("n_test",       len(X_test))
            mlflow.log_param("test_size",    cfg.test_size)
            mlflow.log_param("scaler",       cfg.scaler)
            mlflow.log_param("val_strategy", cfg.val_strategy)
            mlflow.log_param("cv_folds",     cfg.cv_folds)
            mlflow.log_param("random_state", cfg.random_state)
            mlflow.log_param("imbalance_method", cfg.imbalance_method)
            mlflow.log_param("train_pca_components", cfg.train_pca_components)
            mlflow.log_param("categorical_encoding", cfg.categorical_encoding)
            for k, v in hyperparams.get(name, {}).items():
                mlflow.log_param(k, v)

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            probs = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

            accuracy = round(float(accuracy_score(y_test, preds)), 4)
            f1       = round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4)
            cv_scores = _cv_classification_scores(pipe, X_train, y_train, cfg)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            for k, v in cv_scores.items():
                if isinstance(v, float): mlflow.log_metric(k, v)

            result = {"model": name, "accuracy": accuracy, "f1_score": f1,
                      **{k: v for k, v in cv_scores.items()}}

            if len(np.unique(y_test)) == 2 and probs is not None:
                roc_auc = round(float(roc_auc_score(y_test, probs[:, 1])), 4)
                mlflow.log_metric("roc_auc", roc_auc)
                result["roc_auc"] = roc_auc

            mlflow.sklearn.log_model(pipe, artifact_path="model",
                                     registered_model_name=f"{dataset_name}-{name}")
            results.append(result)

    return sorted(results, key=lambda x: x["accuracy"], reverse=True)


def train_all(df, target_col, dataset_name, task_type=None,
              hyperparams: dict | None = None,
              selected_models: list[str] | None = None,
              split_config: dict | None = None):

    hyperparams = hyperparams or {}
    cfg = split_config_from_dict(split_config)

    if task_type is None:
        # peek at target to auto-detect before splitting
        y_peek = df[target_col].dropna()
        task_type = detect_task_type(y_peek)

    X_train, X_test, y_train, y_test, feature_names, num_cols, cat_cols = load_and_prepare_data(
        df, target_col, cfg, task_type
    )

    try:
        mlflow.create_experiment(f"dataset-{dataset_name}")
    except Exception:
        pass
    mlflow.set_experiment(f"dataset-{dataset_name}")

    if task_type == "classification":
        results = train_classification_models(
            X_train, X_test, y_train, y_test, feature_names,
            dataset_name, task_type, hyperparams, selected_models, cfg,
            num_cols, cat_cols,
        )
    else:
        results = train_regression_models(
            X_train, X_test, y_train, y_test, feature_names,
            dataset_name, task_type, hyperparams, selected_models, cfg,
            num_cols, cat_cols,
        )

    n_tot = len(X_train) + len(X_test)
    test_frac = (len(X_test) / n_tot) if n_tot else cfg.test_size
    split_cfg_out = {
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
        "train_pca_components":  cfg.train_pca_components,
        "categorical_encoding":  cfg.categorical_encoding,
        "max_category_cardinality": cfg.max_category_cardinality,
    }
    if ML_PIPELINE_SPLIT_COL in df.columns:
        split_cfg_out["holdout_from_transforms"] = True
        split_cfg_out["test_size_effective"] = round(float(test_frac), 4)

    return {
        "task_type":     task_type,
        "n_features":    len(feature_names),
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "feature_names": feature_names,
        "split_config":  split_cfg_out,
        "results":       results,
    }
