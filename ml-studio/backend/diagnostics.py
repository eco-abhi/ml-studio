"""
Post-training diagnostics: confusion matrix, ROC, calibration, regression residuals,
learning curve, and permutation importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:
    from sklearn.calibration import calibration_curve
except ImportError:
    calibration_curve = None


def _json_float(x) -> float:
    return round(float(x), 6)


def _compute_learning_curve(
    pipe,
    X_train,
    y_train,
    max_train_samples_lc: int,
    *,
    scoring: str,
    negate_mean: bool,
) -> dict | None:
    """Subsampled learning_curve with cv=2; returns train_sizes and mean train/val scores."""
    try:
        Xt = X_train.reset_index(drop=True) if hasattr(X_train, "reset_index") else X_train
        yt_tr = pd.Series(y_train).reset_index(drop=True)
        if len(Xt) > max_train_samples_lc:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(Xt), max_train_samples_lc, replace=False)
            Xt = Xt.iloc[idx] if hasattr(Xt, "iloc") else Xt[idx]
            yt_tr = yt_tr.iloc[idx] if hasattr(yt_tr, "iloc") else yt_tr.iloc[idx]
        n_abs = len(Xt)
        raw_sizes = sorted(
            {min(n_abs, max(5, int(n_abs * p))) for p in (0.25, 0.5, 0.75, 1.0)}
        )
        if len(raw_sizes) < 2 and n_abs >= 2:
            raw_sizes = sorted({min(n_abs, max(2, n_abs // 2)), n_abs})
        train_sizes, train_scores, val_scores = learning_curve(
            pipe,
            Xt,
            yt_tr,
            train_sizes=raw_sizes,
            cv=2,
            scoring=scoring,
            n_jobs=1,
            random_state=42,
        )
        train_m = np.mean(train_scores, axis=1)
        val_m = np.mean(val_scores, axis=1)
        if negate_mean:
            train_m = -train_m
            val_m = -val_m
        return {
            "train_sizes": train_sizes.tolist(),
            "train_score_mean": np.round(train_m, 4).tolist(),
            "val_score_mean": np.round(val_m, 4).tolist(),
        }
    except Exception:
        return None


def _subsample_indices(n: int, max_n: int, seed: int = 44) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, max_n, replace=False))


def compute_scaler_baseline_compare(
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    max_rows: int = 8000,
) -> dict | None:
    """
    Fit LinearRegression + each scaler on numeric features only (train-only medians for NaN).
    For fair comparison with notebook-style workflows; independent of the saved model type.
    """
    try:
        Xtr = X_train.select_dtypes(include=[np.number])
        Xte = X_test.select_dtypes(include=[np.number])
        common = [c for c in Xtr.columns if c in Xte.columns]
        if not common:
            return None
        Xtr = Xtr[common].copy()
        Xte = Xte[common].copy()
        med = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(med)
        Xte = Xte.fillna(med)
        yt_tr = pd.Series(y_train).astype(float).values
        yt_te = pd.Series(y_test).astype(float).values
        if len(Xtr) > max_rows:
            idx = _subsample_indices(len(Xtr), max_rows, 45)
            Xtr = Xtr.iloc[idx]
            yt_tr = yt_tr[idx]
        scalers = [
            ("standard", StandardScaler()),
            ("minmax", MinMaxScaler()),
            ("robust", RobustScaler()),
        ]
        rows: list[dict] = []
        for name, scaler in scalers:
            pipe = SkPipeline([("scaler", scaler), ("lr", LinearRegression())])
            pipe.fit(Xtr, yt_tr)
            pred = pipe.predict(Xte)
            rmse = float(np.sqrt(mean_squared_error(yt_te, pred)))
            r2 = float(r2_score(yt_te, pred))
            rows.append({"scaler": name, "rmse": _json_float(rmse), "r2": _json_float(r2)})
        best = min(rows, key=lambda r: r["rmse"])
        return {"comparisons": rows, "best_scaler": best["scaler"]}
    except Exception:
        return None


def compute_diagnostics(
    pipe,
    X_train,
    X_test,
    y_train,
    y_test,
    task_type: str,
    feature_names: list[str],
    *,
    max_train_samples_lc: int = 900,
    max_test_samples_perm: int = 700,
    perm_repeats: int = 4,
) -> dict:
    """Run diagnostics on hold-out test data. May subsample for heavy estimators."""
    y_test_s = pd.Series(y_test).reset_index(drop=True)
    y_train_s = pd.Series(y_train).reset_index(drop=True)
    result: dict = {"task_type": task_type}

    if task_type == "classification":
        y_pred = pipe.predict(X_test)
        labels = np.unique(np.concatenate([y_test_s.values, np.asarray(y_pred)]))
        label_list = [str(x) for x in labels]
        cm = confusion_matrix(y_test_s, y_pred, labels=labels)
        result["confusion_matrix"] = {"labels": label_list, "matrix": cm.tolist()}

        rep = classification_report(
            y_test_s, y_pred, labels=labels, output_dict=True, zero_division=0,
        )
        report_out: dict = {}
        for k, v in rep.items():
            if isinstance(v, dict):
                report_out[k] = {
                    sk: _json_float(sv) if isinstance(sv, (float, np.floating)) else int(sv)
                    for sk, sv in v.items()
                }
            elif isinstance(v, (float, np.floating)):
                report_out[k] = _json_float(v)
        result["classification_report"] = report_out

        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_test)
                classes = np.asarray(pipe.classes_)
                if proba.shape[1] == 2:
                    pos = classes[1]
                    y_bin = (y_test_s.astype(str) == str(pos)).astype(np.int32)
                    fpr, tpr, _ = roc_curve(y_bin, proba[:, 1])
                    n = len(fpr)
                    if n > 80:
                        idx = np.linspace(0, n - 1, 80, dtype=int)
                        fpr, tpr = fpr[idx], tpr[idx]
                    auc = roc_auc_score(y_bin, proba[:, 1])
                    result["roc_curve"] = {
                        "fpr": np.round(fpr, 5).tolist(),
                        "tpr": np.round(tpr, 5).tolist(),
                        "auc": _json_float(auc),
                    }
                    if calibration_curve is not None and y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
                        try:
                            frac, mean_pred = calibration_curve(
                                y_bin, proba[:, 1], n_bins=min(10, max(3, len(y_bin) // 8)),
                            )
                            result["calibration"] = {
                                "mean_predicted": np.round(mean_pred, 5).tolist(),
                                "fraction_positives": np.round(frac, 5).tolist(),
                            }
                        except Exception:
                            pass
                elif proba.shape[1] > 2:
                    try:
                        auc_m = roc_auc_score(
                            y_test_s, proba, multi_class="ovr", average="macro",
                        )
                        result["roc_auc_macro_ovr"] = _json_float(auc_m)
                    except Exception:
                        pass
            except Exception:
                pass

        lc_cls = _compute_learning_curve(
            pipe,
            X_train,
            y_train_s,
            max_train_samples_lc,
            scoring="accuracy",
            negate_mean=False,
        )
        if lc_cls:
            lc_cls["metric"] = "accuracy"
            result["learning_curve"] = lc_cls
        else:
            result["learning_curve"] = None

    else:
        y_pred = pipe.predict(X_test)
        yt = np.asarray(y_test_s, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        res = yt - yp
        result["regression"] = {
            "rmse": _json_float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": _json_float(mean_absolute_error(yt, yp)),
            "r2": _json_float(r2_score(yt, yp)),
            "residual_mean": _json_float(np.mean(res)),
            "residual_std": _json_float(np.std(res)),
        }
        hist, edges = np.histogram(res, bins=min(24, max(8, len(res) // 20)))
        result["residual_histogram"] = {
            "counts": hist.tolist(),
            "edges": [round(float(e), 5) for e in edges],
        }

        n_sc = len(yt)
        idx = _subsample_indices(n_sc, 500, 44)
        act_s = yt[idx]
        pred_s = yp[idx]
        res_s = act_s - pred_s
        result["predicted_vs_actual"] = {
            "actual": np.round(act_s, 5).tolist(),
            "predicted": np.round(pred_s, 5).tolist(),
        }
        result["residuals_vs_predicted"] = {
            "predicted": np.round(pred_s, 5).tolist(),
            "residuals": np.round(res_s, 5).tolist(),
        }

        sb = compute_scaler_baseline_compare(X_train, X_test, y_train, y_test)
        if sb:
            result["scaler_baseline_compare"] = sb

        lc_reg = _compute_learning_curve(
            pipe,
            X_train,
            y_train_s,
            max_train_samples_lc,
            scoring="neg_root_mean_squared_error",
            negate_mean=True,
        )
        if lc_reg:
            lc_reg["metric"] = "rmse"
            result["learning_curve"] = lc_reg
        else:
            result["learning_curve"] = None

    try:
        Xp = X_test.reset_index(drop=True) if hasattr(X_test, "reset_index") else X_test
        yp = y_test_s
        if len(Xp) > max_test_samples_perm:
            rng = np.random.RandomState(43)
            idx = rng.choice(len(Xp), max_test_samples_perm, replace=False)
            Xp = Xp.iloc[idx] if hasattr(Xp, "iloc") else Xp[idx]
            yp = yp.iloc[idx] if hasattr(yp, "iloc") else yp.iloc[idx]
        scoring = "accuracy" if task_type == "classification" else "neg_root_mean_squared_error"
        pi = permutation_importance(
            pipe, Xp, yp, n_repeats=perm_repeats, random_state=42, scoring=scoring, n_jobs=1,
        )
        cols = list(Xp.columns) if hasattr(Xp, "columns") else feature_names
        order = np.argsort(pi.importances_mean)[::-1]
        result["permutation_importance"] = [
            {
                "feature": cols[i] if i < len(cols) else f"col_{i}",
                "mean": round(float(pi.importances_mean[i]), 5),
                "std": round(float(pi.importances_std[i]), 5),
            }
            for i in order[: min(40, len(order))]
        ]
    except Exception:
        result["permutation_importance"] = []

    return result
