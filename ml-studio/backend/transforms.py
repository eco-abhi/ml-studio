"""
Preprocessing / transformation pipeline.

Each step is a dict with at least a "type" key.
Supported steps:

  {"type": "drop_columns",   "columns": [...]}
  {"type": "impute",         "columns": [...], "strategy": "mean|median|mode|zero"}
  {"type": "one_hot_encode", "columns": [...]}
  {"type": "label_encode",   "columns": [...]}
  {"type": "clip_outliers",  "columns": [...], "method": "iqr|zscore"}
  {"type": "scale",          "columns": [...], "method": "standard|minmax|robust"}
  {"type": "rename_columns",    "mapping": {"old_name": "new_name", ...}}
  {"type": "math_transform",   "columns": [...], "method": "log1p|sqrt|square|reciprocal|abs"}
  {"type": "fix_skewness",     "columns": [...], "method": "auto|log1p|sqrt|box_cox|yeo_johnson", "threshold": 0.5}
  {"type": "bin_numeric",      "columns": [...], "n_bins": 5, "strategy": "equal_width|quantile"}
  {"type": "drop_duplicates",  "columns": [...], "keep": "first|last|none"}   # columns=[] → all
  {"type": "drop_nulls",       "columns": [...], "how": "any|all"}            # columns=[] → all
  {"type": "frequency_encode", "columns": [...], "normalize": true}
  {"type": "cast_dtype",       "columns": [...], "dtype": "float|int|str"}
  {"type": "polynomial_features", "columns": [...], "degree": 2, "interaction_only": false, "include_bias": false}
  {"type": "extract_datetime", "columns": [...], "parts": ["year","month","day","hour","dow","doy"]}
  {"type": "pca_projection", "columns": [...], "n_components": 3, "prefix": "PC_", "drop_original": false}
  {"type": "tfidf_column", "column": "text_col", "max_features": 50, "ngram_max": 1}
  {"type": "derive_numeric", "column_a": "a", "column_b": "b", "op": "add|subtract|multiply|divide", "output_column": "feat"}
  {"type": "target_encode_dataset", "columns": [...], "target_column": "y"}  # uses full data — can leak; prefer Train tab encoding
  {"type": "train_test_split", "target_column": "y", "test_size": 0.2, "random_state": 42, "shuffle": true, "stratify": true}
  # Adds column ML_PIPELINE_SPLIT_COL ("__ml_split__") with "train" / "test". May appear anywhere; later steps should not drop it if the Train tab should use it. Rows with null target are not assigned.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

# Reserved column: rows labeled for holdout when training (see train.load_and_prepare_data).
ML_PIPELINE_SPLIT_COL = "__ml_split__"


def validate_transform_steps(steps: list[dict]) -> None:
    """Enforce at most one train/test split step."""
    split_idxs = [i for i, s in enumerate(steps) if s.get("type") == "train_test_split"]
    if len(split_idxs) > 1:
        raise ValueError("Only one train/test split step is allowed in the pipeline.")


def _stratify_ok_for_split(y: pd.Series) -> bool:
    s = y.dropna()
    if len(s) < 2 or len(s.unique()) < 2:
        return False
    if pd.api.types.is_float_dtype(s) and (s % 1 != 0).any():
        return False
    if len(s.unique()) >= 20:
        return False
    return True


def apply_transforms(df: pd.DataFrame, steps: list[dict]) -> pd.DataFrame:
    df = df.copy()
    for step in steps:
        kind = step.get("type", "")
        cols: list[str] = [c for c in step.get("columns", []) if c in df.columns]

        if kind == "drop_columns":
            df = df.drop(columns=cols, errors="ignore")

        elif kind == "impute":
            strategy = step.get("strategy", "mean")
            for col in cols:
                if strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mode":
                    mode = df[col].mode()
                    df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "")
                elif strategy == "zero":
                    df[col] = df[col].fillna(0)

        elif kind == "one_hot_encode":
            df = pd.get_dummies(df, columns=cols, drop_first=True, dtype=int)

        elif kind == "label_encode":
            le = LabelEncoder()
            for col in cols:
                df[col] = le.fit_transform(df[col].astype(str))

        elif kind == "clip_outliers":
            method = step.get("method", "iqr")
            for col in cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                if method == "iqr":
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q3 - q1
                    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                elif method == "zscore":
                    mean, std = df[col].mean(), df[col].std()
                    if std > 0:
                        df[col] = df[col].clip(mean - 3 * std, mean + 3 * std)

        elif kind == "scale":
            method = step.get("method", "standard")
            scaler_map = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
            }
            scaler = scaler_map.get(method, StandardScaler())
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                df[num_cols] = scaler.fit_transform(df[num_cols])

        elif kind == "rename_columns":
            mapping = step.get("mapping", {})
            df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns and v and v.strip()})

        elif kind == "math_transform":
            method = step.get("method", "log1p")
            for col in cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                if method == "log1p":
                    df[col] = np.log1p(df[col].clip(lower=0))
                elif method == "sqrt":
                    df[col] = np.sqrt(df[col].clip(lower=0))
                elif method == "square":
                    df[col] = df[col] ** 2
                elif method == "reciprocal":
                    df[col] = 1.0 / df[col].replace(0, np.nan)
                elif method == "abs":
                    df[col] = df[col].abs()

        elif kind == "fix_skewness":
            method = step.get("method", "auto")
            threshold = float(step.get("threshold", 0.5))
            for col in cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                s = df[col].dropna()
                if len(s) < 3:
                    continue
                skew = float(s.skew())
                if abs(skew) < threshold:
                    continue  # not skewed enough to bother

                chosen = method
                if method == "auto":
                    # Positive skew: prefer log1p if all values ≥ 0, else yeo-johnson
                    # Negative skew: reflect then log1p
                    if skew > 0 and s.min() >= 0:
                        chosen = "log1p"
                    else:
                        chosen = "yeo_johnson"

                try:
                    if chosen == "log1p":
                        df[col] = np.log1p(df[col].clip(lower=0))
                    elif chosen == "sqrt":
                        df[col] = np.sqrt(df[col].clip(lower=0))
                    elif chosen == "box_cox":
                        # box-cox requires strictly positive values
                        pt = PowerTransformer(method="box-cox")
                        vals = df[[col]].astype(float)
                        shift = max(0, -vals[col].min() + 1e-6)
                        vals[col] = vals[col] + shift
                        df[col] = pt.fit_transform(vals).ravel()
                    elif chosen == "yeo_johnson":
                        pt = PowerTransformer(method="yeo-johnson")
                        df[col] = pt.fit_transform(df[[col]].astype(float)).ravel()
                except Exception:
                    pass

        elif kind == "bin_numeric":
            n = int(step.get("n_bins", 5))
            strategy = step.get("strategy", "equal_width")
            for col in cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                new_col = f"{col}_bin"
                try:
                    if strategy == "quantile":
                        df[new_col] = pd.qcut(df[col], q=n, labels=False, duplicates="drop")
                    else:
                        df[new_col] = pd.cut(df[col], bins=n, labels=False)
                    df[new_col] = df[new_col].astype("Int64")
                except Exception:
                    pass

        elif kind == "drop_duplicates":
            subset = cols if cols else None
            keep_val = step.get("keep", "first")
            df = df.drop_duplicates(subset=subset, keep=False if keep_val == "none" else keep_val)

        elif kind == "drop_nulls":
            subset = cols if cols else None
            how = step.get("how", "any")
            df = df.dropna(subset=subset, how=how)

        elif kind == "frequency_encode":
            normalize = step.get("normalize", True)
            for col in cols:
                freq = df[col].value_counts(normalize=normalize)
                df[f"{col}_freq"] = df[col].map(freq)

        elif kind == "cast_dtype":
            dtype = step.get("dtype", "float")
            for col in cols:
                try:
                    if dtype == "float":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    elif dtype == "int":
                        df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
                    elif dtype == "str":
                        df[col] = df[col].astype(str)
                except Exception:
                    pass

        elif kind == "polynomial_features":
            degree = max(2, min(int(step.get("degree", 2)), 4))
            interaction_only = bool(step.get("interaction_only", False))
            include_bias = bool(step.get("include_bias", False))
            num_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) < 1:
                continue
            try:
                X = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
                poly = PolynomialFeatures(
                    degree=degree,
                    interaction_only=interaction_only,
                    include_bias=include_bias,
                )
                out = poly.fit_transform(X)
                names = poly.get_feature_names_out(num_cols)
                for j, name in enumerate(names):
                    safe = str(name).replace(" ", "_").replace("^", "pow")
                    df[f"poly_{safe}"] = out[:, j]
            except Exception:
                pass

        elif kind == "extract_datetime":
            parts = step.get("parts") or ["year", "month", "day"]
            part_map = {
                "year": lambda s: s.dt.year,
                "month": lambda s: s.dt.month,
                "day": lambda s: s.dt.day,
                "hour": lambda s: s.dt.hour,
                "minute": lambda s: s.dt.minute,
                "dow": lambda s: s.dt.dayofweek,
                "dayofweek": lambda s: s.dt.dayofweek,
                "doy": lambda s: s.dt.dayofyear,
                "dayofyear": lambda s: s.dt.dayofyear,
                "week": lambda s: s.dt.isocalendar().week.astype("int64"),
            }
            for col in cols:
                if col not in df.columns:
                    continue
                s = pd.to_datetime(df[col], errors="coerce")
                for p in parts:
                    key = str(p).lower()
                    if key not in part_map:
                        continue
                    try:
                        df[f"{col}_{key}"] = part_map[key](s)
                    except Exception:
                        pass

        elif kind == "pca_projection":
            n_comp = max(1, int(step.get("n_components", 2)))
            prefix = (step.get("prefix") or "PC_").strip() or "PC_"
            drop_original = bool(step.get("drop_original", False))
            num_cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
            if len(num_cols) < 2:
                continue
            try:
                X = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
                k = min(n_comp, X.shape[1], max(1, X.shape[0] - 1))
                pca = PCA(n_components=k, random_state=42)
                Z = pca.fit_transform(X)
                for j in range(Z.shape[1]):
                    df[f"{prefix}{j + 1}"] = Z[:, j]
                if drop_original:
                    df = df.drop(columns=[c for c in num_cols if c in df.columns], errors="ignore")
            except Exception:
                pass

        elif kind == "tfidf_column":
            col = step.get("column") or (cols[0] if cols else None)
            max_feat = max(5, min(int(step.get("max_features", 50)), 200))
            ngram_max = max(1, min(int(step.get("ngram_max", 1)), 3))
            if not col or col not in df.columns:
                continue
            try:
                text = df[col].fillna("").astype(str)
                vec = TfidfVectorizer(
                    max_features=max_feat,
                    ngram_range=(1, ngram_max),
                    token_pattern=r"(?u)\b\w\w+\b",
                )
                mat = vec.fit_transform(text)
                dens = mat.toarray()
                for j in range(dens.shape[1]):
                    df[f"tfidf_{col}_{j}"] = dens[:, j]
            except Exception:
                pass

        elif kind == "derive_numeric":
            op = str(step.get("op", "add")).lower()
            a = step.get("column_a")
            b = step.get("column_b")
            out = step.get("output_column") or "derived_feature"
            if not a or not b or a not in df.columns or b not in df.columns:
                continue
            if out in df.columns and out not in (a, b):
                pass
            va = pd.to_numeric(df[a], errors="coerce")
            vb = pd.to_numeric(df[b], errors="coerce")
            try:
                if op == "add":
                    df[out] = va + vb
                elif op in ("subtract", "sub"):
                    df[out] = va - vb
                elif op in ("multiply", "mul"):
                    df[out] = va * vb
                elif op in ("divide", "div"):
                    df[out] = va / vb.replace(0, np.nan)
            except Exception:
                pass

        elif kind == "target_encode_dataset":
            target = step.get("target_column")
            if not target or target not in df.columns:
                continue
            gy = pd.to_numeric(df[target], errors="coerce")
            global_mean = float(gy.mean()) if gy.notna().any() else 0.0
            tmp = df.assign(_te_y_=gy)
            for col in cols:
                if col not in df.columns or col == target:
                    continue
                try:
                    enc = tmp.groupby(col, dropna=False)["_te_y_"].transform("mean")
                    df[f"{col}_tgtenc"] = enc.fillna(global_mean)
                except Exception:
                    pass

        elif kind == "train_test_split":
            target = step.get("target_column")
            if not target or target not in df.columns:
                raise ValueError(
                    "train_test_split requires target_column to exist in the dataframe."
                )
            test_size = float(step.get("test_size", 0.2))
            test_size = min(0.95, max(0.05, test_size))
            random_state = step.get("random_state", 42)
            try:
                random_state = int(random_state)
            except (TypeError, ValueError):
                random_state = 42
            shuffle = bool(step.get("shuffle", True))
            stratify_flag = bool(step.get("stratify", True))

            if ML_PIPELINE_SPLIT_COL in df.columns:
                df = df.drop(columns=[ML_PIPELINE_SPLIT_COL])

            usable = df.dropna(subset=[target])
            if len(usable) < 2:
                raise ValueError(
                    "Need at least 2 rows with a non-null target to assign train/test split."
                )

            idx = usable.index.to_numpy()
            y_sub = usable[target]
            strat = None
            if stratify_flag and _stratify_ok_for_split(y_sub):
                strat = y_sub

            try:
                idx_train, idx_test = sk_train_test_split(
                    idx,
                    test_size=test_size,
                    random_state=random_state if shuffle else None,
                    shuffle=shuffle,
                    stratify=strat,
                )
            except ValueError as e:
                raise ValueError(
                    "Train/test split failed (try disabling stratify or check class counts). "
                    f"Details: {str(e)}"
                ) from e

            df = df.copy()
            lab = pd.Series(index=df.index, dtype=object)
            lab.loc[idx_train] = "train"
            lab.loc[idx_test] = "test"
            df[ML_PIPELINE_SPLIT_COL] = lab

    return df


def preview_transforms(df: pd.DataFrame, steps: list[dict], n: int = 5) -> dict:
    """Return before/after previews (first n rows) as JSON-serialisable dicts."""
    before = df.head(n)
    after = apply_transforms(df, steps).head(n)
    return {
        "before": {
            "columns": before.columns.tolist(),
            "rows": before.values.tolist(),
        },
        "after": {
            "columns": after.columns.tolist(),
            "rows": [[None if (isinstance(v, float) and np.isnan(v)) else v for v in row]
                     for row in after.values.tolist()],
        },
    }
