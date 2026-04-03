import numpy as np
import pandas as pd


def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Select numeric columns and cast booleans to int so corr() works."""
    num = df.select_dtypes(include=[np.number])
    bool_cols = [c for c in num.columns if num[c].dtype.kind == 'b']
    if bool_cols:
        num = num.copy()
        num[bool_cols] = num[bool_cols].astype(int)
    return num


# ── Original function (unchanged) ────────────────────────────────────────────

def compute_eda(df: pd.DataFrame) -> dict:
    """Per-column stats + correlations (used by GET /eda/{id})."""
    stats = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        col_data = df[col].dropna().astype(float)
        if len(col_data) == 0:
            continue
        histogram, bin_edges = np.histogram(col_data, bins=20)
        stats[col] = {
            "mean":        round(float(col_data.mean()), 4),
            "std":         round(float(col_data.std()), 4),
            "min":         round(float(col_data.min()), 4),
            "max":         round(float(col_data.max()), 4),
            "median":      round(float(col_data.median()), 4),
            "q25":         round(float(col_data.quantile(0.25)), 4),
            "q75":         round(float(col_data.quantile(0.75)), 4),
            "hist_values": [round(float(v), 2) for v in histogram.tolist()],
            "hist_bins":   [round(float(b), 4) for b in bin_edges.tolist()],
        }

    numeric_df = _numeric_df(df)
    correlations: dict = {}
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr()
        for col in numeric_df.columns:
            correlations[col] = {
                k: (round(float(v), 4) if not np.isnan(v) else None)
                for k, v in corr_matrix[col].items()
            }

    return {
        "stats":        stats,
        "correlations": correlations,
        "shape":        list(numeric_df.shape),
        "columns":      numeric_df.columns.tolist(),
    }


# ── New endpoints ─────────────────────────────────────────────────────────────

def compute_correlation_matrix(df: pd.DataFrame) -> dict:
    """Full N×N Pearson correlation matrix for heatmap rendering."""
    numeric_df = _numeric_df(df)
    if len(numeric_df.columns) < 2:
        return {"columns": numeric_df.columns.tolist(), "matrix": []}
    corr = numeric_df.corr()
    cols = corr.columns.tolist()
    matrix = [
        [(round(float(corr.loc[r, c]), 4) if not np.isnan(corr.loc[r, c]) else None) for c in cols]
        for r in cols
    ]
    return {"columns": cols, "matrix": matrix}


def compute_missing_values(df: pd.DataFrame) -> dict:
    """Count and percentage of missing values per column."""
    total = len(df)
    result: dict = {}
    for col in df.columns:
        count = int(df[col].isnull().sum())
        result[col] = {
            "count": count,
            "pct":   round(count / total * 100, 2) if total > 0 else 0.0,
        }
    return result


def compute_outliers(df: pd.DataFrame) -> dict:
    """IQR-based outlier bounds and count per numeric column."""
    result: dict = {}
    numeric_df = _numeric_df(df)
    for col in numeric_df.columns:
        q1  = float(numeric_df[col].quantile(0.25))
        q3  = float(numeric_df[col].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = int(((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum())
        result[col] = {
            "lower":     round(lower, 4),
            "upper":     round(upper, 4),
            "n_outliers": n_out,
        }
    return result


def compute_scatter(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    target_col: str | None = None,
) -> dict:
    """Return up to 500 sampled points for a scatter plot."""
    sample = df[[c for c in [col_x, col_y, target_col] if c and c in df.columns]].dropna()
    sample = sample.sample(min(500, len(sample)), random_state=42)
    result: dict = {
        "x": [round(float(v), 4) for v in sample[col_x].tolist()],
        "y": [round(float(v), 4) for v in sample[col_y].tolist()],
    }
    if target_col and target_col in sample.columns:
        result["color"] = sample[target_col].tolist()
    return result


def compute_categorical_stats(df: pd.DataFrame) -> dict:
    """Value counts for non-numeric columns (top 10 values each)."""
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    result: dict = {}
    for col in cat_cols:
        vc = df[col].value_counts().head(10)
        result[col] = {
            "n_unique":   int(df[col].nunique()),
            "top_values": [[str(k), int(v)] for k, v in vc.items()],
        }
    return result


# ── New analytics endpoints ───────────────────────────────────────────────────

def compute_health(df: pd.DataFrame) -> dict:
    """Dataset-level health overview: duplicates, constants, cardinality, memory."""
    n_rows, n_cols = df.shape
    numeric_cols  = _numeric_df(df).columns.tolist()
    cat_cols      = df.select_dtypes(exclude=[np.number]).columns.tolist()

    n_duplicates   = int(df.duplicated().sum())
    constant_cols  = [c for c in numeric_cols if df[c].nunique() <= 1]
    near_zero_cols = [c for c in numeric_cols
                      if 1 < df[c].nunique() <= max(2, int(n_rows * 0.01))]

    high_card_cols = [c for c in cat_cols
                      if n_rows > 0 and df[c].nunique() / n_rows > 0.5]

    memory_mb = round(float(df.memory_usage(deep=True).sum()) / 1024 / 1024, 3)

    completeness = {}
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        completeness[col] = round((1 - n_missing / n_rows) * 100, 1) if n_rows > 0 else 100.0

    # Variance per numeric column
    variance = {}
    for col in numeric_cols:
        s = df[col].dropna()
        variance[col] = round(float(s.var()), 6) if len(s) > 1 else 0.0

    return {
        "n_rows":           n_rows,
        "n_cols":           n_cols,
        "n_duplicates":     n_duplicates,
        "memory_mb":        memory_mb,
        "constant_cols":    constant_cols,
        "near_zero_cols":   near_zero_cols,
        "high_card_cols":   high_card_cols,
        "completeness":     completeness,
        "variance":         variance,
        "dtype_counts": {
            "numeric":     len(numeric_cols),
            "categorical": len(cat_cols),
        },
        "numeric_cols": numeric_cols,
        "cat_cols":     cat_cols,
    }


def compute_target_analysis(df: pd.DataFrame, target_col: str) -> dict:
    """Distribution and class-balance analysis for the target column."""
    series    = df[target_col].dropna()
    is_num    = pd.api.types.is_numeric_dtype(series)
    result: dict = {"target_col": target_col, "is_numeric": is_num, "n": len(series)}

    if is_num:
        histogram, bin_edges = np.histogram(series, bins=20)
        skewness = float(series.skew())
        kurtosis = float(series.kurtosis())

        # Suggest transform based on skewness
        if abs(skewness) < 0.5:
            transform_hint = None
        elif skewness > 2:
            transform_hint = "log1p (strong right skew)"
        elif skewness > 0.5:
            transform_hint = "sqrt (moderate right skew)"
        elif skewness < -2:
            transform_hint = "square / reciprocal (strong left skew)"
        else:
            transform_hint = "sqrt or reflect + log1p (moderate left skew)"

        result.update({
            "hist_values":    [int(v) for v in histogram.tolist()],
            "hist_bins":      [round(float(b), 4) for b in bin_edges.tolist()],
            "skewness":       round(skewness, 4),
            "kurtosis":       round(kurtosis, 4),
            "mean":           round(float(series.mean()), 4),
            "std":            round(float(series.std()), 4),
            "min":            round(float(series.min()), 4),
            "max":            round(float(series.max()), 4),
            "median":         round(float(series.median()), 4),
            "transform_hint": transform_hint,
        })
    else:
        vc    = series.value_counts()
        total = len(series)
        class_counts = [[str(k), int(v), round(v / total * 100, 2)] for k, v in vc.items()]
        imbalance    = round(float(vc.iloc[0] / vc.iloc[-1]), 2) if len(vc) > 1 else 1.0
        result.update({
            "n_classes":       int(len(vc)),
            "class_counts":    class_counts,
            "imbalance_ratio": imbalance,
            "is_imbalanced":   imbalance > 3.0,
        })

    return result


def compute_skewness(df: pd.DataFrame) -> list:
    """Skewness + kurtosis for every numeric column, sorted by |skewness|."""
    numeric_df = _numeric_df(df)
    result = []
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        if len(s) < 3:
            continue
        skew = float(s.skew())
        kurt = float(s.kurtosis())
        abs_skew = abs(skew)

        if abs_skew < 0.5:
            severity = "normal"
            hint = None
        elif abs_skew < 1.0:
            severity = "moderate"
            hint = "sqrt" if skew > 0 else "square"
        else:
            severity = "high"
            hint = "log1p" if skew > 0 else "reciprocal"

        result.append({
            "column":      col,
            "skewness":    round(skew, 4),
            "kurtosis":    round(kurt, 4),
            "abs_skewness":round(abs_skew, 4),
            "severity":    severity,
            "hint":        hint,
        })

    result.sort(key=lambda x: x["abs_skewness"], reverse=True)
    return result


def compute_boxplots(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Box-and-whisker stats (IQR, fences, sampled outliers) per column."""
    numeric_df = _numeric_df(df)
    cols = [c for c in (columns or numeric_df.columns.tolist()) if c in numeric_df.columns][:20]

    result: dict = {}
    for col in cols:
        s = numeric_df[col].dropna()
        if len(s) == 0:
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        outliers = s[(s < lo) | (s > hi)]
        result[col] = {
            "min":         round(float(s.min()), 4),
            "q1":          round(q1, 4),
            "median":      round(float(s.median()), 4),
            "mean":        round(float(s.mean()), 4),
            "q3":          round(q3, 4),
            "max":         round(float(s.max()), 4),
            "lower_fence": round(lo, 4),
            "upper_fence": round(hi, 4),
            "outliers":    [round(float(v), 4) for v in
                            outliers.sample(min(60, len(outliers)), random_state=42).tolist()],
        }
    return result


def compute_pairplot(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Scatter + histogram data for an N×N pairplot grid (max 6 columns)."""
    numeric_df = _numeric_df(df)
    cols = [c for c in (columns or numeric_df.columns.tolist()) if c in numeric_df.columns][:6]
    sample_df  = numeric_df[cols].dropna().sample(min(300, len(numeric_df)), random_state=42)

    pairs: dict = {}
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            key = f"{c1}___{c2}"
            if i == j:
                hist, bins = np.histogram(sample_df[c1].dropna(), bins=15)
                pairs[key] = {
                    "type":   "hist",
                    "values": [int(v) for v in hist.tolist()],
                    "bins":   [round(float(b), 4) for b in bins.tolist()],
                }
            else:
                pairs[key] = {
                    "type": "scatter",
                    "x":    [round(float(v), 4) for v in sample_df[c1].tolist()],
                    "y":    [round(float(v), 4) for v in sample_df[c2].tolist()],
                }

    return {"columns": cols, "pairs": pairs}


def compute_feature_target(df: pd.DataFrame, target_col: str) -> dict:
    """Per-feature relationship to the target: scatter+corr for regression, boxgroup for classification."""
    target         = df[target_col]
    is_target_num  = pd.api.types.is_numeric_dtype(target)
    numeric_df     = _numeric_df(df)
    feature_cols   = [c for c in numeric_df.columns if c != target_col]

    features: dict = {}
    for col in feature_cols:
        sub = df[[col, target_col]].dropna()
        if len(sub) == 0:
            continue

        if is_target_num:
            sample  = sub.sample(min(400, len(sub)), random_state=42)
            corr    = round(float(sub[col].corr(sub[target_col])), 4)
            # simple linear regression for trend line
            x_vals  = sub[col].values.astype(float)
            y_vals  = sub[target_col].values.astype(float)
            if x_vals.std() > 0:
                coeffs = np.polyfit(x_vals, y_vals, 1)
                slope, intercept = float(coeffs[0]), float(coeffs[1])
                x_min, x_max = float(x_vals.min()), float(x_vals.max())
                trend = {
                    "x":     [round(x_min, 4), round(x_max, 4)],
                    "y":     [round(slope * x_min + intercept, 4), round(slope * x_max + intercept, 4)],
                    "slope": round(slope, 6),
                }
            else:
                trend = None

            features[col] = {
                "type":        "scatter",
                "x":           [round(float(v), 4) for v in sample[col].tolist()],
                "y":           [round(float(v), 4) for v in sample[target_col].tolist()],
                "correlation": corr,
                "trend":       trend,
            }
        else:
            classes   = [str(c) for c in target.unique().tolist()[:10]]
            boxdata: dict = {}
            for cls in classes:
                mask = sub[target_col].astype(str) == cls
                s    = sub.loc[mask, col]
                if len(s) == 0:
                    continue
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                boxdata[cls] = {
                    "q1":    round(q1, 4),
                    "median":round(float(s.median()), 4),
                    "mean":  round(float(s.mean()), 4),
                    "q3":    round(q3, 4),
                    "min":   round(float(s.min()), 4),
                    "max":   round(float(s.max()), 4),
                }
            features[col] = {"type": "boxgroup", "classes": boxdata}

    return {
        "target_col":       target_col,
        "is_target_numeric":is_target_num,
        "features":         features,
    }
