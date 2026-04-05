import { Check, Copy, Info, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";

type Page =
  | "upload"
  | "split"
  | "eda"
  | "transforms"
  | "train"
  | "experiments"
  | "importances"
  | "diagnostics"
  | "export"
  | "predict";

interface HParamDoc {
  name: string;
  type: string;
  default: string;
  description: string;
}

interface Section {
  heading: string;
  /** Short label for the tab (defaults to full heading). */
  tab?: string;
  body: string;
  code: string;
  /** Code window subtitle, e.g. "Python · pandas". */
  lang?: string;
  params?: HParamDoc[];
}

const CONTENT: Record<Page, { title: string; summary: string; sections: Section[] }> = {
  upload: {
    title: "Upload & Train",
    summary:
      "Load a CSV, inspect it, choose a target column, and kick off training. The app auto-detects whether it's a regression or classification problem based on the target's cardinality.",
    sections: [
      {
        heading: "Loading & splitting data",
        tab: "Load & split",
        body: "pandas reads the CSV. Save a hold-out on the Split tab before Transforms; the Train tab then uses __ml_split__ and skips its own random split when that column is present. Otherwise Train falls back to sklearn train_test_split (default 20% test).",
        code: `import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("wine.csv")
X = df.drop(columns=["quality"])
y = df["quality"]

# Regression
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Classification — stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y  # preserves class ratios
)`,
      },
      {
        heading: "Auto-detecting task type",
        tab: "Task type",
        body: "If the target has fewer than 20 unique values the app treats it as classification, otherwise regression.",
        code: `n_unique = y.nunique()
task = "classification" if n_unique < 20 else "regression"
print(task)  # "regression" for quality 0-10`,
      },
    ],
  },

  split: {
    title: "Hold-out split",
    summary:
      "Assign every row to train or test on a copy of your original upload. This must be saved before Transforms or EDA quick-apply steps run, so the evaluation set is fixed before preprocessing.",
    sections: [
      {
        heading: "Why a separate tab",
        tab: "Overview",
        body: "The Split tab writes __ml_split__ from sklearn’s train_test_split on row indices (with optional stratification). Saving clears any prior transform pipeline and replaces the working CSV. Re-saving after Reset to original reuses your stored options.",
        code: `from sklearn.model_selection import train_test_split

idx_train, idx_test = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=y
)
df.loc[idx_train, "__ml_split__"] = "train"
df.loc[idx_test, "__ml_split__"] = "test"`,
      },
    ],
  },

  eda: {
    title: "Exploratory Data Analysis",
    summary:
      "Understand the data before modelling. EDA surfaces distributions, correlations, missing values, outliers, and categorical frequencies. The Recommendations tab turns those signals into suggested pipeline steps you can apply one-by-one or all at once.",
    sections: [
      {
        heading: "Descriptive statistics",
        body: "pandas describe() gives count, mean, std, min, quartiles, and max for every numeric column in one call.",
        code: `import pandas as pd

df = pd.read_csv("wine.csv")

# Per-column summary
print(df.describe().T)

# Single column
col = df["alcohol"]
print(f"mean={col.mean():.3f}  std={col.std():.3f}")
print(f"median={col.median():.3f}")
print(f"Q1={col.quantile(0.25):.3f}  Q3={col.quantile(0.75):.3f}")`,
      },
      {
        heading: "Correlation matrix",
        tab: "Correlations",
        lang: "Python · pandas · seaborn",
        body: "Pearson r measures linear relationships between numeric columns. Values close to ±1 signal redundant features or strong predictors.",
        code: `import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr(numeric_only=True)

# Heatmap
sns.heatmap(corr, annot=True, fmt=".2f",
            cmap="coolwarm", center=0,
            vmin=-1, vmax=1)
plt.tight_layout()
plt.show()

# Top correlations with target
print(corr["quality"].sort_values())`,
      },
      {
        heading: "Missing values",
        body: "Find and quantify NaNs before they silently break your pipeline.",
        code: `# Count and percentage
missing = df.isnull().sum()
pct     = missing / len(df) * 100

summary = pd.DataFrame({"count": missing, "pct": pct})
print(summary[summary["count"] > 0])`,
      },
      {
        heading: "Transform recommendations",
        tab: "Recommendations",
        body: "Save hold-out on the Split tab first so __ml_split__ exists. Recommendations then distinguish table-level cleaning (e.g. drop constants) from steps you typically run after the split is fixed (impute, clip, skew fix, frequency encode). Quick-apply requires the split column. Pandas transforms still see all rows unless a step drops rows.",
        code: `# Before split: table-level cleaning
df = df.drop_duplicates()
# After split: in sklearn, fit on X_train only
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median").fit(X_train)
X_test = imp.transform(X_test)`,
      },
      {
        heading: "Outlier detection (IQR)",
        body: "Values outside 1.5 × IQR from Q1/Q3 are flagged as outliers. They can skew linear models significantly.",
        code: `import numpy as np

Q1  = df.quantile(0.25)
Q3  = df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df < Q1 - 1.5 * IQR) |
            (df > Q3 + 1.5 * IQR)).sum()
print(outliers[outliers > 0])`,
      },
    ],
  },

  transforms: {
    title: "Preprocessing / Transforms",
    summary:
      "Raw data almost never feeds directly into a model. Transforms fix missing values, encode categoricals, scale numerics, and clip outliers — all using sklearn's Transformer API.",
    sections: [
      {
        heading: "Impute missing values",
        tab: "Impute",
        body: "SimpleImputer replaces NaN with a statistic (mean, median, most-frequent) or a constant. The Transforms tab often uses pandas fillna with column-wise statistics—same idea, different API.",
        code: `from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="mean")   # mean | median | most_frequent | constant
X_train_imp = imp.fit_transform(X_train)
X_test_imp  = imp.transform(X_test)    # use training stats on test!`,
      },
      {
        heading: "One-hot encoding",
        body: "Turns a categorical column with k categories into k-1 binary columns (drop_first avoids multicollinearity).",
        code: `from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(drop="first", sparse_output=False)
X_enc = enc.fit_transform(X_train[["region"]])
print(enc.get_feature_names_out())`,
      },
      {
        heading: "Label encoding",
        body: "Maps each category to an integer. Useful for ordinal variables or tree-based models.",
        code: `from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_enc = le.fit_transform(y_train)
print(le.classes_)   # ['low', 'medium', 'high']`,
      },
      {
        heading: "Scaling",
        body: "Linear models and SVMs require scaled features. Tree models don't — but it never hurts.",
        code: `from sklearn.preprocessing import (
    StandardScaler,   # (x - mean) / std  → N(0,1)
    MinMaxScaler,     # (x - min) / (max - min) → [0,1]
    RobustScaler,     # uses median + IQR, robust to outliers
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)`,
      },
      {
        heading: "Clip outliers",
        body: "Cap extreme values so they don't dominate loss functions.",
        code: `import numpy as np

# IQR method
Q1, Q3 = np.percentile(X_train, [25, 75], axis=0)
IQR    = Q3 - Q1
lower  = Q1 - 1.5 * IQR
upper  = Q3 + 1.5 * IQR
X_clipped = np.clip(X_train, lower, upper)

# Z-score method (clip beyond ±3σ)
from scipy import stats
X_clipped = np.clip(X_train,
    X_train.mean() - 3 * X_train.std(),
    X_train.mean() + 3 * X_train.std())`,
      },
      {
        heading: "Row ops: drop duplicates & nulls",
        tab: "Rows",
        lang: "Python · pandas",
        body: "The Transforms backend uses pandas for these steps. Dropping rows changes counts and can bias results if missingness relates to the target—check previews.",
        code: `import pandas as pd

df = df.drop_duplicates(subset=["customer_id"], keep="first")
df = df.dropna(subset=["target", "feature_a"], how="any")`,
      },
      {
        heading: "One-hot with pandas (get_dummies)",
        tab: "get_dummies",
        lang: "Python · pandas",
        body: "The app applies pd.get_dummies with drop_first=True and integer dtype—close to OneHotEncoder(drop='first') for modeling.",
        code: `import pandas as pd

df = pd.get_dummies(
    df,
    columns=["region", "channel"],
    drop_first=True,
    dtype=int,
)`,
      },
      {
        heading: "Skewness: Yeo-Johnson / Box-Cox",
        tab: "Skewness",
        lang: "Python · sklearn",
        body: "PowerTransformer reduces skew. Box-Cox needs strictly positive values; Yeo-Johnson handles zeros and negatives. The Fix Skewness step automates similar choices per column.",
        code: `from sklearn.preprocessing import PowerTransformer
import pandas as pd

pt = PowerTransformer(method="yeo-johnson")
df["income_t"] = pt.fit_transform(df[["income"]])`,
      },
      {
        heading: "Polynomial features & PCA",
        tab: "Poly & PCA",
        lang: "Python · sklearn",
        body: "PolynomialFeatures expands numeric inputs; PCA projects onto orthogonal components. Both are fit on the data seen at that step—order relative to train/test matters.",
        code: `from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(df[["x1", "x2", "x3"]].fillna(0))

pca = PCA(n_components=4, random_state=42)
scores = pca.fit_transform(df[["a", "b", "c", "d"]].fillna(0))`,
      },
      {
        heading: "TF‑IDF on one text column",
        tab: "TF‑IDF",
        lang: "Python · sklearn",
        body: "TfidfVectorizer turns text into sparse token weights. The app caps features and expands into numeric columns for one chosen column.",
        code: `from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(max_features=200, ngram_range=(1, 2), token_pattern=r"(?u)\\b\\w\\w+\\b")
X_sp = vec.fit_transform(df["description"].fillna("").astype(str))`,
      },
      {
        heading: "Train/test split column (__ml_split__)",
        tab: "Split column",
        lang: "Python · sklearn · pandas",
        body: "Use the Split tab to add __ml_split__ before Transforms. Older pipelines may still contain a legacy Train/Test Split transform step. The Train tab uses __ml_split__ when present and skips its own random hold-out.",
        code: `from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")
usable = df.dropna(subset=["target"])
idx = usable.index.to_numpy()
y_sub = usable["target"]
tr_idx, te_idx = train_test_split(
    idx, test_size=0.2, random_state=42, stratify=y_sub
)

lab = pd.Series(index=df.index, dtype=object)
lab.loc[tr_idx] = "train"
lab.loc[te_idx] = "test"
df["__ml_split__"] = lab`,
      },
      {
        heading: "Target encoding on the full table (leakage)",
        tab: "Target encode",
        lang: "Python · pandas",
        body: "Replacing categories with the mean target uses information from every row—including what you would not have at inference for a true hold-out. Fine for exploration; for modeling use CV-safe encoding in the Train pipeline.",
        code: `m = df["target"].mean()
df["cat_te"] = df.groupby("category")["target"].transform("mean")
df["cat_te"] = df["cat_te"].fillna(m)`,
      },
    ],
  },

  experiments: {
    title: "Model Experiments",
    summary:
      "Browse MLflow runs for this dataset: test metrics, optional train metrics (after retraining), and CV. Charts and tables hide columns when older runs do not have them. For the MLflow UI, use the copyable command on this page so the UI reads the same store as the API (SQLite by default)—plain mlflow ui without --backend-store-uri points at a different ./mlruns folder and will not show these runs.",
    sections: [
      {
        heading: "Regression models",
        body: "Five estimators cover the linear-to-ensemble spectrum. Each is evaluated with RMSE, MAE, R², and 5-fold CV RMSE.",
        code: `from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np

models = {
    "linear_regression":   LinearRegression(),
    "ridge":               Ridge(alpha=1.0),
    "random_forest":       RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting":   GradientBoostingRegressor(n_estimators=100, random_state=42),
    "svr":                 SVR(kernel="rbf", C=1.0),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_train, y_train, cv=5,
               scoring="neg_root_mean_squared_error")

    print(f"{name}: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  CV={-cv.mean():.4f}")`,
      },
      {
        heading: "Classification models",
        body: "Same estimators adapted for classification. Evaluated with accuracy, weighted F1, and ROC-AUC (one-vs-rest).",
        code: `from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 (w)   : {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob, multi_class='ovr'):.4f}")`,
      },
      {
        heading: "MLflow tracking",
        tab: "MLflow",
        body: "The API defaults to an absolute SQLite path next to main.py (see backend logs for the exact URI). Every run logs params, metrics (including train_* metrics for newer runs), and the model artifact. To open the MLflow UI for that same database, run mlflow ui with --backend-store-uri set to that URI—not plain mlflow ui, which uses ./mlruns in your shell’s current directory and is a separate, unrelated store.",
        code: `import mlflow
import mlflow.sklearn

# Use an ABSOLUTE path in the URI (three slashes after sqlite: + path)
# Example: sqlite:////Users/you/proj/ml-studio/backend/mlruns.db
mlflow.set_tracking_uri("sqlite:////ABSOLUTE/PATH/TO/backend/mlruns.db")
mlflow.set_experiment("dataset-<your_dataset_id>")

with mlflow.start_run(run_name="random_forest"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("train_rmse", rmse_train)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")

# UI (same machine): from backend/, recommended:
# uv run python mlflow_ui_cli.py
# Or copy the full mlflow ui --backend-store-uri ... command from Experiments.`,
      },
      {
        heading: "Hyperparameter reference",
        tab: "Model params",
        body: "Key knobs available per model. All are tunable from the Upload → Hyperparameters panel before training.",
        code: `# Ridge / RidgeClassifier
Ridge(alpha=1.0)          # higher α → more regularisation

# Logistic Regression
LogisticRegression(C=1.0, max_iter=1000)  # C = 1/λ

# Decision Tree
DecisionTreeRegressor(
    max_depth=5,           # None = unlimited (overfits)
    min_samples_split=2,   # min samples to split a node
)

# Random Forest
RandomForestRegressor(
    n_estimators=100,      # more trees → better, slower
    max_depth=None,
    min_samples_split=2,
)

# Gradient Boosting
GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,     # shrinkage — lower needs more trees
    max_depth=3,
    subsample=1.0,         # <1.0 = stochastic boosting
)`,
        params: [
          { name: "alpha (Ridge)", type: "float", default: "1.0", description: "L2 regularisation strength. Higher shrinks weights more, reducing overfitting." },
          { name: "C (LogReg)", type: "float", default: "1.0", description: "Inverse regularisation. Smaller C = stronger penalty, simpler model." },
          { name: "max_depth", type: "int", default: "5", description: "Max tree depth. 0/None = grow until pure. Start at 3–6 to avoid overfitting." },
          { name: "n_estimators", type: "int", default: "100", description: "Number of trees in the ensemble. Diminishing returns after ~300; costs training time." },
          { name: "learning_rate", type: "float", default: "0.1", description: "Shrinks each tree's contribution. Lower (0.01–0.05) + more trees often beats high rate." },
          { name: "subsample", type: "float", default: "1.0", description: "Fraction of rows sampled per tree. Values 0.5–0.8 add regularisation via randomness." },
          { name: "min_samples_split", type: "int", default: "2", description: "Minimum samples to attempt a split. Higher values prune small, noisy branches." },
        ],
      },
    ],
  },

  importances: {
    title: "Feature Importances",
    summary:
      "Compare how much each model relies on every feature. Select any subset of trained models (or use Select all); importances load in parallel and are shown as stacked bars per feature. Tree models use built-in split importances; linear models use |coefficient| × feature std as a proxy.",
    sections: [
      {
        heading: "Comparing runs",
        tab: "Compare",
        body: "Each checkbox is one MLflow run (same algorithm name can appear more than once after retraining). Labels show the last 8 characters of the run id and the start time when available. Select all includes every run; Single only keeps one. The API loads importances from the exact run you select. Features are ordered by the highest importance among selected runs.",
        code: `# Conceptually: one GET /importances/{dataset_id}/{model_name} per selected model,
# then align rows by feature name and plot normalized bar widths.

import pandas as pd

# Example: two model importance series aligned on columns
s_a = pd.Series({...}, name="gradient_boosting")
s_b = pd.Series({...}, name="random_forest")
df = pd.concat([s_a, s_b], axis=1).fillna(0)
df["max"] = df.max(axis=1)
df = df.sort_values("max", ascending=False)`,
      },
      {
        heading: "Tree-based importances",
        body: "feature_importances_ is the mean decrease in impurity (Gini / variance) across all trees, normalised to sum to 1.",
        code: `from sklearn.ensemble import RandomForestRegressor
import pandas as pd

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

imp = pd.Series(rf.feature_importances_, index=X.columns)
imp.sort_values(ascending=False).plot.barh()

# GradientBoosting exposes the same attribute
# gb.feature_importances_`,
      },
      {
        heading: "Linear model importances",
        body: "Raw coefficients are scale-dependent. Multiply by the feature's standard deviation to make them comparable.",
        code: `from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X_s = scaler.fit_transform(X_train)

ridge = Ridge(alpha=1.0)
ridge.fit(X_s, y_train)

# |coef| × std — same units, comparable magnitudes
imp = np.abs(ridge.coef_) * scaler.scale_
pd.Series(imp, index=X.columns).sort_values(ascending=False)`,
      },
      {
        heading: "Permutation importance (model-agnostic)",
        body: "Shuffle each feature and measure the drop in score. Works for any estimator and captures non-linear effects.",
        code: `from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10, random_state=42,
    scoring="r2",
)

imp = pd.Series(result.importances_mean, index=X.columns)
print(imp.sort_values(ascending=False))`,
      },
    ],
  },

  diagnostics: {
    title: "Model diagnostics",
    summary:
      "Hold-out evaluation: confusion matrix and report for classification; ROC and calibration for binary; residuals, learning curve, and scaler baseline for regression; permutation importance for any task. Learning curve titles follow the metric (accuracy vs RMSE), with short guidance on underfitting, overfitting, and convergence.",
    sections: [
      {
        heading: "What you see",
        body: "Metrics are computed on the same test split used after training (from __ml_split__ or your configured hold-out). Heavy steps subsample rows to keep the UI responsive.",
        code: `# Backend: sklearn.metrics + inspection
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# ROC / calibration need predict_proba (classifiers)
# Learning curve retrains on growing subsets (regression)`,
      },
      {
        heading: "Learning curves",
        tab: "Learning curve",
        body: "The chart plots mean train score vs mean CV validation score as training set size grows. For accuracy, higher is better; for RMSE, lower is better on both lines. If both curves stay poor, the model may be underfitting. If train and validation stay far apart with little convergence, variance or overfitting is more likely. When the curves meet and flatten, collecting more data may help less.",
        code: `from sklearn.model_selection import learning_curve
import numpy as np

# Regression example (negated RMSE for sklearn API)
train_sizes, train_scores, val_scores = learning_curve(
    estimator, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=2,
    scoring="neg_root_mean_squared_error",
)
# Plot train_scores.mean(axis=1) vs val_scores.mean(axis=1)`,
      },
    ],
  },

  train: {
    title: "Train",
    summary:
      "Select a target column, validation strategy, and models, then train. Results show train (in-sample), hold-out test, and CV metrics side by side, plus short fit hints (e.g. train vs test gap, CV spread). Learning curves live on Diagnostics. Every run is logged to MLflow (same store as Experiments).",
    sections: [
      {
        heading: "Results: train, CV, and test",
        tab: "Metrics table",
        body: "Train columns are computed on the full training matrix after pipeline fit. Test columns are the held-out split (__ml_split__ or your configured test fraction). CV is cross-validation on training data only (strategy from Split & Validation). CV σ is the standard deviation of fold scores when available. Fit hints are conservative heuristics, not ground truth. For learning curves, open Diagnostics after training and pick the same model.",
        code: `# Same idea as sklearn: fit on X_train, then
y_pred_train = pipe.predict(X_train)
y_pred_test  = pipe.predict(X_test)

# e.g. regression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# CV: cross_val_score on (X_train, y_train) only`,
      },
      {
        heading: "Auto-detecting task type",
        body: "If the target has fewer than 20 unique integer values the backend treats it as classification, otherwise regression. You can always override.",
        code: `import pandas as pd

y = df["quality"]

# Heuristic used in this app
n_unique = y.nunique()
is_int   = (y.dtype != float) or (y % 1 == 0).all()
task     = "classification" if (n_unique < 20 and is_int) else "regression"
print(task)`,
      },
      {
        heading: "Training pipeline",
        body: "Each model is wrapped in a sklearn Pipeline with StandardScaler → estimator. This ensures leakage-free scaling (scaler is fit only on training rows).",
        code: `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestRegressor(n_estimators=100, random_state=42)),
])

pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)   # scaler applied automatically`,
      },
      {
        heading: "Hold-out from Transforms (__ml_split__)",
        tab: "Pipeline split",
        lang: "Python · pandas",
        body: "If the saved dataset still includes __ml_split__ after your Transforms pipeline, the server builds X_train/X_test from those labels. The Train tab hides duplicate test-size controls; CV and scaling still use only training rows inside the sklearn pipeline.",
        code: `import pandas as pd

m_tr = df["__ml_split__"] == "train"
m_te = df["__ml_split__"] == "test"
drop = ["target", "__ml_split__"]
X_train = df.loc[m_tr].drop(columns=drop, errors="ignore")
y_train = df.loc[m_tr, "target"]
X_test  = df.loc[m_te].drop(columns=drop, errors="ignore")
y_test  = df.loc[m_te, "target"]`,
      },
      {
        heading: "Retraining with new hyperparameters",
        body: "Each click of Train is a fresh MLflow run. Change n_estimators or learning_rate and train again — all runs accumulate in Experiments for comparison.",
        code: `import mlflow

# Run 1 — defaults
with mlflow.start_run(run_name="rf_default"):
    mlflow.log_param("n_estimators", 100)
    rf1 = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    mlflow.log_metric("rmse", rmse(rf1, X_test, y_test))

# Run 2 — tuned
with mlflow.start_run(run_name="rf_tuned"):
    mlflow.log_param("n_estimators", 300)
    rf2 = RandomForestRegressor(n_estimators=300).fit(X_train, y_train)
    mlflow.log_metric("rmse", rmse(rf2, X_test, y_test))`,
      },
      {
        heading: "Hyperparameter reference",
        tab: "Sliders",
        body: "Key parameters available per model in the sliders above.",
        code: `# Ridge / RidgeClassifier
Ridge(alpha=1.0)          # higher α → more regularisation

# Logistic Regression
LogisticRegression(C=1.0, max_iter=1000)

# Decision Tree
DecisionTreeRegressor(max_depth=5, min_samples_split=2)

# Random Forest
RandomForestRegressor(n_estimators=100, max_depth=None)

# Gradient Boosting
GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1,
    max_depth=3,      subsample=1.0,
)`,
        params: [
          { name: "alpha", type: "float", default: "1.0", description: "L2 regularisation strength (Ridge). Higher shrinks weights more." },
          { name: "C", type: "float", default: "1.0", description: "Inverse regularisation for Logistic Regression. Smaller = stronger penalty." },
          { name: "max_depth", type: "int", default: "5", description: "Max tree depth. 0 = unlimited. Keep low (3–6) to prevent overfitting." },
          { name: "n_estimators", type: "int", default: "100", description: "Number of trees. Diminishing returns after ~300; costs training time." },
          { name: "learning_rate", type: "float", default: "0.1", description: "Gradient boosting step size. Lower (0.01–0.05) + more trees often wins." },
          { name: "subsample", type: "float", default: "1.0", description: "Row fraction per tree in GB. Values 0.5–0.8 add regularisation." },
          { name: "min_samples_split", type: "int", default: "2", description: "Min samples to split a node. Higher prunes small noisy branches." },
        ],
      },
    ],
  },

  export: {
    title: "Pipeline export",
    summary:
      "Download the fitted sklearn Pipeline as a gzip-compressed joblib file (same serialization the backend uses for MLflow). Load it in Python with joblib.load and call predict on a DataFrame whose columns match the training feature space (no target, no __ml_split__).",
    sections: [
      {
        heading: "Download",
        tab: "Download",
        body: "Pick a model that has been trained on the active dataset. The browser saves a .joblib file. You need the same library versions (sklearn, xgboost if used, etc.) as the training environment for unpickling to work reliably.",
        code: `# Equivalent endpoint (auth header if API_KEY is set):
# GET /export/{dataset_id}/{model_name}/pipeline.joblib`,
      },
      {
        heading: "Load and predict",
        tab: "joblib.load",
        body: "joblib transparently decompresses gzip. Build X with the same column names the pipeline expects—after training, those names are stored on the dataset record. The Export page generates a starter snippet using EDA means for numeric fields.",
        code: `import joblib
import pandas as pd

pipe = joblib.load("random_forest_pipeline.joblib")
X = pd.DataFrame([{"feature_a": 1.2, "feature_b": 3.4}])
print(pipe.predict(X))`,
      },
    ],
  },

  predict: {
    title: "Predict",
    summary:
      "Pass custom feature values to any trained model and get an instant prediction. Compare all models side-by-side to see how much their outputs diverge.",
    sections: [
      {
        heading: "Single prediction",
        body: "Reconstruct a one-row DataFrame with the same column order as training, then call predict().",
        code: `import pandas as pd

# Mirror training feature order exactly
sample = pd.DataFrame([{
    "fixed acidity":    7.4,
    "volatile acidity": 0.70,
    "citric acid":      0.00,
    "residual sugar":   1.9,
    "chlorides":        0.076,
    "alcohol":          9.4,
}])

prediction = model.predict(sample)
print(f"Predicted quality: {prediction[0]:.3f}")`,
      },
      {
        heading: "Classification with confidence",
        body: "predict_proba() returns per-class probabilities. The highest probability is the confidence score shown in the UI.",
        code: `from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

proba          = clf.predict_proba(sample)      # shape (1, n_classes)
predicted_class = clf.predict(sample)[0]
confidence      = proba.max()

print(f"Class      : {predicted_class}")
print(f"Confidence : {confidence:.1%}")
print(f"All probs  : {dict(zip(clf.classes_, proba[0].round(3)))}")`,
      },
      {
        heading: "Comparing models",
        body: "Run the same input through all trained models to see where they agree and where they diverge — large spread signals high uncertainty.",
        code: `# Assume lr, ridge, rf, gb are already fitted sklearn estimators
models = {
    "linear_regression": lr,
    "ridge":             ridge,
    "random_forest":     rf,
    "gradient_boosting": gb,
}

preds = {name: m.predict(sample)[0]
         for name, m in models.items()}

spread = max(preds.values()) - min(preds.values())
print(preds)
print(f"Spread: {spread:.3f}")`,
      },
    ],
  },
};

// ── Code & section panels ─────────────────────────────────────────────────────

function CodeBlock({ code, lang = "Python · sklearn" }: { code: string; lang?: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = () => {
    void navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    });
  };
  return (
    <div className="rounded-xl overflow-hidden border border-slate-800/90 shadow-md ring-1 ring-slate-900/5">
      <div className="flex items-center justify-between gap-3 px-3 sm:px-4 py-2.5 bg-gradient-to-r from-slate-900 via-slate-900 to-slate-800 border-b border-slate-700/80">
        <div className="flex items-center gap-2 min-w-0">
          <span className="flex gap-1.5 shrink-0" aria-hidden>
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500/90" />
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400/90" />
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400/90" />
          </span>
          <span className="text-[10px] sm:text-[11px] text-slate-400 font-mono uppercase tracking-wide truncate">{lang}</span>
        </div>
        <button
          type="button"
          onClick={onCopy}
          aria-label={copied ? "Copied" : "Copy code"}
          className="inline-flex items-center gap-1 rounded-md px-2 py-1.5 text-[10px] font-medium text-slate-300 hover:bg-white/10 hover:text-white transition-colors shrink-0 border border-transparent hover:border-white/10"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" aria-hidden /> : <Copy className="w-3.5 h-3.5" aria-hidden />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="bg-slate-950 text-slate-100 text-[11px] sm:text-[11.5px] leading-relaxed px-4 sm:px-5 py-4 overflow-x-auto font-mono whitespace-pre border-t border-slate-800/60">
        {code}
      </pre>
    </div>
  );
}

function SectionContent({ sec }: { sec: Section }) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-slate-600 leading-relaxed">{sec.body}</p>
      <CodeBlock code={sec.code} lang={sec.lang} />
      {sec.params && sec.params.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-gradient-to-b from-slate-50/80 to-white overflow-hidden shadow-sm">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 px-4 py-2.5 border-b border-slate-200 bg-white/90">
            Parameter reference
          </p>
          <div className="divide-y divide-slate-100">
            {sec.params.map((p) => (
              <div key={p.name} className="px-4 py-3 hover:bg-slate-50/80 transition-colors">
                <div className="flex flex-wrap items-baseline gap-2 mb-1">
                  <code className="text-xs font-semibold text-blue-800 bg-blue-50 px-2 py-0.5 rounded-md border border-blue-100/80">{p.name}</code>
                  <span className="text-[10px] text-slate-500 font-medium">{p.type}</span>
                  <span className="text-[10px] text-slate-500 ml-auto tabular-nums">
                    default: <strong className="text-slate-800">{p.default}</strong>
                  </span>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">{p.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Drawer ────────────────────────────────────────────────────────────────────

interface Props {
  open: boolean;
  page: Page;
  onClose: () => void;
}

export function InfoDrawer({ open, page, onClose }: Props) {
  const content = CONTENT[page];
  const [tab, setTab] = useState("0");

  useEffect(() => {
    setTab("0");
  }, [page, open]);

  return (
    <>
      {/* Backdrop */}
      <div
        className={`fixed inset-0 z-40 bg-black/35 backdrop-blur-[2px] transition-opacity duration-300 ${
          open ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        }`}
        onClick={onClose}
        aria-hidden={!open}
      />

      {/* Drawer panel */}
      <aside
        className={`fixed top-0 right-0 z-50 flex h-full w-[min(100vw,640px)] max-w-full flex-col bg-white shadow-2xl ring-1 ring-slate-200/80 transition-transform duration-300 ease-in-out ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
        aria-hidden={!open}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-slate-200 bg-white/95 px-5 py-4 backdrop-blur-sm sm:px-6">
          <div className="min-w-0 pr-2">
            <p className="mb-1 text-[10px] font-bold uppercase tracking-widest text-blue-600/90">How it works</p>
            <h2 className="truncate text-lg font-bold leading-tight text-slate-900">{content.title}</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="shrink-0 rounded-lg p-2 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-700"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <div className="shrink-0 border-b border-slate-100 bg-gradient-to-b from-slate-50/90 to-white px-5 pb-3 pt-4 sm:px-6">
            <div className="flex gap-3 rounded-xl border border-blue-200/80 bg-gradient-to-br from-blue-50/95 to-indigo-50/50 px-4 py-3 text-sm text-slate-800 shadow-sm">
              <Info className="mt-0.5 h-5 w-5 shrink-0 text-blue-600" aria-hidden />
              <p className="leading-relaxed">{content.summary}</p>
            </div>
          </div>

          <div className="flex min-h-0 flex-1 flex-col px-5 py-4 sm:px-6">
            <Tabs value={tab} onValueChange={setTab} className="flex min-h-0 flex-1 flex-col">
              <TabsList className="max-h-[140px] w-full shrink-0 flex-wrap justify-start gap-1 overflow-y-auto rounded-xl border border-slate-200/90 bg-slate-100 p-1.5 shadow-inner">
                {content.sections.map((sec, i) => (
                  <TabsTrigger
                    key={`${page}-${i}-${sec.heading}`}
                    value={String(i)}
                    title={sec.heading}
                    className="max-w-[10.5rem] px-2.5 py-2 text-left text-[11px] data-[state=active]:shadow-md sm:max-w-[13rem] sm:text-xs"
                  >
                    <span className="block w-full truncate">{sec.tab ?? sec.heading}</span>
                  </TabsTrigger>
                ))}
              </TabsList>

              <div className="mt-4 min-h-0 flex-1 overflow-y-auto px-0.5 pb-8">
                {content.sections.map((sec, i) => (
                  <TabsContent
                    key={`${page}-${i}-${sec.heading}`}
                    value={String(i)}
                    className="mt-0 outline-none data-[state=inactive]:hidden"
                  >
                    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm ring-1 ring-slate-100 sm:p-5">
                      <h3 className="text-sm font-semibold text-slate-900">{sec.heading}</h3>
                      <div className="mb-4 mt-1 h-0.5 w-10 rounded-full bg-blue-500/90" aria-hidden />
                      <SectionContent sec={sec} />
                    </div>
                  </TabsContent>
                ))}
              </div>
            </Tabs>
          </div>
        </div>
      </aside>
    </>
  );
}
