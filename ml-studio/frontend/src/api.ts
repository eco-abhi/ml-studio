const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

function withAuth(init: RequestInit = {}): RequestInit {
  const key = import.meta.env.VITE_API_KEY;
  const headers = new Headers(init.headers as HeadersInit | undefined);
  if (key) headers.set("X-API-Key", String(key));
  return { ...init, headers };
}

// ── Shared types ──────────────────────────────────────────────────────────────

export type TaskType = "regression" | "classification";

export interface UploadResult {
  dataset_id: string;
  filename: string;
  shape: [number, number];
  columns: string[];
  storage: "supabase" | "local";
}

export interface PreviewResult {
  columns: string[];
  numeric_columns: string[];
  dtypes: Record<string, string>;
  preview: (string | number | null)[][];
  shape: [number, number];
  missing: Record<string, number>;
}

export interface ModelResult {
  model:        string;
  // regression
  rmse?:        number;
  mae?:         number;
  r2?:          number;
  train_rmse?:  number;
  train_mae?:   number;
  train_r2?:    number;
  cv_rmse?:     number;
  cv_mae?:      number;
  cv_r2?:       number;
  // classification
  accuracy?:    number;
  f1_score?:    number;
  train_accuracy?: number;
  train_f1?:    number;
  roc_auc?:     number;
  cv_accuracy?: number;
  cv_f1?:       number;
  // shared cv
  cv_std?:      number;
  cv_folds_run?:number;
}

export interface SplitConfig {
  test_size:    number;
  random_state: number;
  shuffle:      boolean;
  stratify:     boolean;
  val_strategy: string;
  cv_folds:     number;
  cv_repeats:   number;
  val_size:     number;
  scaler:       string;
  /** Classification only: none | smote | random_under */
  imbalance_method: string;
  /** 0 = off; else PCA(n) after preprocessing */
  train_pca_components: number;
  /** numeric_only (default) | target_encode | one_hot */
  categorical_encoding: string;
  max_category_cardinality: number;
  /** Set by backend when the dataset has __ml_split__ from Transforms */
  holdout_from_transforms?: boolean;
  test_size_effective?: number;
}

export interface TrainResult {
  task_type:     TaskType;
  n_features:    number;
  n_train:       number;
  n_test:        number;
  feature_names: string[];
  split_config:  SplitConfig;
  results:       ModelResult[];
}

export interface ExperimentRun extends ModelResult {
  run_id: string;
  status: string;
  params?: Record<string, number | string>;
  started_at?: number;
}

export interface ImportanceItem {
  feature: string;
  importance: number;
}

export interface ColumnStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  q25: number;
  q75: number;
  hist_values: number[];
  hist_bins: number[];
}

export interface EDAResult {
  stats: Record<string, ColumnStats>;
  correlations: Record<string, Record<string, number>>;
  shape: [number, number];
  columns: string[];
}

export interface CorrelationMatrix {
  columns: string[];
  matrix: number[][];
}

export interface MissingValues {
  [col: string]: { count: number; pct: number };
}

export interface OutlierInfo {
  [col: string]: { lower: number; upper: number; n_outliers: number };
}

export interface ScatterData {
  x: number[];
  y: number[];
  color?: (number | string)[];
}

export interface CategoricalStats {
  [col: string]: { n_unique: number; top_values: [string, number][] };
}

export interface HealthResult {
  n_rows: number;
  n_cols: number;
  n_duplicates: number;
  memory_mb: number;
  constant_cols: string[];
  near_zero_cols: string[];
  high_card_cols: string[];
  completeness: Record<string, number>;
  variance: Record<string, number>;
  dtype_counts: { numeric: number; categorical: number };
  numeric_cols: string[];
  cat_cols: string[];
}

export interface TargetAnalysis {
  target_col: string;
  is_numeric: boolean;
  n: number;
  // numeric
  hist_values?: number[];
  hist_bins?: number[];
  skewness?: number;
  kurtosis?: number;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  median?: number;
  transform_hint?: string | null;
  // categorical
  n_classes?: number;
  class_counts?: [string, number, number][];
  imbalance_ratio?: number;
  is_imbalanced?: boolean;
}

export type SkewnessRow = {
  column: string;
  skewness: number;
  kurtosis: number;
  abs_skewness: number;
  severity: "normal" | "moderate" | "high";
  hint: string | null;
};

export interface BoxplotData {
  [col: string]: {
    min: number; q1: number; median: number; mean: number;
    q3: number; max: number;
    lower_fence: number; upper_fence: number;
    outliers: number[];
  };
}

export interface PairplotData {
  columns: string[];
  pairs: Record<string,
    | { type: "hist"; values: number[]; bins: number[] }
    | { type: "scatter"; x: number[]; y: number[] }
  >;
}

export interface FeatureTargetData {
  target_col: string;
  is_target_numeric: boolean;
  features: Record<string,
    | { type: "scatter"; x: number[]; y: number[]; correlation: number; trend: { x: number[]; y: number[]; slope: number } | null }
    | { type: "boxgroup"; classes: Record<string, { q1: number; median: number; mean: number; q3: number; min: number; max: number }> }
  >;
}

export interface PredictionResult {
  prediction: number | string | null;
  confidence: number | null;
}

export interface TransformPreview {
  before: { columns: string[]; rows: unknown[][] };
  after:  { columns: string[]; rows: unknown[][] };
}

export interface TransformApplyResult {
  shape: [number, number];
  columns: string[];
  steps_applied: number;
}

// ── Upload & preview ──────────────────────────────────────────────────────────

export const uploadFile = (file: File): Promise<UploadResult> => {
  const fd = new FormData();
  fd.append("file", file);
  return fetch(`${BASE}/upload`, withAuth({ method: "POST", body: fd })).then((r) => r.json());
};

export const uploadURL = (url: string): Promise<UploadResult> =>
  fetch(`${BASE}/upload?${new URLSearchParams({ url })}`, withAuth({ method: "POST" })).then((r) => r.json());

export const previewDataset = (datasetId: string, limit = 10): Promise<PreviewResult> =>
  fetch(`${BASE}/preview/${datasetId}?limit=${limit}`, withAuth()).then((r) => r.json());

// ── Training ──────────────────────────────────────────────────────────────────

export const trainModels = (
  datasetId: string,
  targetCol: string,
  taskType: TaskType | null = null,
  hyperparams: Record<string, Record<string, number | string>> = {},
  models: string[] | null = null,
  splitConfig: Partial<SplitConfig> = {},
): Promise<TrainResult> => {
  const p = new URLSearchParams({ target_col: targetCol });
  if (taskType) p.append("task_type", taskType);
  return fetch(`${BASE}/train/${datasetId}?${p}`, withAuth({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hyperparams, models, split_config: splitConfig }),
  })).then((r) => r.json());
};

/** Progress lines from POST /train/{id}/stream (before final `complete`). */
export type TrainStreamProgress =
  | { phase: "prepare"; message?: string }
  | { phase: "model"; current: number; total: number; model: string };

/** Stream training with NDJSON progress; same end result as {@link trainModels}. */
export async function trainModelsWithProgress(
  datasetId: string,
  targetCol: string,
  taskType: TaskType | null,
  hyperparams: Record<string, Record<string, number | string>>,
  models: string[] | null,
  splitConfig: Partial<SplitConfig>,
  onProgress: (e: TrainStreamProgress) => void,
): Promise<TrainResult> {
  const p = new URLSearchParams({ target_col: targetCol });
  if (taskType) p.append("task_type", taskType);
  const res = await fetch(`${BASE}/train/${datasetId}/stream?${p}`, withAuth({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hyperparams, models, split_config: splitConfig }),
  }));
  if (!res.ok) {
    const text = await res.text();
    let msg = text || res.statusText;
    try {
      const j = JSON.parse(text) as { detail?: unknown };
      if (typeof j.detail === "string") msg = j.detail;
    } catch {
      /* keep msg */
    }
    throw new Error(msg);
  }
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      const t = line.trim();
      if (!t) continue;
      const obj = JSON.parse(t) as Record<string, unknown>;
      if (obj.phase === "error") {
        throw new Error(String(obj.detail ?? "Training failed"));
      }
      if (obj.phase === "complete" && obj.result != null) {
        return obj.result as TrainResult;
      }
      if (obj.phase === "prepare" || obj.phase === "model") {
        onProgress(obj as TrainStreamProgress);
      }
    }
  }
  throw new Error("Training stream ended without a result");
}

// ── Experiments ───────────────────────────────────────────────────────────────

export const getExperiments = (datasetId: string): Promise<{ runs: ExperimentRun[] }> =>
  fetch(`${BASE}/experiments/${datasetId}`, withAuth()).then((r) => r.json());

function runIdQuery(runId?: string | null): string {
  const t = runId?.trim();
  if (!t) return "";
  return `run_id=${encodeURIComponent(t)}`;
}

// ── Importances ───────────────────────────────────────────────────────────────

export const getImportances = (
  datasetId: string,
  modelName: string,
  runId?: string | null,
): Promise<ImportanceItem[]> => {
  const q = runIdQuery(runId);
  const path = `${BASE}/importances/${datasetId}/${encodeURIComponent(modelName)}${q ? `?${q}` : ""}`;
  return fetch(path, withAuth()).then(async (r) => {
    const j = await r.json();
    if (!r.ok) throw new Error((j as { detail?: string }).detail || "Importances failed");
    return j as ImportanceItem[];
  });
};

// ── Diagnostics ───────────────────────────────────────────────────────────────

export interface DiagnosticsPayload {
  task_type: TaskType;
  confusion_matrix?: { labels: string[]; matrix: number[][] };
  classification_report?: Record<string, Record<string, number> | number | undefined>;
  roc_curve?: { fpr: number[]; tpr: number[]; auc: number };
  roc_auc_macro_ovr?: number;
  calibration?: { mean_predicted: number[]; fraction_positives: number[] };
  regression?: Record<string, number>;
  residual_histogram?: { counts: number[]; edges: number[] };
  predicted_vs_actual?: { actual: number[]; predicted: number[] };
  residuals_vs_predicted?: { predicted: number[]; residuals: number[] };
  scaler_baseline_compare?: {
    comparisons: { scaler: string; rmse: number; r2: number }[];
    best_scaler: string;
  };
  learning_curve?: {
    train_sizes: number[];
    train_score_mean: number[];
    val_score_mean: number[];
    metric: "rmse" | "accuracy";
  } | null;
  permutation_importance?: { feature: string; mean: number; std: number }[];
}

export const getDiagnostics = (
  datasetId: string,
  modelName: string,
  runId?: string | null,
): Promise<DiagnosticsPayload> => {
  const q = runIdQuery(runId);
  const path = `${BASE}/diagnostics/${datasetId}/${encodeURIComponent(modelName)}${q ? `?${q}` : ""}`;
  return fetch(path, withAuth()).then(async (r) => {
    const j = await r.json();
    if (!r.ok) throw new Error((j as { detail?: string }).detail || "Diagnostics failed");
    return j as DiagnosticsPayload;
  });
};

/** Download the trained sklearn Pipeline as gzip-compressed joblib (same artifact MLflow stores). */
export async function downloadTrainedPipeline(
  datasetId: string,
  modelName: string,
  runId?: string | null,
): Promise<void> {
  const q = runIdQuery(runId);
  const url = `${BASE}/export/${datasetId}/${encodeURIComponent(modelName)}/pipeline.joblib${q ? `?${q}` : ""}`;
  const r = await fetch(url, withAuth());
  if (!r.ok) {
    const j = (await r.json().catch(() => ({}))) as { detail?: string };
    throw new Error(j.detail || "Pipeline download failed");
  }
  const blob = await r.blob();
  const a = document.createElement("a");
  const cd = r.headers.get("Content-Disposition");
  const fnMatch = cd?.match(/filename="([^"]+)"/) ?? cd?.match(/filename=([^;]+)/);
  const safe = modelName.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_|_$/g, "") || "pipeline";
  const fallback =
    runId?.trim() ? `${safe}_pipeline_${runId.trim().slice(0, 8)}.joblib` : `${safe}_pipeline.joblib`;
  a.download = (fnMatch?.[1] ?? fallback).trim();
  a.href = URL.createObjectURL(blob);
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Predict ───────────────────────────────────────────────────────────────────

export const predict = (
  datasetId: string,
  modelName: string,
  features: Record<string, number>,
  runId?: string | null,
): Promise<PredictionResult> => {
  const q = runIdQuery(runId);
  const path = `${BASE}/predict/${datasetId}/${encodeURIComponent(modelName)}${q ? `?${q}` : ""}`;
  return fetch(path, withAuth({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  })).then(async (r) => {
    const j = await r.json();
    if (!r.ok) throw new Error((j as { detail?: string }).detail || "Prediction failed");
    return j as PredictionResult;
  });
};

// ── EDA ───────────────────────────────────────────────────────────────────────

/** `transformed` = active working CSV (post-transform if any); `original` = uploaded file only. */
export type EdaDataSource = "transformed" | "original";

function edaSourceQuery(source?: EdaDataSource): string {
  return source === "original" ? "data_source=original" : "";
}

export const getEDA = (datasetId: string, source?: EdaDataSource): Promise<EDAResult> => {
  const q = edaSourceQuery(source);
  return fetch(`${BASE}/eda/${datasetId}${q ? `?${q}` : ""}`, withAuth()).then((r) => r.json());
};

export const getCorrelationMatrix = (datasetId: string, source?: EdaDataSource): Promise<CorrelationMatrix> => {
  const q = edaSourceQuery(source);
  return fetch(`${BASE}/eda/${datasetId}/matrix${q ? `?${q}` : ""}`, withAuth()).then((r) => r.json());
};

export const getMissingValues = (datasetId: string, source?: EdaDataSource): Promise<MissingValues> => {
  const q = edaSourceQuery(source);
  return fetch(`${BASE}/eda/${datasetId}/missing${q ? `?${q}` : ""}`, withAuth()).then((r) => r.json());
};

export const getOutliers = (datasetId: string, source?: EdaDataSource): Promise<OutlierInfo> => {
  const q = edaSourceQuery(source);
  return fetch(`${BASE}/eda/${datasetId}/outliers${q ? `?${q}` : ""}`, withAuth()).then((r) => r.json());
};

export const getScatter = (
  datasetId: string,
  colX: string,
  colY: string,
  target?: string,
  source?: EdaDataSource
): Promise<ScatterData> => {
  const p = new URLSearchParams({ col_x: colX, col_y: colY });
  if (target) p.append("target", target);
  if (source === "original") p.set("data_source", "original");
  return fetch(`${BASE}/eda/${datasetId}/scatter?${p}`, withAuth()).then((r) => r.json());
};

export const getCategorical = (datasetId: string, source?: EdaDataSource): Promise<CategoricalStats> => {
  const q = edaSourceQuery(source);
  return fetch(`${BASE}/eda/${datasetId}/categorical${q ? `?${q}` : ""}`, withAuth()).then((r) => r.json());
};

export const getHealth = (datasetId: string, source?: EdaDataSource): Promise<HealthResult> => {
  const q = edaSourceQuery(source);
  return fetch(`${BASE}/eda/${datasetId}/health${q ? `?${q}` : ""}`, withAuth()).then((r) => r.json());
};

export const getTargetAnalysis = (datasetId: string, targetCol: string, source?: EdaDataSource): Promise<TargetAnalysis> => {
  const p = new URLSearchParams({ target_col: targetCol });
  if (source === "original") p.set("data_source", "original");
  return fetch(`${BASE}/eda/${datasetId}/target?${p}`, withAuth()).then((r) => r.json());
};

export const getSkewness = (datasetId: string, source?: EdaDataSource): Promise<SkewnessRow[]> => {
  const q = edaSourceQuery(source);
  return fetch(`${BASE}/eda/${datasetId}/skewness${q ? `?${q}` : ""}`, withAuth()).then((r) => r.json());
};

export const getBoxplots = (datasetId: string, columns?: string[], source?: EdaDataSource): Promise<BoxplotData> => {
  const p = new URLSearchParams();
  if (columns?.length) p.set("columns", columns.join(","));
  if (source === "original") p.set("data_source", "original");
  const qs = p.toString();
  return fetch(`${BASE}/eda/${datasetId}/boxplot${qs ? `?${qs}` : ""}`, withAuth()).then((r) => r.json());
};

export const getPairplot = (datasetId: string, columns?: string[], source?: EdaDataSource): Promise<PairplotData> => {
  const p = new URLSearchParams();
  if (columns?.length) p.set("columns", columns.join(","));
  if (source === "original") p.set("data_source", "original");
  const qs = p.toString();
  return fetch(`${BASE}/eda/${datasetId}/pairplot${qs ? `?${qs}` : ""}`, withAuth()).then((r) => r.json());
};

export const getFeatureTarget = (datasetId: string, targetCol: string, source?: EdaDataSource): Promise<FeatureTargetData> => {
  const p = new URLSearchParams({ target_col: targetCol });
  if (source === "original") p.set("data_source", "original");
  return fetch(`${BASE}/eda/${datasetId}/feature-target?${p}`, withAuth()).then((r) => r.json());
};

// ── Tune ──────────────────────────────────────────────────────────────────────

export interface TuneTrial {
  number: number;
  value: number;
  params: Record<string, number | string>;
  state: string;
}

export interface TuneResult {
  model: string;
  task_type: string;
  n_trials: number;
  best_params: Record<string, number | string>;
  best_value: number;
  metric_name: string;
  metrics: Record<string, number>;
  trials: TuneTrial[];
  feature_names: string[];
  split_config: SplitConfig;
}

export const tuneModel = (
  datasetId: string,
  targetCol: string,
  model: string,
  nTrials: number,
  taskType: string | null,
  splitConfig: Partial<SplitConfig> = {},
): Promise<TuneResult> => {
  const p = new URLSearchParams({ target_col: targetCol, model, n_trials: String(nTrials) });
  if (taskType) p.append("task_type", taskType);
  return fetch(`${BASE}/tune/${datasetId}?${p}`, withAuth({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ split_config: splitConfig }),
  })).then((r) => r.json());
};

// ── Hold-out split (Split tab — before transforms) ────────────────────────────

export interface HoldoutSplitConfig {
  target_column: string;
  test_size: number;
  random_state: number;
  shuffle: boolean;
  stratify: boolean;
}

export interface HoldoutSplitStatus {
  configured: boolean;
  config: HoldoutSplitConfig | null;
  column_present: boolean;
  counts: { train: number; test: number } | null;
}

export const getHoldoutSplitStatus = (datasetId: string): Promise<HoldoutSplitStatus> =>
  fetch(`${BASE}/datasets/${datasetId}/holdout-split`, withAuth()).then((r) => r.json());

export const saveHoldoutSplit = (
  datasetId: string,
  body: HoldoutSplitConfig,
): Promise<{ shape: [number, number]; columns: string[]; counts: { train: number; test: number } }> =>
  fetch(`${BASE}/datasets/${datasetId}/holdout-split`, withAuth({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })).then(async (r) => {
    const j = await r.json();
    if (!r.ok) throw new Error((j as { detail?: string }).detail || "Hold-out split failed");
    return j as { shape: [number, number]; columns: string[]; counts: { train: number; test: number } };
  });

// ── Transforms ────────────────────────────────────────────────────────────────

export const previewTransform = (
  datasetId: string,
  steps: object[],
  n = 5,
  fromOriginal = false
): Promise<TransformPreview> =>
  fetch(`${BASE}/transform/${datasetId}/preview`, withAuth({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ steps, n, from_original: fromOriginal }),
  })).then((r) => r.json());

export const applyTransform = (
  datasetId: string,
  steps: object[],
  fromOriginal = false
): Promise<TransformApplyResult> =>
  fetch(`${BASE}/transform/${datasetId}/apply`, withAuth({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ steps, from_original: fromOriginal }),
  })).then(async (r) => {
    const j = await r.json();
    if (!r.ok) throw new Error((j as { detail?: string }).detail || "Transform apply failed");
    return j as TransformApplyResult;
  });

export const resetTransform = (datasetId: string): Promise<{ status: string }> =>
  fetch(`${BASE}/transform/${datasetId}/reset`, withAuth({ method: "POST" })).then((r) => r.json());

export interface TransformHistory {
  steps: object[];
  applied_at: string | null;
  active: boolean;
}

export const getTransformHistory = (datasetId: string): Promise<TransformHistory> =>
  fetch(`${BASE}/transform/${datasetId}/history`, withAuth()).then((r) => r.json());

// ── Misc ──────────────────────────────────────────────────────────────────────

export const listDatasets = (): Promise<{ datasets: unknown[] }> =>
  fetch(`${BASE}/datasets`, withAuth()).then((r) => r.json());

export const health = (): Promise<{ status: string; storage: string }> =>
  fetch(`${BASE}/health`, withAuth()).then((r) => r.json());

export interface MLflowInfo {
  tracking_uri: string;
  ui_url: string;
  is_remote: boolean;
  /** When using local sqlite/file store: shell command to open MLflow UI against the same backend. */
  local_ui_command?: string | null;
  /** When using uv from `backend/`: `uv run python mlflow_ui_cli.py` (same store as the API). */
  local_ui_command_uv?: string | null;
  /** Why plain `mlflow ui` without --backend-store-uri breaks (different store than the API). */
  local_tracking_note?: string | null;
}

export const getMLflowInfo = (): Promise<MLflowInfo> =>
  fetch(`${BASE}/mlflow-info`, withAuth()).then((r) => r.json());
