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
  cv_rmse?:     number;
  cv_mae?:      number;
  cv_r2?:       number;
  // classification
  accuracy?:    number;
  f1_score?:    number;
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

// ── Experiments ───────────────────────────────────────────────────────────────

export const getExperiments = (datasetId: string): Promise<{ runs: ExperimentRun[] }> =>
  fetch(`${BASE}/experiments/${datasetId}`, withAuth()).then((r) => r.json());

// ── Importances ───────────────────────────────────────────────────────────────

export const getImportances = (datasetId: string, modelName: string): Promise<ImportanceItem[]> =>
  fetch(`${BASE}/importances/${datasetId}/${modelName}`, withAuth()).then((r) => r.json());

// ── Predict ───────────────────────────────────────────────────────────────────

export const predict = (
  datasetId: string,
  modelName: string,
  features: Record<string, number>
): Promise<PredictionResult> =>
  fetch(`${BASE}/predict/${datasetId}/${modelName}`, withAuth({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
  })).then((r) => r.json());

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
  })).then((r) => r.json());

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
}

export const getMLflowInfo = (): Promise<MLflowInfo> =>
  fetch(`${BASE}/mlflow-info`, withAuth()).then((r) => r.json());
