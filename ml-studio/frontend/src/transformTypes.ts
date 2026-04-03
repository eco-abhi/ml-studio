// Shared transform type definitions used by both Transforms.tsx and EDA.tsx

export type StepType =
  | "drop_columns"
  | "impute"
  | "one_hot_encode"
  | "label_encode"
  | "clip_outliers"
  | "scale"
  | "rename_columns"
  | "math_transform"
  | "fix_skewness"
  | "bin_numeric"
  | "drop_duplicates"
  | "drop_nulls"
  | "frequency_encode"
  | "cast_dtype"
  | "polynomial_features"
  | "extract_datetime"
  | "pca_projection"
  | "tfidf_column"
  | "derive_numeric"
  | "target_encode_dataset"
  | "train_test_split";

/** Column added by the Train/Test Split transform; training uses it instead of re-splitting. */
export const ML_PIPELINE_SPLIT_COLUMN = "__ml_split__" as const;

export type DatetimePart = "year" | "month" | "day" | "hour" | "minute" | "dow" | "doy" | "week";

export interface StepBase { id: string; type: StepType }
export interface DropStep       extends StepBase { type: "drop_columns";      columns: string[] }
export interface ImputeStep     extends StepBase { type: "impute";             columns: string[]; strategy: "mean" | "median" | "mode" | "zero" }
export interface OneHotStep     extends StepBase { type: "one_hot_encode";     columns: string[] }
export interface LabelEncStep   extends StepBase { type: "label_encode";       columns: string[] }
export interface ClipStep       extends StepBase { type: "clip_outliers";      columns: string[]; method: "iqr" | "zscore" }
export interface ScaleStep      extends StepBase { type: "scale";              columns: string[]; method: "standard" | "minmax" | "robust" }
export interface RenameStep     extends StepBase { type: "rename_columns";     mapping: Record<string, string> }
export interface MathStep       extends StepBase { type: "math_transform";     columns: string[]; method: "log1p" | "sqrt" | "square" | "reciprocal" | "abs" }
export interface FixSkewStep    extends StepBase { type: "fix_skewness";        columns: string[]; method: "auto" | "log1p" | "sqrt" | "box_cox" | "yeo_johnson"; threshold: number }
export interface BinStep        extends StepBase { type: "bin_numeric";        columns: string[]; n_bins: number; strategy: "equal_width" | "quantile" }
export interface DropDupStep    extends StepBase { type: "drop_duplicates";    columns: string[]; keep: "first" | "last" | "none" }
export interface DropNullStep   extends StepBase { type: "drop_nulls";         columns: string[]; how: "any" | "all" }
export interface FreqEncStep    extends StepBase { type: "frequency_encode";   columns: string[]; normalize: boolean }
export interface CastStep       extends StepBase { type: "cast_dtype";         columns: string[]; dtype: "float" | "int" | "str" }
export interface PolynomialStep extends StepBase {
  type: "polynomial_features";
  columns: string[];
  degree: number;
  interaction_only: boolean;
  include_bias: boolean;
}
export interface ExtractDtStep extends StepBase {
  type: "extract_datetime";
  columns: string[];
  parts: DatetimePart[];
}
export interface PcaStep extends StepBase {
  type: "pca_projection";
  columns: string[];
  n_components: number;
  prefix: string;
  drop_original: boolean;
}
export interface TfidfStep extends StepBase {
  type: "tfidf_column";
  column: string;
  max_features: number;
  ngram_max: number;
}
export interface DeriveStep extends StepBase {
  type: "derive_numeric";
  column_a: string;
  column_b: string;
  op: "add" | "subtract" | "multiply" | "divide";
  output_column: string;
}
export interface TargetEncDatasetStep extends StepBase {
  type: "target_encode_dataset";
  columns: string[];
  target_column: string;
}
export interface TrainTestSplitStep extends StepBase {
  type: "train_test_split";
  target_column: string;
  test_size: number;
  random_state: number;
  shuffle: boolean;
  stratify: boolean;
}

export type Step = DropStep | ImputeStep | OneHotStep | LabelEncStep | ClipStep | ScaleStep
                | RenameStep | MathStep | FixSkewStep | BinStep | DropDupStep | DropNullStep | FreqEncStep | CastStep
                | PolynomialStep | ExtractDtStep | PcaStep | TfidfStep | DeriveStep | TargetEncDatasetStep
                | TrainTestSplitStep;

export const STEP_META: Record<StepType, { label: string; description: string }> = {
  drop_columns:      { label: "Drop Columns",        description: "Remove selected columns from the dataset" },
  rename_columns:    { label: "Rename Columns",       description: "Rename one or more columns to new names" },
  drop_duplicates:   { label: "Drop Duplicates",      description: "Remove duplicate rows (optionally scoped to selected columns)" },
  drop_nulls:        { label: "Drop Null Rows",       description: "Remove rows that have missing values in any/all selected columns" },
  cast_dtype:        { label: "Cast Dtype",           description: "Convert column values to float, int, or string" },
  impute:            { label: "Impute Missing",       description: "Fill NaN values with a calculated statistic" },
  clip_outliers:     { label: "Clip Outliers",        description: "Cap extreme values using IQR or Z-score bounds" },
  scale:             { label: "Scale / Normalise",    description: "Standardise or normalise numeric column values" },
  math_transform:    { label: "Math Transform",       description: "Apply log1p, sqrt, square, reciprocal, or abs to numeric columns" },
  fix_skewness:      { label: "Fix Skewness",         description: "Auto-correct skewed distributions using log1p, sqrt, Box-Cox, or Yeo-Johnson — skips columns below the threshold" },
  bin_numeric:       { label: "Bin Numeric",          description: "Discretise a numeric column into equal-width or quantile bins" },
  one_hot_encode:    { label: "One-Hot Encode",       description: "Encode categorical column as binary dummy variables" },
  label_encode:      { label: "Label Encode",         description: "Map categories to integers (0, 1, 2…)" },
  frequency_encode:  { label: "Frequency Encode",     description: "Replace each category with its frequency (count or proportion)" },
  polynomial_features: { label: "Polynomial Features", description: "Generate polynomial and interaction terms from selected numeric columns" },
  extract_datetime:  { label: "Extract Datetime Parts", description: "Parse columns as dates and add year, month, day, hour, etc. as numeric features" },
  pca_projection:    { label: "PCA Projection",       description: "Fit PCA on selected numeric columns and append principal component scores" },
  tfidf_column:      { label: "TF‑IDF (text column)", description: "Vectorise one text column into a bag of TF‑IDF numeric features (max 200 terms)" },
  derive_numeric:    { label: "Derive Column (arithmetic)", description: "Create a new numeric column from two columns using +, −, ×, or ÷" },
  target_encode_dataset: { label: "Target Encode (dataset)", description: "Replace categories with mean target (uses full data — can leak; prefer Train tab target encoding for modeling)" },
  train_test_split: { label: "Train / Test Split", description: "Label each row as train or test (adds __ml_split__). Reorder like any other step. Train tab uses __ml_split__ when that column remains after all steps." },
};

/** UI grouping for searchable transform pickers (EDA Quick Transform, Transforms add-step). */
export const TRANSFORM_OPTION_GROUPS: { category: string; types: readonly StepType[] }[] = [
  { category: "Schema & columns", types: ["drop_columns", "rename_columns", "cast_dtype"] },
  { category: "Rows", types: ["drop_duplicates", "drop_nulls"] },
  { category: "Missing values", types: ["impute"] },
  { category: "Outliers & scale", types: ["clip_outliers", "scale"] },
  { category: "Distributions", types: ["math_transform", "fix_skewness", "bin_numeric"] },
  { category: "Encoding", types: ["one_hot_encode", "label_encode", "frequency_encode", "target_encode_dataset"] },
  { category: "Feature engineering", types: ["polynomial_features", "extract_datetime", "pca_projection", "tfidf_column", "derive_numeric"] },
  { category: "Modeling prep", types: ["train_test_split"] },
];

export function makeStep(type: StepType): Step {
  const id = Math.random().toString(36).slice(2, 8);
  switch (type) {
    case "impute":           return { id, type, columns: [], strategy: "mean" };
    case "clip_outliers":    return { id, type, columns: [], method: "iqr" };
    case "scale":            return { id, type, columns: [], method: "standard" };
    case "rename_columns":   return { id, type, mapping: {} };
    case "math_transform":   return { id, type, columns: [], method: "log1p" };
    case "fix_skewness":     return { id, type, columns: [], method: "auto", threshold: 0.5 };
    case "bin_numeric":      return { id, type, columns: [], n_bins: 5, strategy: "equal_width" };
    case "drop_duplicates":  return { id, type, columns: [], keep: "first" };
    case "drop_nulls":       return { id, type, columns: [], how: "any" };
    case "frequency_encode": return { id, type, columns: [], normalize: true };
    case "cast_dtype":       return { id, type, columns: [], dtype: "float" };
    case "polynomial_features":
      return { id, type, columns: [], degree: 2, interaction_only: false, include_bias: false };
    case "extract_datetime":
      return { id, type, columns: [], parts: ["year", "month", "day"] };
    case "pca_projection":
      return { id, type, columns: [], n_components: 3, prefix: "PC_", drop_original: false };
    case "tfidf_column":
      return { id, type, column: "", max_features: 50, ngram_max: 1 };
    case "derive_numeric":
      return { id, type, column_a: "", column_b: "", op: "add", output_column: "derived_feature" };
    case "target_encode_dataset":
      return { id, type, columns: [], target_column: "" };
    case "train_test_split":
      return {
        id,
        type,
        target_column: "",
        test_size: 0.2,
        random_state: 42,
        shuffle: true,
        stratify: true,
      };
    default:                 return { id, type, columns: [] } as DropStep | OneHotStep | LabelEncStep;
  }
}

export function serializeStep(s: Step): object {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { id: _id, ...rest } = s;
  return rest;
}

/** Re-attach a random id to a raw step object coming from the API */
export function deserializeStep(raw: Record<string, unknown>): Step {
  return { id: Math.random().toString(36).slice(2, 8), ...raw } as Step;
}

/** Clone steps for editing in the UI (fresh ids; same serialized payload as server). */
export function cloneStepsForEdit(steps: Step[]): Step[] {
  return steps.map((s) => deserializeStep(serializeStep(s) as Record<string, unknown>));
}
