import { AlertTriangle } from "lucide-react";
import type { StepType } from "./transformTypes";
import { STEP_META, TRANSFORM_OPTION_GROUPS } from "./transformTypes";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";

/** One caution: short title plus why it matters. */
export interface TransformWarningItem {
  title: string;
  reason: string;
}

/**
 * Warnings and caveats per transform type (educational; not exhaustive).
 * Keys must cover every StepType.
 */
export const TRANSFORM_WARNINGS: Record<StepType, TransformWarningItem[]> = {
  drop_columns: [
    {
      title: "Easy to remove the target or ID by mistake",
      reason:
        "If you drop the column you later use as the prediction target in Train, training will fail. Dropping leaky IDs (good) is easy to confuse with dropping legitimate features—double-check column names before apply.",
    },
    {
      title: "Downstream steps and docs still expect old names",
      reason:
        "Later transforms, EDA, and notebooks that reference removed columns will break. Renaming is often safer than dropping if you only need to exclude a column from modeling (you can also drop in Train via feature selection).",
    },
  ],
  rename_columns: [
    {
      title: "Names must match exactly everywhere afterward",
      reason:
        "The Train tab target picker, other transform steps, and API clients use string column names. A typo in the mapping silently breaks later steps or training.",
    },
    {
      title: "Collisions overwrite data",
      reason:
        "Renaming two different source columns to the same target name yields undefined behavior (last write wins). Ensure every new name is unique.",
    },
  ],
  cast_dtype: [
    {
      title: "Invalid values become NaN or null",
      reason:
        "Forcing float/int on messy strings drops information: coercions fail silently into NaN, which can cascade into imputation or row drops later.",
    },
    {
      title: "Int cast can truncate real-valued data",
      reason:
        "Rounding to integers loses magnitude detail. Use only when the column is truly ordinal/count-like.",
    },
  ],
  drop_duplicates: [
    {
      title: "Legitimate repeated rows can be deleted",
      reason:
        "If empty subset is used, full-row duplicates are removed—but repeated measurements (same features, different time or batch) are not duplicates and should not be dropped without domain review.",
    },
    {
      title: "Subset choice changes what counts as duplicate",
      reason:
        "Deduplicating on a subset keeps rows that differ only in non-subset columns; conversely, too-wide a subset may leave duplicates you intended to remove.",
    },
  ],
  drop_nulls: [
    {
      title: "High row loss on sparse features",
      reason:
        "Dropping rows with any null in selected columns (or in all columns if misconfigured) can erase most of the dataset. Check row counts after preview.",
    },
    {
      title: "Bias if missingness relates to the target",
      reason:
        "If missing values are not missing at random, removing those rows skews the distribution of y and can inflate or deflate model quality versus production.",
    },
  ],
  impute: [
    {
      title: "Statistics are computed on the current table",
      reason:
        "Mean/median/mode use all rows present at this step. If train/test split happens later in another tool, this still matches “fit on full preprocessing table” behavior—be intentional about order relative to your split strategy.",
    },
    {
      title: "Mean/median on skewed or categorical-as-numeric columns",
      reason:
        "Imputing with mean on heavy-tailed data pulls mass toward the center; mode on high-cardinality text stored as object may be misleading. Zero-fill changes scale and interpretation.",
    },
  ],
  clip_outliers: [
    {
      title: "Real extremes are flattened",
      reason:
        "Clipping replaces genuine tail events (fraud spikes, rare outcomes) with caps, which can hurt models that rely on extremes and changes calibration.",
    },
    {
      title: "IQR assumes a roughly unimodal spread",
      reason:
        "Multimodal or heavy-tailed distributions make IQR thresholds arbitrary; Z-score clipping assumes approximate normality per column.",
    },
  ],
  scale: [
    {
      title: "Fitted on all rows in the pipeline at this point",
      reason:
        "Standardisation uses mean/std (or robust equivalents) from the dataframe as transformed so far. If you have not fixed a train/test split yet, test rows influenced the scaler statistics—prefer a pipeline order where split is last, or scale inside Train’s pipeline only on training folds.",
    },
    {
      title: "Sensitive to outliers before robust scaling",
      reason:
        "Standard and min–max scalers are pulled by outliers; robust helps but does not remove the need to inspect extremes.",
    },
  ],
  math_transform: [
    {
      title: "Domain constraints",
      reason:
        "log1p and sqrt clip or require non-negative inputs; reciprocal creates infinities where denominators are zero unless handled; interpret coefficients on transformed scale only.",
    },
    {
      title: "Irreversible for reporting",
      reason:
        "Stakeholders often need predictions on the original scale; keep an inverse mapping or document the transform chain.",
    },
  ],
  fix_skewness: [
    {
      title: "Auto mode can pick different transforms per column",
      reason:
        "Reproducibility is fine with fixed random seeds elsewhere, but “auto” paths (log1p vs Yeo-Johnson) change interpretation and comparability across runs if data drifts.",
    },
    {
      title: "Box-Cox needs strictly positive values",
      reason:
        "The backend shifts values when needed, but edge cases can still fail or produce unstable transforms on very small samples.",
    },
  ],
  bin_numeric: [
    {
      title: "Information loss",
      reason:
        "Bins collapse continuous signal; model capacity needed to recover smooth relationships is gone unless you use many bins (which reintroduces sparsity and noise).",
    },
    {
      title: "Quantile bins unstable on small n",
      reason:
        "Duplicate edges and uneven bin counts happen with few rows or discrete inputs; equal-width bins can leave empty bins.",
    },
  ],
  one_hot_encode: [
    {
      title: "Cardinality explosion",
      reason:
        "High-cardinality categoricals create hundreds of sparse columns, slowing training and increasing overfitting risk. Consider frequency encoding, target encoding (CV-safe in Train), or grouping rare levels first.",
    },
    {
      title: "Rare categories at scoring time",
      reason:
        "New category levels in production won’t match training dummies unless your modeling pipeline handles unknowns explicitly.",
    },
  ],
  label_encode: [
    {
      title: "Implies an order linear models do not intend",
      reason:
        "Integer codes look ordinal; linear and distance-based models treat 2 as “between” 1 and 3 even for nominal categories. Trees handle this better but interpretation is still opaque.",
    },
  ],
  frequency_encode: [
    {
      title: "Uses full-dataset frequencies",
      reason:
        "Counts/proportions leak global structure into every row. For strict hold-out evaluation, prefer encodings fit only on training data inside the Train tab or a CV-safe recipe.",
    },
    {
      title: "Unseen categories at inference map poorly",
      reason:
        "New levels get missing or zero frequency unless you define a fallback policy outside this transform.",
    },
  ],
  target_encode_dataset: [
    {
      title: "Severe target leakage for the same file you train on",
      reason:
        "Category means use the target over the entire dataframe. Metrics on a model trained on this file will be optimistically biased. For real modeling, use Train → categorical encoding with target encoding inside CV.",
    },
    {
      title: "Only meaningful when target is numeric or coerced",
      reason:
        "Non-numeric targets are coerced with errors='coerce'; check for NaN means and empty groups.",
    },
  ],
  polynomial_features: [
    {
      title: "Column count explodes",
      reason:
        "Degree 3+ on many numeric columns creates an enormous, correlated feature matrix—slow, memory-heavy, and prone to overfitting without strong regularisation.",
    },
    {
      title: "Multicollinearity",
      reason:
        "Polynomials and interactions are highly correlated; linear models need regularisation; trees may redundant-split.",
    },
  ],
  extract_datetime: [
    {
      title: "Parse failures become missing parts",
      reason:
        "Unparseable strings yield NaT; derived numeric parts can be NaN and interact badly with later impute/drop-null steps.",
    },
    {
      title: "Time zones and locale",
      reason:
        "Parsing is naive unless your source strings include offsets; daylight saving and locale formats can shift day/hour features.",
    },
  ],
  pca_projection: [
    {
      title: "Fit uses all rows at this step",
      reason:
        "Principal components reflect the combined train+test (pre-split) geometry if you have not isolated rows yet. That leaks test structure into the basis unless split is already fixed and you only fit on train (this app fits PCA on the full transformed table).",
    },
    {
      title: "Interpretability",
      reason:
        "Components mix original features; explaining “PC2” to stakeholders is harder than explaining raw inputs.",
    },
  ],
  tfidf_column: [
    {
      title: "Vocabulary fixed to training-like corpus",
      reason:
        "max_features truncates to top terms in this dataset; completely new words at prediction time become zeros—monitor OOV rate.",
    },
    {
      title: "Short or noisy text",
      reason:
        "Very short strings produce sparse or empty vectors; heavy punctuation and casing affect tokens.",
    },
  ],
  derive_numeric: [
    {
      title: "Division by zero",
      reason:
        "Dividing by a column that contains zeros yields infinities or NaNs in pandas unless handled; downstream steps may drop or distort those rows.",
    },
    {
      title: "Name collisions",
      reason:
        "Overwriting an existing output_column destroys prior data silently.",
    },
  ],
  train_test_split: [
    {
      title: "Order and leakage",
      reason:
        "This pipeline applies each step to the whole dataframe: impute, scale, PCA, etc. still compute statistics on every row in the selected columns, even if __ml_split__ was added earlier. Putting the split last only limits how many steps see both splits together; it does not replace sklearn-style fit-on-train, transform-test inside a model pipeline.",
    },
    {
      title: "Stratify target must align with Train",
      reason:
        "Use the same column as your training target for stratification. Rows with a null target are not assigned a split label and are dropped when modeling after row-wise dropna.",
    },
    {
      title: "Changing the pipeline invalidates the split",
      reason:
        "If you remove or redo this step, row labels change—compare experiments only when the split column is stable or retrain from scratch.",
    },
  ],
};

/** Second tab on Transforms: grouped warnings with rationale per step type. */
export function TransformsWarningsGuide() {
  return (
    <div className="space-y-8 max-w-4xl">
      <p className="text-sm text-slate-600 leading-relaxed">
        Each transform can change your data in ways that affect validity, leakage, and interpretation. Below are common{" "}
        <strong className="text-slate-800">warnings</strong> and the <strong className="text-slate-800">reason</strong> each
        matters. This is guidance, not an exhaustive safety review—always inspect previews and row counts.
      </p>
      {TRANSFORM_OPTION_GROUPS.map((group) => (
        <section key={group.category} className="space-y-4">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 border-b border-slate-200 pb-2">
            {group.category}
          </h2>
          <div className="space-y-4">
            {group.types.map((type) => (
              <Card key={type} className="border-slate-200 shadow-sm">
                <CardHeader className="pb-2 pt-4">
                  <CardTitle className="text-sm font-semibold text-slate-900">{STEP_META[type].label}</CardTitle>
                  <p className="text-xs text-slate-500 font-normal mt-1 leading-relaxed">{STEP_META[type].description}</p>
                </CardHeader>
                <CardContent className="space-y-3 pt-0 pb-4">
                  {TRANSFORM_WARNINGS[type].map((w, i) => (
                    <div
                      key={i}
                      className="rounded-lg border border-amber-200/90 bg-amber-50/70 px-3 py-2.5"
                    >
                      <div className="flex gap-2.5">
                        <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" aria-hidden />
                        <div className="min-w-0">
                          <p className="text-xs font-semibold text-amber-950">{w.title}</p>
                          <p className="text-xs text-amber-950/85 mt-1 leading-relaxed">{w.reason}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}
