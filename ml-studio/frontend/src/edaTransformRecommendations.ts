/**
 * Heuristic transform suggestions from EDA signals (missing, outliers, skewness, health).
 * Steps match the backend `transforms.apply_transforms` schema.
 */

import type { HealthResult, MissingValues, OutlierInfo, SkewnessRow, TargetAnalysis } from "./api";
import type {
  ClipStep,
  DropDupStep,
  DropStep,
  FixSkewStep,
  FreqEncStep,
  ImputeStep,
  Step,
} from "./transformTypes";
import { makeStep } from "./transformTypes";

export const ML_SPLIT_COL = "__ml_split__";

/** Steps that should run on the full table before assigning train/test labels. */
export type TransformRecommendationPhase = "pre_split" | "post_split";

export interface TransformRecommendation {
  id: string;
  phase: TransformRecommendationPhase;
  title: string;
  description: string;
  step: Step;
}

export function splitRecommendationsByPhase(recs: TransformRecommendation[]): {
  pre: TransformRecommendation[];
  post: TransformRecommendation[];
} {
  return {
    pre: recs.filter((r) => r.phase === "pre_split"),
    post: recs.filter((r) => r.phase === "post_split"),
  };
}

function notSplit(cols: string[]): string[] {
  return cols.filter((c) => c !== ML_SPLIT_COL);
}

export interface BuildRecommendationsInput {
  missing: MissingValues;
  outliers: OutlierInfo;
  skewness: SkewnessRow[];
  health: HealthResult;
  /** When set and numeric skew is notable, target may be added to skew-fix columns. */
  targetAnalysis: TargetAnalysis | null;
}

/**
 * Ordered pipeline-friendly recommendations. Empty if nothing applies.
 */
export function buildTransformRecommendations(input: BuildRecommendationsInput): TransformRecommendation[] {
  const { missing, outliers, skewness, health, targetAnalysis } = input;
  const numericSet = new Set(health.numeric_cols);
  const catSet = new Set(health.cat_cols);
  const out: TransformRecommendation[] = [];

  const constantDrop = notSplit(health.constant_cols ?? []);
  if (constantDrop.length > 0) {
    const step: DropStep = { ...(makeStep("drop_columns") as DropStep), columns: constantDrop };
    out.push({
      id: "drop-constants",
      phase: "pre_split",
      title: "Drop constant columns",
      description: `Remove ${constantDrop.length} column(s) with no variance: ${constantDrop.slice(0, 6).join(", ")}${constantDrop.length > 6 ? "…" : ""}.`,
      step,
    });
  }

  if (health.n_duplicates > 0) {
    const step: DropDupStep = { ...(makeStep("drop_duplicates") as DropDupStep), columns: [], keep: "first" };
    out.push({
      id: "drop-duplicates",
      phase: "pre_split",
      title: "Drop duplicate rows",
      description: `${health.n_duplicates.toLocaleString()} duplicate row(s) detected — keep first occurrence.`,
      step,
    });
  }

  const missingNumeric = notSplit(
    Object.entries(missing)
      .filter(([, v]) => v.count > 0)
      .map(([col]) => col)
      .filter((col) => numericSet.has(col)),
  );
  if (missingNumeric.length > 0) {
    const step: ImputeStep = {
      ...(makeStep("impute") as ImputeStep),
      columns: missingNumeric,
      strategy: "median",
    };
    out.push({
      id: "impute-numeric",
      phase: "post_split",
      title: "Impute missing (numeric)",
      description: `Median imputation for ${missingNumeric.length} column(s): ${missingNumeric.slice(0, 5).join(", ")}${missingNumeric.length > 5 ? "…" : ""}.`,
      step,
    });
  }

  const missingCat = notSplit(
    Object.entries(missing)
      .filter(([, v]) => v.count > 0)
      .map(([col]) => col)
      .filter((col) => catSet.has(col)),
  );
  if (missingCat.length > 0) {
    const step: ImputeStep = {
      ...(makeStep("impute") as ImputeStep),
      columns: missingCat,
      strategy: "mode",
    };
    out.push({
      id: "impute-categorical",
      phase: "post_split",
      title: "Impute missing (categorical)",
      description: `Mode imputation for ${missingCat.length} column(s): ${missingCat.slice(0, 5).join(", ")}${missingCat.length > 5 ? "…" : ""}.`,
      step,
    });
  }

  const outlierCols = notSplit(
    Object.entries(outliers)
      .filter(([, v]) => v.n_outliers > 0)
      .map(([col]) => col)
      .filter((col) => numericSet.has(col)),
  );
  if (outlierCols.length > 0) {
    const step: ClipStep = {
      ...(makeStep("clip_outliers") as ClipStep),
      columns: outlierCols,
      method: "iqr",
    };
    out.push({
      id: "clip-outliers",
      phase: "post_split",
      title: "Clip outliers (IQR)",
      description: `Cap extremes on ${outlierCols.length} numeric column(s) using 1.5×IQR fences.`,
      step,
    });
  }

  const skewCols = notSplit(
    skewness.filter((r) => r.severity !== "normal").map((r) => r.column).filter((c) => numericSet.has(c)),
  );
  const targetCol = targetAnalysis?.target_col;
  const targetSkewOk =
    targetAnalysis?.is_numeric &&
    targetCol &&
    typeof targetAnalysis.skewness === "number" &&
    Math.abs(targetAnalysis.skewness) >= 0.5 &&
    Boolean(targetAnalysis.transform_hint);

  let skewFixCols = [...new Set(skewCols)];
  if (targetSkewOk && targetCol && !constantDrop.includes(targetCol) && numericSet.has(targetCol)) {
    if (!skewFixCols.includes(targetCol)) skewFixCols.push(targetCol);
  }
  skewFixCols = skewFixCols.filter((c) => !constantDrop.includes(c));

  if (skewFixCols.length > 0) {
    const step: FixSkewStep = {
      ...(makeStep("fix_skewness") as FixSkewStep),
      columns: skewFixCols,
      method: "auto",
      threshold: 0.5,
    };
    out.push({
      id: "fix-skewness",
      phase: "post_split",
      title: "Fix skewness (auto)",
      description: `Power / log-style correction for ${skewFixCols.length} column(s) with |skew| above threshold (per-column skip inside the step).${targetSkewOk ? ` Includes target "${targetCol}".` : ""}`,
      step,
    });
  }

  const highCard = notSplit(health.high_card_cols ?? []).filter((c) => !constantDrop.includes(c));
  if (highCard.length > 0) {
    const step: FreqEncStep = {
      ...(makeStep("frequency_encode") as FreqEncStep),
      columns: highCard,
      normalize: true,
    };
    out.push({
      id: "frequency-encode",
      phase: "post_split",
      title: "Frequency-encode high-cardinality columns",
      description: `${highCard.length} categorical column(s) with &gt;50% unique values — adds *_freq columns (consider target encoding on the Train tab for modeling).`,
      step,
    });
  }

  return out;
}
