import type { ModelResult } from "../api";

/** Short, conservative hints comparing train vs test and CV stability. */
export function regressionFitHints(r: ModelResult): string[] {
  const hints: string[] = [];
  const tr = r.train_r2;
  const te = r.r2;
  const trmse = r.train_rmse;
  const temse = r.rmse;
  if (tr != null && te != null) {
    if (tr - te > 0.12 && tr > 0.25) {
      hints.push("Possible overfit (train R² much higher than test)");
    }
    if (tr < 0.12 && te < 0.12) {
      hints.push("Possible underfit (weak train & test)");
    }
  }
  if (trmse != null && temse != null && temse > 1e-9 && trmse < temse * 0.55) {
    hints.push("Large RMSE gap (train vs test)");
  }
  const cv = r.cv_rmse;
  const cvstd = r.cv_std;
  if (cv != null && cvstd != null && cv > 1e-9 && cvstd / cv >= 0.18) {
    hints.push("High CV σ vs mean (unstable folds)");
  }
  return hints;
}

export function classificationFitHints(r: ModelResult): string[] {
  const hints: string[] = [];
  const tr = r.train_accuracy;
  const te = r.accuracy;
  if (tr != null && te != null) {
    if (tr - te > 0.1 && tr >= 0.72) {
      hints.push("Possible overfit (train acc. much higher than test)");
    }
    if (tr < 0.52 && te < 0.52) {
      hints.push("Possible underfit (weak train & test)");
    }
  }
  const cvstd = r.cv_std;
  if (cvstd != null && cvstd >= 0.07) {
    hints.push("High CV σ (unstable folds)");
  }
  return hints;
}
