import { useEffect, useRef, useState, type ReactNode } from "react";
import { toast } from "sonner";
import {
  getDiagnostics,
  getExperiments,
  type DiagnosticsPayload,
  type ExperimentRun,
} from "../api";
import { formatRunLabel } from "../lib/runLabel";
import { ChartExportButtons } from "../components/ChartExportButtons";
import { LoadingState } from "../components/LoadingState";
import { PageShell } from "../components/PageShell";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Select } from "../components/ui/select";

interface Props {
  datasetId: string | null;
  experimentsSyncKey?: number;
}

function exportHref(datasetId: string | null) {
  return datasetId ? `/export?d=${encodeURIComponent(datasetId)}` : "/export";
}

function formatRegressionMetric(key: string, v: number): string {
  if (key === "r2") return v.toFixed(4);
  if (key === "rmse" || key === "mae" || key === "residual_mean" || key === "residual_std") return v.toFixed(4);
  return String(v);
}

function cmColor(v: number, max: number) {
  if (max <= 0) return "rgb(241 245 249)";
  const t = v / max;
  const r = Math.round(219 - t * 100);
  const g = Math.round(234 - t * 40);
  const b = Math.round(254 - t * 54);
  return `rgb(${r} ${g} ${b})`;
}

function RocChart({ fpr, tpr, auc }: { fpr: number[]; tpr: number[]; auc: number }) {
  const W = 320;
  const H = 260;
  const pad = 36;
  const sx = (x: number) => pad + x * (W - pad * 2);
  const sy = (y: number) => H - pad - y * (H - pad * 2);
  const pts = fpr.map((x, i) => `${sx(x)},${sy(tpr[i])}`).join(" ");
  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={W} height={H} className="text-slate-300">
        <rect x={0} y={0} width={W} height={H} fill="#fafafa" rx={8} />
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="#cbd5e1" strokeWidth={1} strokeDasharray="4 3" />
        <polyline fill="none" stroke="#2563eb" strokeWidth={2} points={pts} />
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(0)} stroke="#94a3b8" strokeWidth={1} />
        <line x1={sx(0)} y1={sy(0)} x2={sx(0)} y2={sy(1)} stroke="#94a3b8" strokeWidth={1} />
        <text x={W / 2} y={H - 8} textAnchor="middle" className="fill-slate-500 text-[10px]">
          False positive rate
        </text>
        <text x={12} y={H / 2} className="fill-slate-500 text-[10px]" transform={`rotate(-90 12 ${H / 2})`}>
          True positive rate
        </text>
      </svg>
      <Badge variant="secondary">AUC = {auc.toFixed(4)}</Badge>
    </div>
  );
}

function CalibrationChart({
  mean_predicted,
  fraction_positives,
}: {
  mean_predicted: number[];
  fraction_positives: number[];
}) {
  const W = 300;
  const H = 240;
  const pad = 32;
  const sx = (x: number) => pad + Math.min(1, Math.max(0, x)) * (W - pad * 2);
  const sy = (y: number) => H - pad - Math.min(1, Math.max(0, y)) * (H - pad * 2);
  const diag = `${sx(0)},${sy(0)} ${sx(1)},${sy(1)}`;
  const pts = mean_predicted.map((m, i) => `${sx(m)},${sy(fraction_positives[i])}`).join(" ");
  return (
    <svg width={W} height={H} className="text-slate-300">
      <rect x={0} y={0} width={W} height={H} fill="#fafafa" rx={8} />
      <polyline fill="none" stroke="#cbd5e1" strokeWidth={1} strokeDasharray="4 3" points={diag} />
      <polyline fill="none" stroke="#059669" strokeWidth={2} points={pts} />
      <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(0)} stroke="#94a3b8" strokeWidth={1} />
      <line x1={sx(0)} y1={sy(0)} x2={sx(0)} y2={sy(1)} stroke="#94a3b8" strokeWidth={1} />
    </svg>
  );
}

/** Shared axis scaling so y = x reads as a 45° line on screen. */
function RegressionScatterChart({
  title,
  subtitle,
  xs,
  ys,
  xLabel,
  yLabel,
  yEqualsX,
}: {
  title: string;
  subtitle?: string;
  xs: number[];
  ys: number[];
  xLabel: string;
  yLabel: string;
  yEqualsX?: boolean;
}) {
  const W = 360;
  const H = 300;
  const pad = 44;
  if (!xs.length || xs.length !== ys.length) return null;
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const lo = yEqualsX ? Math.min(minX, minY) : minX;
  const hi = yEqualsX ? Math.max(maxX, maxY) : maxX;
  const loY = yEqualsX ? lo : minY;
  const hiY = yEqualsX ? hi : maxY;
  const spanX = hi - lo || 1;
  const spanY = hiY - loY || 1;
  const sx = (v: number) => pad + ((v - lo) / spanX) * (W - pad * 2);
  const sy = (v: number) => H - pad - ((v - loY) / spanY) * (H - pad * 2);
  const diag =
    yEqualsX ? `${sx(lo)},${sy(lo)} ${sx(hi)},${sy(hi)}` : null;
  return (
    <div className="flex flex-col items-center gap-2 w-full">
      {subtitle && <p className="text-xs text-slate-500 self-start">{subtitle}</p>}
      <svg width={W} height={H} className="text-slate-300 max-w-full">
        <rect x={0} y={0} width={W} height={H} fill="#fafafa" rx={8} />
        {diag && (
          <polyline fill="none" stroke="#94a3b8" strokeWidth={1} strokeDasharray="5 4" points={diag} />
        )}
        <g className="fill-blue-600" fillOpacity={0.35}>
          {xs.map((xv, i) => (
            <circle key={i} cx={sx(xv)} cy={sy(ys[i])} r={2.2} />
          ))}
        </g>
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#94a3b8" strokeWidth={1} />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#94a3b8" strokeWidth={1} />
        <text x={W / 2} y={H - 8} textAnchor="middle" className="fill-slate-500 text-[10px]">
          {xLabel}
        </text>
        <text x={10} y={H / 2} className="fill-slate-500 text-[10px]" transform={`rotate(-90 10 ${H / 2})`}>
          {yLabel}
        </text>
        <text x={pad + 4} y={pad - 6} className="fill-slate-600 text-[10px] font-medium">
          {title}
        </text>
      </svg>
    </div>
  );
}

function ExportableFigureCard({
  title,
  subtitle,
  filenameSlug,
  modelName,
  children,
  contentClassName,
}: {
  title: string;
  subtitle?: ReactNode;
  filenameSlug: string;
  modelName: string;
  children: React.ReactNode;
  contentClassName?: string;
}) {
  const captureRef = useRef<HTMLDivElement>(null);
  const base = `diagnostics_${modelName}_${filenameSlug}`;
  return (
    <Card>
      <CardHeader className="flex flex-col gap-2 space-y-0 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <CardTitle className="text-base">{title}</CardTitle>
          {subtitle ? <div className="text-xs text-slate-500">{subtitle}</div> : null}
        </div>
        <ChartExportButtons targetRef={captureRef} filenameBase={base} className="shrink-0" />
      </CardHeader>
      <CardContent>
        <div ref={captureRef} className={contentClassName ?? "w-full rounded-md bg-white p-2"}>
          {children}
        </div>
      </CardContent>
    </Card>
  );
}

function learningCurveFigureTitle(metric: "accuracy" | "rmse"): string {
  return metric === "rmse" ? "Learning curve (RMSE)" : "Learning curve (accuracy)";
}

function LearningCurveHowToRead({ metric }: { metric: "accuracy" | "rmse" }) {
  const acc = metric === "accuracy";
  return (
    <div className="mt-3 max-w-lg mx-auto text-[11px] text-slate-600 leading-relaxed space-y-1.5 border-t border-slate-100 pt-3">
      <p className="font-semibold text-slate-700">How to read this</p>
      {acc ? (
        <>
          <p>• Both train and validation accuracy stay low → possible <strong>underfitting</strong> (model too simple or weak signal).</p>
          <p>• Train much higher than validation with a gap that does not shrink → <strong>overfitting</strong> / high variance.</p>
          <p>• Curves converge as training size grows → more data may yield limited gains.</p>
        </>
      ) : (
        <>
          <p>• Both train and validation RMSE stay high → possible <strong>underfitting</strong>.</p>
          <p>• Validation RMSE much worse than train with a persistent gap → variance / <strong>overfitting</strong> risk.</p>
          <p>• Curves flatten together at larger sample sizes → diminishing returns from more data.</p>
        </>
      )}
    </div>
  );
}

function LearningCurveChart(payload: NonNullable<DiagnosticsPayload["learning_curve"]>) {
  const { train_sizes, train_score_mean, val_score_mean, metric } = payload;
  const W = 360;
  const H = 220;
  const pad = 40;
  const all = [...train_score_mean, ...val_score_mean];
  const rawMax = all.length ? Math.max(...all) : 0;
  const rawMin = all.length ? Math.min(...all) : 0;
  let minY: number;
  let maxY: number;
  if (metric === "accuracy") {
    minY = Math.max(0, rawMin - 0.03);
    maxY = Math.min(1, rawMax + 0.03);
    if (maxY - minY < 0.05) {
      minY = Math.max(0, rawMin - 0.1);
      maxY = Math.min(1, rawMax + 0.1);
    }
  } else {
    minY = 0;
    maxY = Math.max(rawMax, 1e-9);
  }
  const span = maxY - minY || 1;
  const n = train_sizes.length;
  const sx = (i: number) => pad + (i / Math.max(1, n - 1)) * (W - pad * 2);
  const sy = (v: number) => H - pad - ((v - minY) / span) * (H - pad * 2);
  const ptTrain = train_score_mean.map((v, i) => `${sx(i)},${sy(v)}`).join(" ");
  const ptVal = val_score_mean.map((v, i) => `${sx(i)},${sy(v)}`).join(" ");
  const trainLabel = metric === "accuracy" ? "Train acc." : "Train RMSE";
  const valLabel = metric === "accuracy" ? "CV val acc." : "CV val RMSE";
  const yHint = metric === "accuracy" ? "Accuracy (CV)" : "RMSE (CV)";
  return (
    <svg width={W} height={H} className="text-slate-300">
      <rect x={0} y={0} width={W} height={H} fill="#fafafa" rx={8} />
      <polyline fill="none" stroke="#2563eb" strokeWidth={2} points={ptTrain} />
      <polyline fill="none" stroke="#d97706" strokeWidth={2} points={ptVal} />
      <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#94a3b8" strokeWidth={1} />
      <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#94a3b8" strokeWidth={1} />
      <text x={W / 2} y={H - 10} textAnchor="middle" className="fill-slate-500 text-[9px]">
        Training set size
      </text>
      <text x={pad + 4} y={pad - 8} className="fill-blue-600 text-[9px]">
        {trainLabel}
      </text>
      <text x={pad + 88} y={pad - 8} className="fill-amber-700 text-[9px]">
        {valLabel}
      </text>
      <text x={W - pad} y={pad - 8} textAnchor="end" className="fill-slate-400 text-[8px]">
        {yHint}
      </text>
    </svg>
  );
}

export default function Diagnostics({ datasetId, experimentsSyncKey = 0 }: Props) {
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [data, setData] = useState<DiagnosticsPayload | null>(null);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loadingDiag, setLoadingDiag] = useState(false);

  const selectedRun = runs.find((r) => r.run_id === selectedRunId);
  const model = selectedRun?.model ?? "";

  useEffect(() => {
    if (!datasetId) return;
    setLoadingRuns(true);
    getExperiments(datasetId)
      .then((d) => {
        setRuns(d.runs);
        setSelectedRunId((id) => {
          if (id && d.runs.some((r) => r.run_id === id)) return id;
          return d.runs[0]?.run_id ?? "";
        });
      })
      .catch(() => toast.error("Could not load experiments."))
      .finally(() => setLoadingRuns(false));
  }, [datasetId, experimentsSyncKey]);

  const loadDiag = () => {
    if (!datasetId || !model || !selectedRunId) return;
    setLoadingDiag(true);
    setData(null);
    getDiagnostics(datasetId, model, selectedRunId)
      .then(setData)
      .catch((e: Error) => {
        toast.error(e.message || "Diagnostics failed");
      })
      .finally(() => setLoadingDiag(false));
  };

  useEffect(() => {
    if (model && selectedRunId && datasetId) loadDiag();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- loadDiag is stable enough for refetch triggers
  }, [model, selectedRunId, datasetId, experimentsSyncKey]);

  if (!datasetId) {
    return (
      <PageShell title="Diagnostics">
        <p className="text-sm text-slate-500">Upload a dataset and train a model first.</p>
      </PageShell>
    );
  }

  if (loadingRuns) {
    return (
      <PageShell title="Diagnostics" description="Evaluation plots and metrics on the hold-out set.">
        <LoadingState variant="page" message="Loading models…" />
      </PageShell>
    );
  }

  if (!runs.length) {
    return (
      <PageShell title="Diagnostics" description="Evaluation plots and metrics on the hold-out set.">
        <p className="text-sm text-slate-500">No runs yet — train at least one model on the Train tab.</p>
      </PageShell>
    );
  }

  const cm = data?.confusion_matrix;
  const cmMax = cm ? Math.max(...cm.matrix.flat(), 1) : 0;
  const rep = data?.classification_report;
  const perm = data?.permutation_importance ?? [];
  const maxPerm = perm.length ? Math.max(...perm.map((p) => p.mean), 1e-9) : 1;

  return (
    <PageShell
      title="Diagnostics"
      description="Hold-out metrics with PNG/PDF export per figure. Quick pipeline download below — full joblib + load snippet on Export."
    >
      <div className="mb-6 flex flex-wrap items-end gap-3">
        <div className="min-w-[240px] flex-1 max-w-md">
          <label className="mb-1 block text-xs font-medium text-slate-600">Run</label>
          <Select
            value={selectedRunId}
            onChange={setSelectedRunId}
            options={runs.map((r) => ({
              value: r.run_id,
              label: formatRunLabel(r),
            }))}
          />
        </div>
        <Button type="button" variant="secondary" onClick={loadDiag} disabled={loadingDiag || !model}>
          {loadingDiag ? "Computing…" : "Refresh"}
        </Button>
      </div>

      {loadingDiag && <LoadingState variant="page" message="Computing diagnostics (may take a few seconds)…" />}

      {!loadingDiag && data?.task_type === "classification" && (
        <div className="space-y-6">
          {cm && (
            <ExportableFigureCard
              title="Confusion matrix"
              subtitle="Rows = actual, columns = predicted"
              filenameSlug="confusion_matrix"
              modelName={model}
              contentClassName="w-full overflow-x-auto rounded-md bg-white p-2"
            >
              <table className="border-collapse text-sm">
                <thead>
                  <tr>
                    <th className="p-2 border border-slate-200 bg-slate-50" />
                    {cm.labels.map((l) => (
                      <th key={l} className="p-2 border border-slate-200 bg-slate-50 font-medium text-slate-700">
                        {l}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {cm.matrix.map((row, ri) => (
                    <tr key={ri}>
                      <td className="p-2 border border-slate-200 bg-slate-50 font-medium text-slate-700">
                        {cm.labels[ri]}
                      </td>
                      {row.map((cell, ci) => (
                        <td
                          key={ci}
                          className="p-3 border border-slate-200 text-center font-semibold tabular-nums"
                          style={{ backgroundColor: cmColor(cell, cmMax) }}
                        >
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </ExportableFigureCard>
          )}

          {rep && (
            <ExportableFigureCard
              title="Classification report"
              filenameSlug="classification_report"
              modelName={model}
              contentClassName="w-full overflow-x-auto rounded-md bg-white p-2"
            >
              <table className="w-full text-xs border border-slate-200 rounded-lg overflow-hidden">
                <thead>
                  <tr className="bg-slate-50 text-slate-600">
                    <th className="text-left p-2 font-semibold">Class / avg</th>
                    <th className="p-2 font-semibold">Precision</th>
                    <th className="p-2 font-semibold">Recall</th>
                    <th className="p-2 font-semibold">F1</th>
                    <th className="p-2 font-semibold">Support</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(rep).map(([k, v]) => {
                    if (typeof v !== "object" || v === null) return null;
                    const row = v as Record<string, number>;
                    return (
                      <tr key={k} className="border-t border-slate-100">
                        <td className="p-2 font-medium text-slate-800">{k}</td>
                        <td className="p-2 tabular-nums text-slate-600">{row.precision?.toFixed(3) ?? "—"}</td>
                        <td className="p-2 tabular-nums text-slate-600">{row.recall?.toFixed(3) ?? "—"}</td>
                        <td className="p-2 tabular-nums text-slate-600">{row["f1-score"]?.toFixed(3) ?? "—"}</td>
                        <td className="p-2 tabular-nums text-slate-500">{row.support ?? "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </ExportableFigureCard>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {data.roc_curve && (
              <ExportableFigureCard
                title="ROC curve"
                subtitle="Binary classification; positive class = second label"
                filenameSlug="roc_curve"
                modelName={model}
                contentClassName="flex w-full justify-center rounded-md bg-white p-2"
              >
                <RocChart {...data.roc_curve} />
              </ExportableFigureCard>
            )}
            {data.calibration && (
              <ExportableFigureCard
                title="Calibration"
                subtitle="Predicted probability vs observed frequency"
                filenameSlug="calibration"
                modelName={model}
                contentClassName="flex w-full justify-center rounded-md bg-white p-2"
              >
                <CalibrationChart {...data.calibration} />
              </ExportableFigureCard>
            )}
            {data.roc_auc_macro_ovr != null && !data.roc_curve && (
              <Card>
                <CardContent className="pt-6">
                  <p className="text-sm text-slate-600">
                    Multiclass ROC AUC (one-vs-rest, macro):{" "}
                    <strong className="tabular-nums">{data.roc_auc_macro_ovr.toFixed(4)}</strong>
                  </p>
                </CardContent>
              </Card>
            )}
          </div>

          {data.learning_curve && (
            <ExportableFigureCard
              title="Learning curve (accuracy)"
              subtitle="Training vs 2-fold CV score on growing subsets — gap suggests variance / overfit"
              filenameSlug="learning_curve_accuracy"
              modelName={model}
              contentClassName="flex w-full justify-center overflow-x-auto rounded-md bg-white p-2"
            >
              <LearningCurveChart {...data.learning_curve} />
            </ExportableFigureCard>
          )}
        </div>
      )}

      {!loadingDiag && data?.task_type === "regression" && data.regression && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {Object.entries(data.regression).map(([k, v]) => (
              <Card key={k}>
                <CardContent className="pt-4 pb-3">
                  <p className="text-[10px] uppercase tracking-wide text-slate-500">{k.replace(/_/g, " ")}</p>
                  <p className="text-lg font-bold tabular-nums text-slate-900">{formatRegressionMetric(k, v)}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          {data.residual_histogram && (
            <ExportableFigureCard
              title="Residual distribution"
              subtitle="Test set: actual − predicted"
              filenameSlug="residual_histogram"
              modelName={model}
              contentClassName="w-full rounded-md bg-white p-2"
            >
              <div className="flex h-32 items-end gap-px">
                {data.residual_histogram.counts.map((c, i) => {
                  const maxC = Math.max(...data.residual_histogram!.counts, 1);
                  return (
                    <div key={i} className="flex min-w-0 flex-1 flex-col items-center gap-1">
                      <div
                        className="min-h-[2px] w-full rounded-t bg-blue-500/80"
                        style={{ height: `${(c / maxC) * 100}%` }}
                        title={`${data.residual_histogram!.edges[i]?.toFixed(3)} … ${data.residual_histogram!.edges[i + 1]?.toFixed(3)}: ${c}`}
                      />
                    </div>
                  );
                })}
              </div>
            </ExportableFigureCard>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {data.predicted_vs_actual && (
              <ExportableFigureCard
                title="Predicted vs actual"
                subtitle="Hold-out subsample; dashed line is y = x"
                filenameSlug="predicted_vs_actual"
                modelName={model}
                contentClassName="flex w-full justify-center overflow-x-auto rounded-md bg-white p-2"
              >
                <RegressionScatterChart
                  title="Fit"
                  xs={data.predicted_vs_actual.predicted}
                  ys={data.predicted_vs_actual.actual}
                  xLabel="Predicted"
                  yLabel="Actual"
                  yEqualsX
                />
              </ExportableFigureCard>
            )}
            {data.residuals_vs_predicted && (
              <ExportableFigureCard
                title="Residuals vs predicted"
                subtitle="Look for funnel shape or curvature"
                filenameSlug="residuals_vs_predicted"
                modelName={model}
                contentClassName="flex w-full justify-center overflow-x-auto rounded-md bg-white p-2"
              >
                <RegressionScatterChart
                  title="Residual check"
                  xs={data.residuals_vs_predicted.predicted}
                  ys={data.residuals_vs_predicted.residuals}
                  xLabel="Predicted"
                  yLabel="Residual"
                />
              </ExportableFigureCard>
            )}
          </div>

          {data.scaler_baseline_compare && (
            <ExportableFigureCard
              title="Linear regression × scaler (baseline)"
              subtitle={
                <>
                  Same numeric features as your pipeline; independent of the selected model. Best by RMSE:{" "}
                  <span className="font-semibold text-slate-700">{data.scaler_baseline_compare.best_scaler}</span>
                </>
              }
              filenameSlug="scaler_baseline_compare"
              modelName={model}
              contentClassName="w-full overflow-x-auto rounded-md bg-white p-2"
            >
              <table className="w-full text-sm border border-slate-200 rounded-lg overflow-hidden">
                <thead>
                  <tr className="bg-slate-50 text-slate-600">
                    <th className="text-left p-2 font-semibold">Scaler</th>
                    <th className="text-right p-2 font-semibold">RMSE</th>
                    <th className="text-right p-2 font-semibold">R²</th>
                  </tr>
                </thead>
                <tbody>
                  {data.scaler_baseline_compare.comparisons.map((row) => (
                    <tr
                      key={row.scaler}
                      className={
                        row.scaler === data.scaler_baseline_compare!.best_scaler
                          ? "border-t border-slate-100 bg-blue-50/80"
                          : "border-t border-slate-100"
                      }
                    >
                      <td className="p-2 font-medium text-slate-800">{row.scaler}</td>
                      <td className="p-2 text-right tabular-nums text-slate-700">{row.rmse.toFixed(4)}</td>
                      <td className="p-2 text-right tabular-nums text-slate-700">{row.r2.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ExportableFigureCard>
          )}

          {data.learning_curve && (
            <ExportableFigureCard
              title={learningCurveFigureTitle(data.learning_curve.metric)}
              subtitle="Subsampled training data; 2-fold CV — validation above training often means noisy folds or high variance"
              filenameSlug={data.learning_curve.metric === "rmse" ? "learning_curve_rmse" : "learning_curve_accuracy"}
              modelName={model}
              contentClassName="flex w-full flex-col justify-center overflow-x-auto rounded-md bg-white p-2"
            >
              <LearningCurveChart {...data.learning_curve} />
              <LearningCurveHowToRead metric={data.learning_curve.metric} />
            </ExportableFigureCard>
          )}
        </div>
      )}

      {!loadingDiag && perm.length > 0 && (
        <div className={data ? "mt-6" : ""}>
          <ExportableFigureCard
            title="Permutation importance"
            subtitle={`Drop in hold-out score when each column is shuffled (top ${Math.min(40, perm.length)} features)`}
            filenameSlug="permutation_importance"
            modelName={model}
            contentClassName="w-full space-y-2 rounded-md bg-white p-2"
          >
            {perm.slice(0, 25).map((p) => (
              <div key={p.feature}>
                <div className="mb-0.5 flex justify-between text-xs">
                  <span className="max-w-[60%] truncate font-medium text-slate-700">{p.feature}</span>
                  <span className="tabular-nums text-slate-500">
                    {p.mean.toFixed(4)} ± {p.std.toFixed(4)}
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-full rounded-full bg-violet-500/90"
                    style={{ width: `${Math.min(100, (p.mean / maxPerm) * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </ExportableFigureCard>
        </div>
      )}
    </PageShell>
  );
}
