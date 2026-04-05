import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  getDiagnostics,
  getExperiments,
  type DiagnosticsPayload,
  type ExperimentRun,
} from "../api";
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
  const [model, setModel] = useState<string>("");
  const [data, setData] = useState<DiagnosticsPayload | null>(null);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loadingDiag, setLoadingDiag] = useState(false);

  useEffect(() => {
    if (!datasetId) return;
    setLoadingRuns(true);
    getExperiments(datasetId)
      .then((d) => {
        setRuns(d.runs);
        const unique = d.runs.filter((r, i, a) => a.findIndex((x) => x.model === r.model) === i);
        setModel((m) => (m && unique.some((r) => r.model === m) ? m : unique[0]?.model ?? ""));
      })
      .catch(() => toast.error("Could not load experiments."))
      .finally(() => setLoadingRuns(false));
  }, [datasetId, experimentsSyncKey]);

  const loadDiag = () => {
    if (!datasetId || !model) return;
    setLoadingDiag(true);
    setData(null);
    getDiagnostics(datasetId, model)
      .then(setData)
      .catch((e: Error) => {
        toast.error(e.message || "Diagnostics failed");
      })
      .finally(() => setLoadingDiag(false));
  };

  useEffect(() => {
    if (model && datasetId) loadDiag();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- loadDiag is stable enough for refetch triggers
  }, [model, datasetId, experimentsSyncKey]);

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

  const uniqueModels = runs.filter((r, i, a) => a.findIndex((x) => x.model === r.model) === i);
  const cm = data?.confusion_matrix;
  const cmMax = cm ? Math.max(...cm.matrix.flat(), 1) : 0;
  const rep = data?.classification_report;
  const perm = data?.permutation_importance ?? [];
  const maxPerm = perm.length ? Math.max(...perm.map((p) => p.mean), 1e-9) : 1;

  return (
    <PageShell
      title="Diagnostics"
      description="Hold-out metrics: confusion matrix, ROC, calibration (binary classification), residuals & learning curve (regression), and permutation importance."
    >
      <div className="mb-6 flex flex-wrap items-end gap-3">
        <div className="min-w-[200px] flex-1">
          <label className="mb-1 block text-xs font-medium text-slate-600">Model</label>
          <Select
            value={model}
            onChange={setModel}
            options={uniqueModels.map((r) => ({
              value: r.model,
              label: r.model.replace(/_/g, " "),
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
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Confusion matrix</CardTitle>
                <p className="text-xs text-slate-500">Rows = actual, columns = predicted</p>
              </CardHeader>
              <CardContent className="overflow-x-auto">
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
              </CardContent>
            </Card>
          )}

          {rep && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Classification report</CardTitle>
              </CardHeader>
              <CardContent className="overflow-x-auto">
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
              </CardContent>
            </Card>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {data.roc_curve && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">ROC curve</CardTitle>
                  <p className="text-xs text-slate-500">Binary classification; positive class = second label</p>
                </CardHeader>
                <CardContent className="flex justify-center">
                  <RocChart {...data.roc_curve} />
                </CardContent>
              </Card>
            )}
            {data.calibration && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Calibration</CardTitle>
                  <p className="text-xs text-slate-500">Predicted probability vs observed frequency</p>
                </CardHeader>
                <CardContent className="flex justify-center">
                  <CalibrationChart {...data.calibration} />
                </CardContent>
              </Card>
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
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Learning curve (accuracy)</CardTitle>
                <p className="text-xs text-slate-500">
                  Training vs 2-fold CV score on growing subsets — gap suggests variance / overfit
                </p>
              </CardHeader>
              <CardContent className="flex justify-center overflow-x-auto">
                <LearningCurveChart {...data.learning_curve} />
              </CardContent>
            </Card>
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
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Residual distribution</CardTitle>
                <p className="text-xs text-slate-500">Test set: actual − predicted</p>
              </CardHeader>
              <CardContent>
                <div className="flex items-end gap-px h-32">
                  {data.residual_histogram.counts.map((c, i) => {
                    const maxC = Math.max(...data.residual_histogram!.counts, 1);
                    return (
                      <div key={i} className="flex-1 flex flex-col items-center gap-1 min-w-0">
                        <div
                          className="w-full rounded-t bg-blue-500/80 min-h-[2px]"
                          style={{ height: `${(c / maxC) * 100}%` }}
                          title={`${data.residual_histogram!.edges[i]?.toFixed(3)} … ${data.residual_histogram!.edges[i + 1]?.toFixed(3)}: ${c}`}
                        />
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {data.learning_curve && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Learning curve (RMSE)</CardTitle>
                <p className="text-xs text-slate-500">
                  Subsampled training data; 2-fold CV — validation above training often means noisy folds or high variance
                </p>
              </CardHeader>
              <CardContent className="flex justify-center overflow-x-auto">
                <LearningCurveChart {...data.learning_curve} />
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {!loadingDiag && perm.length > 0 && (
        <Card className={data ? "mt-6" : ""}>
          <CardHeader>
            <CardTitle className="text-base">Permutation importance</CardTitle>
            <p className="text-xs text-slate-500">
              Drop in hold-out score when each column is shuffled (top {Math.min(40, perm.length)} features)
            </p>
          </CardHeader>
          <CardContent className="space-y-2">
            {perm.slice(0, 25).map((p) => (
              <div key={p.feature}>
                <div className="flex justify-between text-xs mb-0.5">
                  <span className="font-medium text-slate-700 truncate max-w-[60%]">{p.feature}</span>
                  <span className="tabular-nums text-slate-500">
                    {p.mean.toFixed(4)} ± {p.std.toFixed(4)}
                  </span>
                </div>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-violet-500/90"
                    style={{ width: `${Math.min(100, (p.mean / maxPerm) * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </PageShell>
  );
}
