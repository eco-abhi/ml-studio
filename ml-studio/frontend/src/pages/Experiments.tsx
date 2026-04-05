import React, { useEffect, useState } from "react";
import { getExperiments, getMLflowInfo, type ExperimentRun, type MLflowInfo, type TaskType } from "../api";
import { formatRunLabel, shortRunId } from "../lib/runLabel";
import { LoadingState } from "../components/LoadingState";
import { Select } from "../components/ui/select";
import { PageShell } from "../components/PageShell";
import { Badge } from "../components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";

type MetricKey =
  | "rmse" | "mae" | "r2" | "cv_rmse"
  | "train_rmse" | "train_mae" | "train_r2"
  | "accuracy" | "f1_score" | "roc_auc"
  | "train_accuracy" | "train_f1";

const META = {
  regression: {
    primary: "rmse" as MetricKey,
    cols: ["model", "train_r2", "r2", "train_rmse", "rmse", "mae", "cv_rmse"] as const,
    lowerBetter: {
      rmse: true,
      mae: true,
      cv_rmse: true,
      train_rmse: true,
      train_mae: true,
      r2: false,
      train_r2: false,
    } as Record<string, boolean>,
    labels: {
      model: "Model / run",
      rmse: "RMSE",
      mae: "MAE",
      r2: "R²",
      cv_rmse: "CV RMSE",
      train_rmse: "Train RMSE",
      train_mae: "Train MAE",
      train_r2: "Train R²",
    } as Record<string, string>,
  },
  classification: {
    primary: "accuracy" as MetricKey,
    cols: ["model", "train_accuracy", "accuracy", "train_f1", "f1_score", "roc_auc"] as const,
    lowerBetter: {
      accuracy: false,
      f1_score: false,
      roc_auc: false,
      train_accuracy: false,
      train_f1: false,
    } as Record<string, boolean>,
    labels: {
      model: "Model / run",
      accuracy: "Accuracy",
      f1_score: "F1",
      roc_auc: "ROC-AUC",
      train_accuracy: "Train acc.",
      train_f1: "Train F1",
    } as Record<string, string>,
  },
};

const COLORS = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#ec4899","#84cc16"];
const ALL_METRICS: MetricKey[] = [
  "rmse", "mae", "r2", "cv_rmse", "train_rmse", "train_mae", "train_r2",
  "accuracy", "f1_score", "roc_auc", "train_accuracy", "train_f1",
];

function getBest(runs: ExperimentRun[], metric: MetricKey, lower: boolean) {
  const valid = runs.filter((r) => r[metric] !== undefined);
  if (!valid.length) return null;
  return lower
    ? valid.reduce((a, b) => (a[metric]! < b[metric]! ? a : b)).run_id
    : valid.reduce((a, b) => (a[metric]! > b[metric]! ? a : b)).run_id;
}

function fmtTs(ms: number | undefined) {
  if (!ms) return "—";
  return new Date(ms).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function MlflowLocalHint({ info }: { info: MLflowInfo | null }) {
  const [copied, setCopied] = useState<"full" | "uv" | null>(null);
  if (!info || info.is_remote || !info.local_ui_command) return null;
  const cmd = info.local_ui_command;
  const cmdUv = info.local_ui_command_uv;
  const copy = (which: "full" | "uv", text: string) => {
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(which);
      window.setTimeout(() => setCopied(null), 2000);
    });
  };
  return (
    <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs text-amber-950">
      <p className="font-semibold text-amber-900">MLflow UI (local tracking)</p>
      {info.local_tracking_note && (
        <p className="mt-2 text-amber-900/95 leading-relaxed border border-amber-300/60 rounded-lg bg-white/60 px-2.5 py-2">
          {info.local_tracking_note}
        </p>
      )}
      <p className="mt-2 text-amber-800/90 leading-relaxed">
        Runs are stored at{" "}
        <code className="rounded bg-white/90 px-1.5 py-0.5 text-[11px] text-amber-950 break-all">{info.tracking_uri}</code>.
        Start the UI with the same store as the API, then open{" "}
        <a href={info.ui_url} className="font-medium text-blue-700 underline" target="_blank" rel="noopener noreferrer">
          {info.ui_url}
        </a>
        .
      </p>
      {cmdUv && (
        <div className="mt-3">
          <p className="text-amber-900 font-medium mb-1">Recommended (from <code className="text-[11px]">backend/</code>)</p>
          <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
            <pre className="min-w-0 flex-1 overflow-x-auto rounded-lg bg-slate-900 p-2.5 font-mono text-[11px] leading-relaxed text-slate-100 whitespace-pre-wrap sm:whitespace-pre">
              {cmdUv}
            </pre>
            <button
              type="button"
              onClick={() => copy("uv", cmdUv)}
              className="shrink-0 rounded-lg border border-amber-300 bg-white px-3 py-2 text-xs font-medium text-amber-900 transition-colors hover:bg-amber-100"
            >
              {copied === "uv" ? "Copied" : "Copy"}
            </button>
          </div>
        </div>
      )}
      <div className="mt-3">
        <p className="text-amber-900/80 mb-1">Full command (any environment)</p>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
          <pre className="min-w-0 flex-1 overflow-x-auto rounded-lg bg-slate-900 p-2.5 font-mono text-[11px] leading-relaxed text-slate-100 whitespace-pre-wrap sm:whitespace-pre">
            {cmd}
          </pre>
          <button
            type="button"
            onClick={() => copy("full", cmd)}
            className="shrink-0 rounded-lg border border-amber-300 bg-white px-3 py-2 text-xs font-medium text-amber-900 transition-colors hover:bg-amber-100"
          >
            {copied === "full" ? "Copied" : "Copy"}
          </button>
        </div>
      </div>
    </div>
  );
}

interface Props {
  datasetId: string | null;
  /** Increment after training completes to refetch runs without a full page reload. */
  experimentsSyncKey?: number;
}

export default function Experiments({ datasetId, experimentsSyncKey = 0 }: Props) {
  const [runs, setRuns]           = useState<ExperimentRun[]>([]);
  const [loading, setLoading]     = useState(true);
  const [taskType, setTaskType]   = useState<TaskType | null>(null);
  const [sortCol, setSortCol]     = useState<MetricKey>("rmse");
  const [sortDir, setSortDir]     = useState<"asc" | "desc">("asc");
  const [chartMetric, setChartMetric] = useState<MetricKey>("rmse");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [mlflowInfo, setMlflowInfo]       = useState<MLflowInfo | null>(null);

  useEffect(() => {
    getMLflowInfo().then(setMlflowInfo).catch(() => null);
  }, []);

  useEffect(() => {
    if (!datasetId) { setLoading(false); return; }
    setLoading(true);
    getExperiments(datasetId)
      .then((d) => {
        setRuns(d.runs);
        if (d.runs.length) {
          const t: TaskType = "accuracy" in d.runs[0] ? "classification" : "regression";
          setTaskType(t);
          const p = META[t].primary;
          setSortCol(p); setChartMetric(p);
          setSelectedRunId(d.runs[0].run_id);
        }
      })
      .finally(() => setLoading(false));
  }, [datasetId, experimentsSyncKey]);

  const mlflowLink = mlflowInfo && (
    <a
      href={mlflowInfo.ui_url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 hover:border-slate-300 text-xs font-medium text-slate-600 transition-colors"
    >
      <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth={2}>
        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
        <polyline points="15 3 21 3 21 9" />
        <line x1="10" y1="14" x2="21" y2="3" />
      </svg>
      {mlflowInfo.is_remote ? "Open DagShub" : "Open MLflow UI"}
    </a>
  );

  if (!datasetId) return <Empty title="Experiments" mlflowLink={mlflowLink} mlflowInfo={mlflowInfo} />;

  if (loading) return (
    <PageShell title="Experiments">
      <MlflowLocalHint info={mlflowInfo} />
      <LoadingState variant="page" message="Loading experiment runs…" />
    </PageShell>
  );

  if (!runs.length) {
    return (
      <Empty title="Experiments" msg="No runs yet — train models first." mlflowLink={mlflowLink} mlflowInfo={mlflowInfo} />
    );
  }

  const meta = META[taskType ?? "regression"];
  const cols = meta.cols.filter((c) => c === "model" || runs.some((r) => r[c as MetricKey] !== undefined));
  const metricCols = cols.filter((c) => c !== "model") as MetricKey[];

  const toggleSort = (col: string) => {
    if (col === "model") return;
    const mc = col as MetricKey;
    if (sortCol === mc) setSortDir((d) => d === "asc" ? "desc" : "asc");
    else { setSortCol(mc); setSortDir(meta.lowerBetter[mc] ? "asc" : "desc"); }
  };

  const sorted = [...runs].sort((a, b) => {
    const av = (a[sortCol] as number) ?? (sortDir === "asc" ? Infinity : -Infinity);
    const bv = (b[sortCol] as number) ?? (sortDir === "asc" ? Infinity : -Infinity);
    return sortDir === "asc" ? av - bv : bv - av;
  });

  const bestRunId = getBest(runs, meta.primary, meta.lowerBetter[meta.primary]);
  const champRun = runs.find((r) => r.run_id === bestRunId) ?? null;
  const color = (runId: string) => COLORS[runs.findIndex((r) => r.run_id === runId) % COLORS.length];

  const chartVals = runs.filter((r) => r[chartMetric] !== undefined)
    .map((r) => ({ run_id: r.run_id, model: r.model, value: r[chartMetric] as number }));
  const maxVal = Math.max(...chartVals.map((v) => v.value), 0.0001);
  const lb = meta.lowerBetter[chartMetric] ?? false;

  const selectedRun = runs.find((r) => r.run_id === selectedRunId) ?? null;

  return (
    <PageShell
      title="Experiments"
      description={`${taskType} · ${runs.length} run${runs.length !== 1 ? "s" : ""}`}
      action={
        <div className="flex items-center gap-2">
          {mlflowLink}
          {champRun && (
            <Badge variant="success" className="max-w-[min(100vw,20rem)] truncate" title={formatRunLabel(champRun)}>
              ★ {formatRunLabel(champRun)}
            </Badge>
          )}
        </div>
      }
    >
      <MlflowLocalHint info={mlflowInfo} />
      <Tabs defaultValue="overview">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="runs">Run Explorer</TabsTrigger>
        </TabsList>

        {/* ── Overview tab ── */}
        <TabsContent value="overview">
          <div className="space-y-5 mt-1">
            {/* Metric bar chart */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-3">
                  <CardTitle>Model comparison</CardTitle>
                  <Select
                    value={chartMetric}
                    onChange={(v) => setChartMetric(v as MetricKey)}
                    options={metricCols.map((c) => ({ value: c, label: meta.labels[c] ?? c }))}
                    size="sm"
                    className="w-32"
                  />
                  <span className="text-xs text-slate-400">{lb ? "lower is better" : "higher is better"}</span>
                </div>
              </CardHeader>
              <CardContent className="space-y-2.5">
                {[...chartVals].sort((a, b) => lb ? a.value - b.value : b.value - a.value).map(({ run_id, model, value }) => {
                  const pct = (value / maxVal) * 100;
                  const isChamp = run_id === bestRunId && chartMetric === meta.primary;
                  const run = runs.find((r) => r.run_id === run_id);
                  return (
                    <div key={run_id} className="flex items-center gap-3">
                      <div className={`w-52 shrink-0 min-w-0 ${isChamp ? "font-semibold text-emerald-700" : "text-slate-600"}`}>
                        <div className="text-sm truncate">
                          {isChamp && <span className="mr-1">★</span>}
                          {model.replace(/_/g, " ")}
                        </div>
                        {run && (
                          <div className="text-[10px] text-slate-400 font-mono truncate font-normal" title={run.run_id}>
                            …{shortRunId(run.run_id)}
                            {run.started_at != null
                              ? ` · ${new Date(run.started_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`
                              : ""}
                          </div>
                        )}
                      </div>
                      <div className="flex-1 h-6 bg-slate-100 rounded overflow-hidden">
                        <div
                          className="h-full rounded transition-all duration-500"
                          style={{ width: `${pct}%`, background: color(run_id), opacity: isChamp ? 1 : 0.75 }}
                        />
                      </div>
                      <span className="w-16 text-right text-sm tabular-nums font-medium text-slate-700">
                        {value.toFixed(4)}
                      </span>
                    </div>
                  );
                })}
              </CardContent>
            </Card>

            {/* Sortable table */}
            <Card>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50">
                        {cols.map((col) => (
                          <th
                            key={col}
                            onClick={() => toggleSort(col as string)}
                            className={`px-4 py-3 text-left font-medium text-slate-600 whitespace-nowrap select-none
                              ${col !== "model" ? "cursor-pointer hover:text-blue-600" : ""}
                              ${sortCol === col ? "text-blue-600" : ""}`}
                          >
                            {meta.labels[col as string] ?? (col as string).replace(/_/g, " ").toUpperCase()}
                            {sortCol === (col as MetricKey) && col !== "model" && (
                              <span className="ml-1">{sortDir === "asc" ? "↑" : "↓"}</span>
                            )}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {sorted.map((run, idx) => {
                        const isChamp = run.run_id === bestRunId;
                        return (
                          <tr key={run.run_id} className={`border-b border-slate-100 last:border-0
                            ${isChamp ? "bg-emerald-50 border-l-4 border-l-emerald-500" : idx % 2 === 0 ? "bg-white" : "bg-slate-50/50"}`}>
                            {cols.map((col) => (
                              <td key={col as string} className="px-4 py-3">
                                {col === "model" ? (
                                  <div className="flex items-center gap-2 min-w-0">
                                    <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: color(run.run_id) }} />
                                    <div className="min-w-0">
                                      <div className={`truncate ${isChamp ? "font-semibold text-slate-900" : "text-slate-700"}`}>
                                        {run.model.replace(/_/g, " ")}
                                      </div>
                                      <div className="text-[10px] text-slate-400 font-mono truncate" title={run.run_id}>
                                        …{shortRunId(run.run_id)}
                                        {run.started_at != null
                                          ? ` · ${new Date(run.started_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`
                                          : ""}
                                      </div>
                                    </div>
                                    {isChamp && <Badge variant="success" className="text-[10px] py-0 px-1.5 shrink-0">best</Badge>}
                                  </div>
                                ) : run[col as MetricKey] !== undefined ? (
                                  <span className={`tabular-nums ${isChamp && col === meta.primary ? "font-bold" : ""}`}>
                                    {(run[col as MetricKey] as number).toFixed(4)}
                                  </span>
                                ) : <span className="text-slate-300">—</span>}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            {/* Summary KPI cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {metricCols.map((mc) => {
                const vals = runs.filter((r) => r[mc] !== undefined).map((r) => r[mc] as number);
                if (!vals.length) return null;
                const lb2 = meta.lowerBetter[mc] ?? false;
                const best = lb2 ? Math.min(...vals) : Math.max(...vals);
                const worst = lb2 ? Math.max(...vals) : Math.min(...vals);
                return (
                  <Card key={mc} className="p-4">
                    <p className="text-xs text-slate-500 font-medium">{meta.labels[mc] ?? mc}</p>
                    <p className="text-xl font-bold text-slate-900 mt-1 tabular-nums">{best.toFixed(4)}</p>
                    <p className="text-[10px] text-slate-400 mt-0.5">best · worst: {worst.toFixed(4)}</p>
                  </Card>
                );
              })}
            </div>
          </div>
        </TabsContent>

        {/* ── Run Explorer tab ── */}
        <TabsContent value="runs">
          <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-5 mt-1">

            {/* Left — run list */}
            <div className="space-y-1.5">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide px-1 mb-2">
                {runs.length} run{runs.length !== 1 ? "s" : ""}
              </p>
              {runs.map((run, idx) => {
                const isChamp = run.run_id === bestRunId;
                const isSelected = run.run_id === selectedRunId;
                const primVal = run[meta.primary] as number | undefined;
                return (
                  <button
                    key={run.run_id}
                    onClick={() => setSelectedRunId(run.run_id)}
                    className={`w-full text-left px-3 py-2.5 rounded-lg border transition-all ${
                      isSelected
                        ? "border-blue-400 bg-blue-50 shadow-sm"
                        : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <span className="w-2 h-2 rounded-full shrink-0 mt-1.5" style={{ background: color(run.run_id) }} />
                      <div className="min-w-0 flex-1">
                        <div className={`text-sm font-medium leading-snug ${isSelected ? "text-blue-800" : "text-slate-700"}`}>
                          {run.model.replace(/_/g, " ")}
                        </div>
                        <div className="text-[10px] text-slate-400 font-mono truncate mt-0.5" title={run.run_id}>
                          …{shortRunId(run.run_id)}
                        </div>
                      </div>
                      {isChamp && <Badge variant="success" className="text-[10px] py-0 px-1.5 shrink-0">best</Badge>}
                    </div>
                    <div className="flex items-center justify-between mt-1 pl-4">
                      <span className="text-[11px] text-slate-400 font-mono">#{idx + 1}</span>
                      {primVal !== undefined && (
                        <span className="text-[11px] tabular-nums text-slate-500">
                          {meta.labels[meta.primary]}: {primVal.toFixed(4)}
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Right — run detail panel */}
            {selectedRun ? (
              <div className="space-y-4">
                {/* Header */}
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="text-lg font-semibold text-slate-900 leading-snug">
                      {formatRunLabel(selectedRun)}
                    </h3>
                    <p className="text-xs text-slate-400 font-mono mt-0.5 break-all">{selectedRun.run_id}</p>
                  </div>
                  <div className="flex gap-2 shrink-0">
                    <Badge variant={selectedRun.status === "FINISHED" ? "success" : "secondary"}>
                      {selectedRun.status}
                    </Badge>
                    {selectedRun.run_id === bestRunId && <Badge variant="success">★ best run</Badge>}
                  </div>
                </div>

                {selectedRun.started_at && (
                  <p className="text-xs text-slate-400">Started: {fmtTs(selectedRun.started_at)}</p>
                )}

                {/* Metrics vs best */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Metrics</CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="space-y-3">
                      {ALL_METRICS.filter((m) => selectedRun[m] !== undefined).map((m) => {
                        const val = selectedRun[m] as number;
                        const bestRun = runs.find((r) => r.run_id === bestRunId);
                        const bestVal = bestRun?.[m] as number | undefined;
                        const lbm = meta.lowerBetter[m] ?? false;
                        const isBestMetric = bestVal !== undefined && (
                          lbm ? val <= bestVal : val >= bestVal
                        );
                        // bar: scale within range of all runs for this metric
                        const allVals = runs.filter(r => r[m] !== undefined).map(r => r[m] as number);
                        const minV = Math.min(...allVals), maxV = Math.max(...allVals);
                        const range = maxV - minV || 1;
                        const pct = ((val - minV) / range) * 100;

                        return (
                          <div key={m}>
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs font-medium text-slate-600">
                                {meta.labels[m] ?? m.toUpperCase()}
                              </span>
                              <div className="flex items-center gap-2">
                                {isBestMetric && <span className="text-[10px] text-emerald-600 font-semibold">★ best</span>}
                                <span className="text-sm tabular-nums font-bold text-slate-900">
                                  {val.toFixed(4)}
                                </span>
                              </div>
                            </div>
                            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full transition-all duration-500"
                                style={{
                                  width: `${lbm ? 100 - pct : pct}%`,
                                  background: isBestMetric ? "#10b981" : color(selectedRun.run_id),
                                }}
                              />
                            </div>
                            {bestVal !== undefined && !isBestMetric && (
                              <p className="text-[10px] text-slate-400 mt-0.5 text-right">
                                best: {bestVal.toFixed(4)} · delta: {(val - bestVal > 0 ? "+" : "")}{(val - bestVal).toFixed(4)}
                              </p>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>

                {/* Hyperparams */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Hyperparameters</CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    {selectedRun.params && Object.keys(selectedRun.params).length > 0 ? (
                      <div className="space-y-0">
                        {Object.entries(selectedRun.params).map(([k, v]) => {
                          // Compare vs best run
                          const bestRun = runs.find((r) => r.run_id === bestRunId);
                          const bestVal = bestRun?.params?.[k];
                          const diff = bestVal !== undefined && typeof v === "number" && typeof bestVal === "number"
                            ? v - (bestVal as number)
                            : null;
                          return (
                            <div key={k} className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0">
                              <span className="text-xs text-slate-500 font-mono">{k}</span>
                              <div className="flex items-center gap-2">
                                {diff !== null && Math.abs(diff) > 1e-9 && (
                                  <span className={`text-[10px] font-mono ${diff > 0 ? "text-amber-500" : "text-blue-500"}`}>
                                    {diff > 0 ? "+" : ""}{diff.toFixed(4).replace(/\.?0+$/, "")}
                                  </span>
                                )}
                                <span className="text-xs font-semibold text-slate-900 tabular-nums font-mono">
                                  {typeof v === "number"
                                    ? v.toFixed(6).replace(/\.?0+$/, "")
                                    : String(v)}
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="text-xs text-slate-400">No hyperparameters logged for this run.</p>
                    )}
                  </CardContent>
                </Card>

                {/* Cross-run param comparison (if multiple runs of same model) */}
                {(() => {
                  const sameModel = runs.filter((r) => r.model === selectedRun.model && r.run_id !== selectedRun.run_id);
                  if (!sameModel.length || !selectedRun.params || !Object.keys(selectedRun.params).length) return null;
                  const paramKeys = Object.keys(selectedRun.params).filter((k) => typeof selectedRun.params![k] === "number");
                  if (!paramKeys.length) return null;
                  return (
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm">
                          vs other {selectedRun.model.replace(/_/g, " ")} runs
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="pt-0 overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-slate-200">
                              <th className="text-left py-1.5 pr-3 text-slate-500 font-medium">Param</th>
                              <th className="text-right py-1.5 px-2 text-blue-600 font-semibold">This run</th>
                              {sameModel.slice(0, 3).map((r, i) => (
                                <th key={r.run_id} className="text-right py-1.5 px-2 text-slate-400 font-medium">
                                  Run {i + 2}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {paramKeys.map((k) => (
                              <tr key={k} className="border-b border-slate-100 last:border-0">
                                <td className="py-1.5 pr-3 font-mono text-slate-500">{k}</td>
                                <td className="text-right py-1.5 px-2 font-semibold text-slate-900 tabular-nums font-mono">
                                  {String(selectedRun.params![k]).slice(0, 10)}
                                </td>
                                {sameModel.slice(0, 3).map((r) => (
                                  <td key={r.run_id} className="text-right py-1.5 px-2 text-slate-400 tabular-nums font-mono">
                                    {r.params?.[k] !== undefined ? String(r.params[k]).slice(0, 10) : "—"}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </CardContent>
                    </Card>
                  );
                })()}
              </div>
            ) : (
              <div className="flex items-center justify-center h-48 text-sm text-slate-400">
                Select a run on the left to see its details.
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </PageShell>
  );
}

function Empty({ title, msg = "Upload and train a dataset first.", mlflowLink, mlflowInfo }: {
  title: string; msg?: string; mlflowLink?: React.ReactNode; mlflowInfo?: MLflowInfo | null;
}) {
  return (
    <PageShell title={title} action={mlflowLink}>
      <MlflowLocalHint info={mlflowInfo ?? null} />
      <p className="text-sm text-slate-500">{msg}</p>
    </PageShell>
  );
}
