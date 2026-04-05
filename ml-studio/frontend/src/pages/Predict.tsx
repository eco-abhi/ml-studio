import { Loader2, RotateCcw, Sparkles, ZapIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  getEDA,
  getExperiments,
  predict,
  type ExperimentRun,
  type PredictionResult,
  type TaskType,
} from "../api";
import { LoadingState } from "../components/LoadingState";
import { PageShell } from "../components/PageShell";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";

const COLORS = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#ec4899","#84cc16"];

interface FeatureMeta { min: number; max: number; mean: number }
interface AllResult extends PredictionResult { model: string; error?: boolean }
interface Props {
  datasetId: string | null;
  experimentsSyncKey?: number;
}

export default function Predict({ datasetId, experimentsSyncKey = 0 }: Props) {
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [features, setFeatures] = useState<Record<string, number>>({});
  const [meta, setMeta] = useState<Record<string, FeatureMeta>>({});
  const [order, setOrder] = useState<string[]>([]);
  const [taskType, setTaskType] = useState<TaskType | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [allResults, setAllResults] = useState<AllResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingAll, setLoadingAll] = useState(false);
  const [init, setInit] = useState(false);

  useEffect(() => {
    if (!datasetId) return;
    Promise.all([getExperiments(datasetId), getEDA(datasetId)]).then(([exp, eda]) => {
      setRuns(exp.runs);
      if (exp.runs[0]) {
        setSelected(exp.runs[0].model);
        setTaskType("accuracy" in exp.runs[0] ? "classification" : "regression");
      }
      const m: Record<string, FeatureMeta> = {};
      const d: Record<string, number> = {};
      eda.columns.forEach((col) => {
        const s = eda.stats[col];
        if (s) { m[col] = { min: s.min, max: s.max, mean: s.mean }; d[col] = s.mean; }
      });
      setMeta(m); setOrder(eda.columns); setFeatures(d); setInit(true);
    });
  }, [datasetId, experimentsSyncKey]);

  const handlePredict = async () => {
    if (!selected || !datasetId) return;
    try {
      setLoading(true); setResult(null);
      const r = await predict(datasetId, selected, features);
      setResult(r);
    } catch (e) { toast.error("Prediction failed: " + (e as Error).message); }
    finally { setLoading(false); }
  };

  const handleCompareAll = async () => {
    if (!datasetId || !runs.length) return;
    try {
      setLoadingAll(true); setAllResults(null);
      const results = await Promise.all(
        runs.map((r) =>
          predict(datasetId, r.model, features)
            .then((res): AllResult => ({ model: r.model, ...res }))
            .catch((): AllResult => ({ model: r.model, prediction: null, confidence: null, error: true }))
        )
      );
      setAllResults(results);
    } catch (e) { toast.error("Comparison failed: " + (e as Error).message); }
    finally { setLoadingAll(false); }
  };

  const reset = () => {
    const d: Record<string, number> = {};
    order.forEach((c) => { if (meta[c]) d[c] = meta[c].mean; });
    setFeatures(d); setResult(null); setAllResults(null);
  };

  const modelColor = (m: string) => COLORS[runs.findIndex((r) => r.model === m) % COLORS.length];

  if (!datasetId) return <Empty />;
  if (!init) {
    return (
      <PageShell title="Predict" description="Adjust feature values and compare model outputs.">
        <LoadingState variant="page" message="Loading models and feature metadata…" />
      </PageShell>
    );
  }
  if (!runs.length) return <Empty msg="No trained models found." />;

  const numPreds = allResults?.filter((r) => typeof r.prediction === "number") ?? [];
  const predMin = numPreds.length ? Math.min(...numPreds.map((r) => r.prediction as number)) : 0;
  const predMax = numPreds.length ? Math.max(...numPreds.map((r) => r.prediction as number)) : 1;
  const spread = predMax - predMin || 0.0001;

  return (
    <PageShell title="Predict" description="Adjust feature values and compare model outputs.">
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-5">
        {/* ── Left: Sliders ── */}
        <Card>
          <CardHeader>
            <CardTitle>Feature values</CardTitle>
            <p className="text-xs text-slate-400 mt-0.5">Grey tick = dataset mean</p>
          </CardHeader>
          <CardContent className="space-y-4">
            {order.filter((c) => meta[c]).map((col) => {
              const { min, max, mean } = meta[col];
              const val = features[col] ?? mean;
              const range = max - min;
              const step = range < 2 ? 0.001 : range < 20 ? 0.01 : 1;
              const meanPct = ((mean - min) / (max - min)) * 100;
              return (
                <div key={col}>
                  <div className="flex justify-between items-baseline mb-1">
                    <label className="text-sm font-medium text-slate-700">{col}</label>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-400">[{min} – {max}]</span>
                      <input
                        type="number"
                        value={val}
                        step={step}
                        min={min}
                        max={max}
                        onChange={(e) => setFeatures((p) => ({ ...p, [col]: parseFloat(e.target.value) || 0 }))}
                        className="w-20 text-right text-xs font-semibold border border-slate-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      />
                    </div>
                  </div>
                  <div className="relative">
                    <input
                      type="range" min={min} max={max} step={step} value={val}
                      onChange={(e) => setFeatures((p) => ({ ...p, [col]: parseFloat(e.target.value) }))}
                    />
                    {/* mean marker */}
                    <div
                      title={`mean: ${mean}`}
                      className="absolute top-0 w-0.5 h-4 bg-slate-400 rounded pointer-events-none"
                      style={{ left: `calc(${meanPct}% - 1px)` }}
                    />
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>

        {/* ── Right: Controls + Results ── */}
        <div className="space-y-4">
          {/* Controls */}
          <Card>
            <CardContent className="pt-5 space-y-3">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">Model</label>
                <select
                  value={selected ?? ""}
                  onChange={(e) => setSelected(e.target.value)}
                  className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {runs
                    .filter((r, i, arr) => arr.findIndex((x) => x.model === r.model) === i)
                    .map((r) => <option key={r.run_id} value={r.model}>{r.model.replace(/_/g, " ")}</option>)}
                </select>
              </div>
              <Button onClick={handlePredict} disabled={loading} className="w-full">
                {loading ? (
                  <><Loader2 className="w-4 h-4 animate-spin" />Predicting…</>
                ) : (
                  <><Sparkles className="w-4 h-4" />Predict</>
                )}
              </Button>
              <Button onClick={handleCompareAll} disabled={loadingAll} variant="success" className="w-full">
                {loadingAll ? (
                  <><Loader2 className="w-4 h-4 animate-spin" />Running all models…</>
                ) : (
                  <><ZapIcon className="w-4 h-4" />Compare All Models</>
                )}
              </Button>
              <Button onClick={reset} variant="ghost" size="sm" className="w-full text-slate-500">
                <RotateCcw className="w-3.5 h-3.5" />Reset to means
              </Button>
            </CardContent>
          </Card>

          {/* Single result */}
          {result && !allResults && (
            <Card className="border-emerald-200 bg-emerald-50">
              <CardContent className="pt-5">
                <p className="text-xs text-emerald-600 font-medium">{selected?.replace(/_/g, " ")} prediction</p>
                <p className="text-4xl font-bold text-emerald-800 mt-1 tabular-nums">
                  {result.prediction !== null
                    ? typeof result.prediction === "number"
                      ? result.prediction.toFixed(3)
                      : result.prediction
                    : "—"}
                </p>
                {result.confidence != null && (
                  <Badge variant="success" className="mt-2">
                    {(result.confidence * 100).toFixed(1)}% confidence
                  </Badge>
                )}
              </CardContent>
            </Card>
          )}

          {/* Compare all */}
          {allResults && (
            <Card>
              <CardHeader><CardTitle>All model predictions</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                {allResults.map((res, i) => {
                  const val = res.prediction;
                  const isNum = typeof val === "number";
                  const pct = isNum && numPreds.length > 1
                    ? (((val as number) - predMin) / spread) * 80 + 10
                    : 50;
                  const c = modelColor(res.model);
                  return (
                    <div key={`${res.model}-${i}`} className="flex items-center gap-2">
                      <div className="flex items-center gap-1.5 w-40 shrink-0">
                        <span className="w-2 h-2 rounded-full shrink-0" style={{ background: c }} />
                        <span className="text-xs text-slate-600 truncate">{res.model.replace(/_/g, " ")}</span>
                      </div>
                      {isNum && taskType === "regression" ? (
                        <>
                          <div className="flex-1 h-6 bg-slate-100 rounded overflow-hidden">
                            <div className="h-full rounded" style={{ width: `${pct}%`, background: c, opacity: 0.85 }} />
                          </div>
                          <span className="w-14 text-right text-sm font-bold tabular-nums">{(val as number).toFixed(3)}</span>
                        </>
                      ) : (
                        <span className={`text-sm font-bold ${res.error ? "text-rose-500" : "text-slate-800"}`}>
                          {res.error ? "error" : String(val)}
                          {res.confidence != null ? ` (${(res.confidence * 100).toFixed(0)}%)` : ""}
                        </span>
                      )}
                    </div>
                  );
                })}
                {taskType === "regression" && numPreds.length > 1 && (
                  <div className="pt-2 border-t border-slate-100 text-xs text-slate-500 flex gap-3">
                    <span>Range: <strong className="text-slate-700">{predMin.toFixed(3)} – {predMax.toFixed(3)}</strong></span>
                    <span>Spread: <strong className="text-slate-700">{(predMax - predMin).toFixed(3)}</strong></span>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </PageShell>
  );
}

function Empty({ msg = "Upload and train a dataset first." }: { msg?: string }) {
  return <PageShell title="Predict"><p className="text-sm text-slate-500">{msg}</p></PageShell>;
}
