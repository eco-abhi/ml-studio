import { useEffect, useState } from "react";
import { getExperiments, getImportances, type ExperimentRun, type ImportanceItem } from "../api";
import { Select } from "../components/ui/select";
import { PageShell } from "../components/PageShell";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Skeleton } from "../components/ui/skeleton";

const COLORS = ["#3b82f6", "#f59e0b"];

interface Props {
  datasetId: string | null;
  experimentsSyncKey?: number;
}

export default function Importances({ datasetId, experimentsSyncKey = 0 }: Props) {
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [modelA, setModelA] = useState<string | null>(null);
  const [modelB, setModelB] = useState<string | null>(null);
  const [impsA, setImpsA] = useState<ImportanceItem[] | null>(null);
  const [impsB, setImpsB] = useState<ImportanceItem[] | null>(null);
  const [loadingA, setLoadingA] = useState(false);
  const [loadingB, setLoadingB] = useState(false);

  useEffect(() => {
    if (!datasetId) return;
    getExperiments(datasetId).then((d) => {
      setRuns(d.runs);
      if (d.runs[0]) setModelA(d.runs[0].model);
      if (d.runs[1]) setModelB(d.runs[1].model);
    });
  }, [datasetId, experimentsSyncKey]);

  useEffect(() => {
    if (!modelA || !datasetId) return;
    setLoadingA(true);
    getImportances(datasetId, modelA)
      .then((res) => setImpsA(Array.isArray(res) ? res : null))
      .finally(() => setLoadingA(false));
  }, [modelA, datasetId]);

  useEffect(() => {
    if (!modelB || !datasetId) return;
    setLoadingB(true);
    getImportances(datasetId, modelB)
      .then((res) => setImpsB(Array.isArray(res) ? res : null))
      .finally(() => setLoadingB(false));
  }, [modelB, datasetId]);

  if (!datasetId) return <Empty />;
  if (!runs.length) return <Empty msg="No trained models found." />;

  // Deduplicate by model name so selectors never have duplicate keys/options
  const uniqueRuns = runs.filter((r, i, arr) => arr.findIndex((x) => x.model === r.model) === i);

  const mapA = impsA ? Object.fromEntries(impsA.map((f) => [f.feature, f.importance])) : {};
  const mapB = impsB ? Object.fromEntries(impsB.map((f) => [f.feature, f.importance])) : {};
  const features = impsA ? [...impsA].sort((a, b) => b.importance - a.importance).map((f) => f.feature) : [];
  const maxVal = Math.max(...Object.values(mapA), ...Object.values(mapB), 0.0001);
  const comparing = !!(modelB && modelB !== modelA && impsB);

  const rankB = impsB ? [...impsB].sort((a, b) => b.importance - a.importance).map((f) => f.feature) : [];
  const sharedTop3 = impsA ? impsA.slice(0, 3).map((f) => f.feature).filter((f) => rankB.slice(0, 3).includes(f)) : [];

  return (
    <PageShell title="Feature Importances" description="Compare which features each model relies on.">
      <div className="space-y-5">
        {/* ── Model Selectors ── */}
        <div className="grid grid-cols-2 gap-4">
          {([
            { label: "Model A", color: COLORS[0], value: modelA, set: setModelA },
            { label: "Model B", color: COLORS[1], value: modelB, set: setModelB },
          ] as const).map(({ label, color, value, set }) => (
            <div key={label}>
              <label className="flex items-center gap-2 text-sm font-medium text-slate-700 mb-1.5">
                <span className="w-3 h-3 rounded" style={{ background: color }} />
                {label}
              </label>
              <Select
                value={value ?? ""}
                onChange={set}
                options={uniqueRuns.map((r) => ({ value: r.model, label: r.model.replace(/_/g, " ") }))}
              />
            </div>
          ))}
        </div>

        {(loadingA || loadingB) && (
          <div className="space-y-3">{[1,2,3,4].map(i => <Skeleton key={i} className="h-10 w-full" />)}</div>
        )}

        {/* ── Legend ── */}
        {comparing && (
          <div className="flex gap-4">
            {[modelA, modelB].map((m, i) => (
              <div key={m} className="flex items-center gap-2 text-xs text-slate-600">
                <span className="w-3 h-3 rounded" style={{ background: COLORS[i] }} />
                {m?.replace(/_/g, " ")}
              </div>
            ))}
          </div>
        )}

        {/* ── Bars ── */}
        {features.length > 0 && !loadingA && (
          <Card>
            <CardContent className="pt-6 space-y-3">
              {features.map((feature) => {
                const vA = mapA[feature] ?? 0;
                const vB = mapB[feature] ?? 0;
                return (
                  <div key={feature}>
                    <div className="flex justify-between items-baseline mb-1">
                      <span className="text-sm font-medium text-slate-700">{feature}</span>
                      {comparing ? (
                        <span className="text-xs text-slate-500 tabular-nums">
                          <span style={{ color: COLORS[0] }} className="font-semibold">{vA.toFixed(4)}</span>
                          <span className="mx-1 text-slate-300">vs</span>
                          <span style={{ color: COLORS[1] }} className="font-semibold">{vB.toFixed(4)}</span>
                        </span>
                      ) : (
                        <span className="text-xs text-slate-500 tabular-nums font-semibold">{vA.toFixed(4)}</span>
                      )}
                    </div>
                    <div className="h-4 bg-slate-100 rounded overflow-hidden mb-0.5">
                      <div className="h-full rounded transition-all duration-300" style={{ width: `${(vA / maxVal) * 100}%`, background: COLORS[0] }} />
                    </div>
                    {comparing && (
                      <div className="h-4 bg-slate-100 rounded overflow-hidden">
                        <div className="h-full rounded transition-all duration-300" style={{ width: `${(vB / maxVal) * 100}%`, background: COLORS[1], opacity: 0.85 }} />
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Agreement summary */}
              {comparing && impsA && impsB && (
                <div className="mt-4 p-3 bg-slate-50 rounded-lg text-xs text-slate-600 border border-slate-200">
                  <strong>Top-3 agreement:</strong>{" "}
                  {sharedTop3.length === 0
                    ? "No overlap — models rely on very different features."
                    : `${sharedTop3.length}/3 shared (${sharedTop3.join(", ")}).`
                  }
                  {" "}Top for <span style={{ color: COLORS[0] }} className="font-semibold">{modelA?.replace(/_/g, " ")}</span>: <strong>{features[0]}</strong>.{" "}
                  Top for <span style={{ color: COLORS[1] }} className="font-semibold">{modelB?.replace(/_/g, " ")}</span>: <strong>{rankB[0]}</strong>.
                </div>
              )}

              <p className="text-xs text-slate-400 pt-2 border-t border-slate-100">
                Tree models: built-in split importances. Linear models: |coef| × feature std.
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </PageShell>
  );
}

function Empty({ msg = "Upload and train a dataset first." }: { msg?: string }) {
  return <PageShell title="Feature Importances"><p className="text-sm text-slate-500">{msg}</p></PageShell>;
}
