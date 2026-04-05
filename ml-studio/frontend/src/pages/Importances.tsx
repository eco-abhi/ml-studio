import { useCallback, useEffect, useMemo, useState } from "react";
import { getExperiments, getImportances, type ExperimentRun, type ImportanceItem } from "../api";
import { compactRunLabel, formatRunLabel, shortRunId } from "../lib/runLabel";
import { LoadingState } from "../components/LoadingState";
import { PageShell } from "../components/PageShell";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";

const PALETTE = [
  "#3b82f6",
  "#f59e0b",
  "#10b981",
  "#ef4444",
  "#8b5cf6",
  "#06b6d4",
  "#ec4899",
  "#84cc16",
  "#64748b",
  "#d946ef",
];

interface Props {
  datasetId: string | null;
  experimentsSyncKey?: number;
}

export default function Importances({ datasetId, experimentsSyncKey = 0 }: Props) {
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  /** MLflow run ids to compare (each training run is distinct, even for the same model name). */
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [byRunId, setByRunId] = useState<Record<string, ImportanceItem[] | null>>({});
  const [loading, setLoading] = useState(false);
  const [runsLoading, setRunsLoading] = useState(false);

  useEffect(() => {
    if (!datasetId) return;
    setRunsLoading(true);
    getExperiments(datasetId)
      .then((d) => {
        setRuns(d.runs);
        setSelected((prev) => {
          if (prev.size && d.runs.some((r) => prev.has(r.run_id))) {
            const next = new Set([...prev].filter((id) => d.runs.some((r) => r.run_id === id)));
            return next.size ? next : new Set(d.runs.map((r) => r.run_id));
          }
          return new Set(d.runs.map((r) => r.run_id));
        });
      })
      .finally(() => setRunsLoading(false));
  }, [datasetId, experimentsSyncKey]);

  const selectedRuns = useMemo(
    () => runs.filter((r) => selected.has(r.run_id)),
    [runs, selected],
  );

  const loadImportances = useCallback(async () => {
    if (!datasetId || selectedRuns.length === 0) {
      setByRunId({});
      return;
    }
    setLoading(true);
    try {
      const entries = await Promise.all(
        selectedRuns.map(async (run) => {
          try {
            const res = await getImportances(datasetId, run.model, run.run_id);
            return [run.run_id, res] as const;
          } catch {
            return [run.run_id, null] as const;
          }
        }),
      );
      setByRunId(Object.fromEntries(entries));
    } finally {
      setLoading(false);
    }
  }, [datasetId, selectedRuns]);

  useEffect(() => {
    void loadImportances();
  }, [loadImportances]);

  const toggleRun = (runId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) {
        if (next.size <= 1) return prev;
        next.delete(runId);
      } else {
        next.add(runId);
      }
      return next;
    });
  };

  const selectAll = () => setSelected(new Set(runs.map((r) => r.run_id)));
  const selectNone = () => {
    const first = runs[0]?.run_id;
    if (first) setSelected(new Set([first]));
  };

  if (!datasetId) return <Empty />;
  if (runsLoading) {
    return (
      <PageShell title="Feature Importances" description="Compare which features each model relies on.">
        <LoadingState variant="page" message="Loading trained models…" />
      </PageShell>
    );
  }
  if (!runs.length) return <Empty msg="No trained models found." />;

  const maps: Record<string, Record<string, number>> = {};
  for (const run of selectedRuns) {
    const list = byRunId[run.run_id];
    maps[run.run_id] = list ? Object.fromEntries(list.map((f) => [f.feature, f.importance])) : {};
  }

  const allFeatures = new Set<string>();
  for (const run of selectedRuns) {
    const list = byRunId[run.run_id];
    if (list) list.forEach((f) => allFeatures.add(f.feature));
  }

  const featureRankScore = (f: string) =>
    Math.max(0, ...selectedRuns.map((run) => maps[run.run_id]?.[f] ?? 0));

  const features = [...allFeatures].sort((a, b) => featureRankScore(b) - featureRankScore(a));

  const maxVal = Math.max(
    0.0001,
    ...selectedRuns.flatMap((run) => Object.values(maps[run.run_id] ?? {})),
  );

  const multiCompare = selectedRuns.length > 1;
  const colorFor = (i: number) => PALETTE[i % PALETTE.length];

  const top3Sets = selectedRuns.map((run) => {
    const list = byRunId[run.run_id];
    if (!list) return new Set<string>();
    return new Set([...list].sort((a, b) => b.importance - a.importance).slice(0, 3).map((f) => f.feature));
  });
  const inEveryTop3 =
    top3Sets.length > 0
      ? [...top3Sets[0]].filter((f) => top3Sets.every((s) => s.has(f)))
      : [];

  const top1ByRun = selectedRuns.map((run) => {
    const list = byRunId[run.run_id];
    if (!list?.length) return null;
    return [...list].sort((a, b) => b.importance - a.importance)[0]?.feature ?? null;
  });

  return (
    <PageShell title="Feature Importances" description="Compare which features each model relies on.">
      <div className="space-y-5">
        <div className="rounded-xl border border-slate-200 bg-slate-50/80 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <p className="text-sm font-medium text-slate-700">Runs to compare</p>
            <div className="flex gap-2">
              <Button type="button" variant="outline" size="sm" onClick={selectAll}>
                Select all
              </Button>
              <Button type="button" variant="outline" size="sm" onClick={selectNone}>
                Single only
              </Button>
            </div>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-2">
            {runs.map((r) => (
              <label
                key={r.run_id}
                className="flex cursor-pointer items-center gap-2 text-sm text-slate-700 select-none max-w-[280px]"
              >
                <input
                  type="checkbox"
                  className="rounded border-slate-300 text-blue-600 focus:ring-blue-500 shrink-0"
                  checked={selected.has(r.run_id)}
                  onChange={() => toggleRun(r.run_id)}
                />
                <span className="leading-tight">
                  <span className="block font-medium">{r.model.replace(/_/g, " ")}</span>
                  <span className="block text-[10px] text-slate-400 font-mono">
                    …{shortRunId(r.run_id)}
                    {r.started_at != null
                      ? ` · ${new Date(r.started_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`
                      : ""}
                  </span>
                </span>
              </label>
            ))}
          </div>
          {selectedRuns.length > 8 && (
            <p className="mt-2 text-[11px] text-amber-800/90">
              Many models selected — the chart uses one bar per model per feature; scroll or narrow the selection if it feels crowded.
            </p>
          )}
        </div>

        {loading && (
          <LoadingState message="Computing feature importances…" className="min-h-[200px]" />
        )}

        {multiCompare && selectedRuns.length > 0 && !loading && (
          <div className="flex flex-wrap gap-x-4 gap-y-2">
            {selectedRuns.map((run, i) => (
              <div key={run.run_id} className="flex items-center gap-2 text-xs text-slate-600 max-w-[220px]">
                <span className="h-3 w-3 shrink-0 rounded" style={{ background: colorFor(i) }} />
                <span className="truncate" title={formatRunLabel(run)}>
                  {compactRunLabel(run)}
                </span>
              </div>
            ))}
          </div>
        )}

        {features.length > 0 && !loading && (
          <Card>
            <CardContent className="space-y-3 pt-6">
              {features.map((feature) => (
                <div key={feature}>
                  <div className="mb-1 flex items-baseline justify-between gap-2">
                    <span className="text-sm font-medium text-slate-700">{feature}</span>
                    <span className="text-right text-[11px] tabular-nums text-slate-500">
                      {selectedRuns.map((run, i) => {
                        const v = maps[run.run_id]?.[feature] ?? 0;
                        return (
                          <span key={run.run_id}>
                            {i > 0 && <span className="text-slate-300"> · </span>}
                            <span style={{ color: colorFor(i) }} className="font-semibold">
                              {v.toFixed(4)}
                            </span>
                          </span>
                        );
                      })}
                    </span>
                  </div>
                  {selectedRuns.map((run, i) => {
                    const v = maps[run.run_id]?.[feature] ?? 0;
                    return (
                      <div key={run.run_id} className="mb-0.5 h-3.5 overflow-hidden rounded bg-slate-100 last:mb-0">
                        <div
                          className="h-full rounded transition-all duration-300"
                          style={{
                            width: `${(v / maxVal) * 100}%`,
                            background: colorFor(i),
                            opacity: selectedRuns.length > 1 ? 0.88 : 1,
                          }}
                        />
                      </div>
                    );
                  })}
                </div>
              ))}

              {multiCompare && selectedRuns.length >= 2 && (
                <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">
                  <strong>Top-3 agreement:</strong>{" "}
                  {inEveryTop3.length === 0 ? (
                    <>No single feature appears in the top 3 for every selected model.</>
                  ) : (
                    <>
                      In all models’ top 3: <strong>{inEveryTop3.join(", ")}</strong>.
                    </>
                  )}
                  <div className="mt-2 space-y-0.5 border-t border-slate-200 pt-2">
                    {selectedRuns.map((run, i) => (
                      <div key={run.run_id}>
                        Top for{" "}
                        <span style={{ color: colorFor(i) }} className="font-semibold">
                          {compactRunLabel(run)}
                        </span>
                        : <strong>{top1ByRun[i] ?? "—"}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <p className="border-t border-slate-100 pt-2 text-xs text-slate-400">
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
  return (
    <PageShell title="Feature Importances">
      <p className="text-sm text-slate-500">{msg}</p>
    </PageShell>
  );
}
