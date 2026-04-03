import { ChevronDown, ChevronRight, LayoutList, Loader2, Pencil, Plus, RotateCcw, Trash2, Wand2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { Select } from "../components/ui/select";
import {
  applyTransform,
  resetTransform,
  getCategorical,
  getCorrelationMatrix,
  getEDA,
  getFeatureTarget,
  getHealth,
  getMissingValues,
  getOutliers,
  getPairplot,
  getBoxplots,
  getScatter,
  getSkewness,
  getTargetAnalysis,
  getTransformHistory,
  previewDataset,
  type BoxplotData,
  type EdaDataSource,
  type CategoricalStats,
  type ColumnStats,
  type CorrelationMatrix,
  type EDAResult,
  type FeatureTargetData,
  type HealthResult,
  type MissingValues,
  type OutlierInfo,
  type PairplotData,
  type ScatterData,
  type SkewnessRow,
  type TargetAnalysis,
  type TransformHistory,
} from "../api";
import {
  type BinStep,
  type CastStep,
  type ClipStep,
  cloneStepsForEdit,
  type DatetimePart,
  type DeriveStep,
  type DropDupStep,
  type DropNullStep,
  deserializeStep,
  type FixSkewStep,
  type FreqEncStep,
  type ImputeStep,
  type MathStep,
  type RenameStep,
  type ScaleStep,
  type Step,
  type StepType,
  STEP_META,
  makeStep,
  serializeStep,
} from "../transformTypes";
import { EdaTransformWarningsTab } from "../edaTransformWarnings";
import { StepCard } from "./Transforms";
import { RenameEditor } from "../components/RenameEditor";
import { TransformTypePicker } from "../components/TransformTypePicker";
import { cn } from "../lib/utils";
import { PageShell } from "../components/PageShell";
import { Badge } from "../components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Skeleton } from "../components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";

interface Props {
  datasetId: string | null;
  transformSyncKey?: number;
  onTransformsMutated?: () => void;
}

/** Sub-tabs to compare original upload vs transformed data when a pipeline is active. */
function EdaBeforeAfterTabs({
  transformActive,
  children,
}: {
  transformActive: boolean;
  children: (source: EdaDataSource) => React.ReactNode;
}) {
  const [view, setView] = useState<"after" | "before">("after");
  const source: EdaDataSource = transformActive && view === "before" ? "original" : "transformed";
  if (!transformActive) {
    return <>{children("transformed")}</>;
  }
  return (
    <div className="space-y-3">
      <div className="inline-flex rounded-lg border border-slate-200 bg-slate-100/80 p-0.5 gap-0.5">
        <button
          type="button"
          onClick={() => setView("after")}
          className={cn(
            "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
            view === "after" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900",
          )}
        >
          After transforms
        </button>
        <button
          type="button"
          onClick={() => setView("before")}
          className={cn(
            "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
            view === "before" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900",
          )}
        >
          Before (original)
        </button>
      </div>
      <div>{children(source)}</div>
    </div>
  );
}

// ── Correlation colour interpolation ─────────────────────────────────────────
function corrColor(r: number) {
  // r in [-1, 1] → blue (negative) .. white (zero) .. red (positive)
  const abs = Math.abs(r);
  if (r >= 0) {
    const g = Math.round(255 - abs * 180);
    const b = Math.round(255 - abs * 180);
    return `rgb(255,${g},${b})`;
  }
  const g = Math.round(255 - abs * 180);
  const r2 = Math.round(255 - abs * 180);
  return `rgb(${r2},${g},255)`;
}

// ── Stats tab ─────────────────────────────────────────────────────────────────
function StatsPanel({ eda }: { eda: EDAResult }) {
  const [sel, setSel] = useState(eda.columns[0] ?? "");
  useEffect(() => {
    setSel((prev) => (eda.columns.includes(prev) ? prev : eda.columns[0] ?? ""));
  }, [eda.columns.join("\0")]);
  const s: ColumnStats | undefined = eda.stats[sel];
  const corr = sel ? eda.correlations[sel] : {};

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <div className="space-y-4">
        <Select
          value={sel}
          onChange={setSel}
          options={eda.columns.map((c) => ({ value: c, label: c }))}
        />

        {s && (() => {
          const statRows: [string, number][] = [
            ["Mean", s.mean], ["Std Dev", s.std], ["Min", s.min], ["Max", s.max],
            ["Median", s.median], ["Q1 (25%)", s.q25], ["Q3 (75%)", s.q75],
          ];
          return (
            <Card>
              <CardContent className="p-0">
                <div className="grid grid-cols-2 divide-x divide-y divide-slate-100">
                  {statRows.map(([label, value]) => (
                    <div key={label} className="p-4">
                      <p className="text-xs text-slate-500">{label}</p>
                      <p className="text-lg font-semibold text-slate-900 tabular-nums mt-0.5">{value}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          );
        })()}

        {/* Histogram */}
        {s && (() => {
          const maxCount = Math.max(...s.hist_values) || 1;
          // y-axis ticks: 0, 25%, 50%, 75%, 100% of max
          const yTicks = [0, 0.25, 0.5, 0.75, 1].map((t) => Math.round(t * maxCount));
          const BAR_H = 140; // px height of bar area
          return (
            <Card>
              <CardHeader><CardTitle>Distribution</CardTitle></CardHeader>
              <CardContent>
                <div className="flex gap-2">
                  {/* Y-axis */}
                  <div className="flex flex-col-reverse justify-between text-right shrink-0" style={{ height: BAR_H }}>
                    {yTicks.map((t) => (
                      <span key={t} className="text-[10px] text-slate-400 tabular-nums leading-none">{t.toLocaleString()}</span>
                    ))}
                  </div>

                  {/* Bars + x labels */}
                  <div className="flex-1 flex flex-col gap-1">
                    {/* Bar area */}
                    <div className="flex items-end gap-px" style={{ height: BAR_H }}>
                      {s.hist_values.map((v, i) => (
                        <div
                          key={i}
                          className="flex-1 bg-blue-500 rounded-t-sm opacity-80 hover:opacity-100 transition-opacity cursor-default"
                          style={{ height: `${(v / maxCount) * 100}%` }}
                          title={`${s.hist_bins[i]?.toFixed(2)} – ${s.hist_bins[i + 1]?.toFixed(2)}: ${v} samples`}
                        />
                      ))}
                    </div>
                    {/* X-axis: show first, middle, last bin edge */}
                    <div className="flex justify-between text-[10px] text-slate-400 tabular-nums px-0">
                      <span>{s.hist_bins[0]?.toFixed(2)}</span>
                      <span>{s.hist_bins[Math.floor(s.hist_bins.length / 2)]?.toFixed(2)}</span>
                      <span>{s.hist_bins[s.hist_bins.length - 1]?.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
                <p className="text-[11px] text-slate-400 mt-2 text-center">
                  {s.hist_values.length} bins · hover a bar for exact range &amp; count
                </p>
              </CardContent>
            </Card>
          );
        })()}
      </div>

      {/* Correlations */}
      <div className="space-y-4">
        <Card>
          <CardHeader><CardTitle>Correlations with "{sel}"</CardTitle></CardHeader>
          <CardContent className="p-0 max-h-80 overflow-y-auto">
            {Object.entries(corr ?? {})
              .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
              .map(([feat, r]) => {
                const abs = Math.abs(r);
                const pos = r >= 0;
                return (
                  <div key={feat} className="flex items-center gap-3 px-4 py-2 border-b border-slate-100 last:border-0">
                    <span className="text-sm text-slate-700 flex-1">{feat}</span>
                    <div className="w-40 h-3 bg-slate-100 rounded-full relative overflow-hidden">
                      <div
                        className="absolute h-full rounded-full"
                        style={{
                          left: pos ? "50%" : `${(1 - abs) * 50}%`,
                          width: `${abs * 50}%`,
                          background: pos ? "#10b981" : "#ef4444",
                        }}
                      />
                    </div>
                    <span className={`text-xs font-semibold w-12 text-right tabular-nums ${pos ? "text-emerald-600" : "text-rose-600"}`}>
                      {r.toFixed(3)}
                    </span>
                  </div>
                );
              })}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function StatsTab({
  eda,
  edaOriginal,
  transformActive,
}: {
  eda: EDAResult;
  edaOriginal: EDAResult | null;
  transformActive: boolean;
}) {
  return (
    <EdaBeforeAfterTabs transformActive={transformActive}>
      {(source) => {
        const activeEda = source === "original" && edaOriginal ? edaOriginal : eda;
        return <StatsPanel eda={activeEda} />;
      }}
    </EdaBeforeAfterTabs>
  );
}

// ── Heatmap tab ───────────────────────────────────────────────────────────────
function HeatmapInner({ datasetId, source }: { datasetId: string; source: EdaDataSource }) {
  const [data, setData] = useState<CorrelationMatrix | null>(null);
  useEffect(() => { getCorrelationMatrix(datasetId, source).then(setData); }, [datasetId, source]);
  if (!data) return <div className="space-y-2">{[1,2,3].map(i=><Skeleton key={i} className="h-8 w-full"/>)}</div>;

  const n = data.columns.length;
  const cellSize = Math.min(52, Math.floor(680 / n));

  return (
    <Card>
      <CardHeader><CardTitle>Correlation Heatmap</CardTitle></CardHeader>
      <CardContent className="overflow-auto">
        <div style={{ display: "grid", gridTemplateColumns: `80px repeat(${n}, ${cellSize}px)`, gap: 1 }}>
          {/* Top-left empty */}
          <div />
          {/* Column headers */}
          {data.columns.map((c) => (
            <div key={c} className="text-[10px] text-slate-500 font-medium text-center truncate px-0.5" style={{ width: cellSize }}>
              {c}
            </div>
          ))}
          {/* Rows */}
          {data.columns.map((row, ri) => (
            <>
              <div key={`lbl-${row}`} className="text-[10px] text-slate-500 font-medium text-right pr-2 flex items-center justify-end truncate">
                {row}
              </div>
              {data.matrix[ri].map((val, ci) => (
                <div
                  key={`${ri}-${ci}`}
                  title={`${row} × ${data.columns[ci]}: ${val.toFixed(3)}`}
                  className="flex items-center justify-center rounded"
                  style={{ width: cellSize, height: cellSize, background: corrColor(val), fontSize: cellSize > 36 ? 9 : 0 }}
                >
                  {cellSize > 36 ? val.toFixed(2) : ""}
                </div>
              ))}
            </>
          ))}
        </div>
        <div className="flex items-center gap-2 mt-4 text-xs text-slate-500">
          <div className="w-4 h-4 rounded" style={{ background: corrColor(-1) }} /><span>−1</span>
          <div className="w-4 h-4 rounded" style={{ background: corrColor(0) }} /><span>0</span>
          <div className="w-4 h-4 rounded" style={{ background: corrColor(1) }} /><span>+1</span>
        </div>
      </CardContent>
    </Card>
  );
}

function HeatmapTab({ datasetId, transformActive }: { datasetId: string; transformActive: boolean }) {
  return (
    <EdaBeforeAfterTabs transformActive={transformActive}>
      {(source) => <HeatmapInner datasetId={datasetId} source={source} />}
    </EdaBeforeAfterTabs>
  );
}

// ── Scatter tab ───────────────────────────────────────────────────────────────
function ScatterInner({
  datasetId,
  columns,
  source,
}: {
  datasetId: string;
  columns: string[];
  source: EdaDataSource;
}) {
  const [colX, setColX] = useState(columns[0] ?? "");
  const [colY, setColY] = useState(columns[1] ?? columns[0] ?? "");
  const [target, setTarget] = useState("");
  const [data, setData] = useState<ScatterData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setColX((x) => (columns.includes(x) ? x : columns[0] ?? ""));
    setColY((y) => (columns.includes(y) ? y : columns[1] ?? columns[0] ?? ""));
    setTarget((t) => (t && columns.includes(t) ? t : ""));
  }, [columns.join("\0")]);

  useEffect(() => {
    setData(null);
  }, [source]);

  const fetchScatter = () => {
    if (!colX || !colY) return;
    setLoading(true);
    getScatter(datasetId, colX, colY, target || undefined, source)
      .then(setData)
      .finally(() => setLoading(false));
  };

  const W = 520, H = 320, PAD = 40;
  const xs = data?.x ?? [];
  const ys = data?.y ?? [];
  const minX = xs.length ? Math.min(...xs) : 0;
  const maxX = xs.length ? Math.max(...xs) : 1;
  const minY = ys.length ? Math.min(...ys) : 0;
  const maxY = ys.length ? Math.max(...ys) : 1;
  const sx = (v: number) => PAD + ((v - minX) / (maxX - minX || 1)) * (W - PAD * 2);
  const sy = (v: number) => H - PAD - ((v - minY) / (maxY - minY || 1)) * (H - PAD * 2);

  // color scale for target
  const colors = data?.color;
  const numericColor = colors?.every((c) => typeof c === "number");
  const minC = numericColor ? Math.min(...(colors as number[])) : 0;
  const maxC = numericColor ? Math.max(...(colors as number[])) : 1;
  const dotColor = (i: number) => {
    if (!colors) return "#3b82f6";
    if (numericColor) {
      const t = ((colors[i] as number) - minC) / (maxC - minC || 1);
      const g = Math.round(255 - t * 200);
      return `rgb(${Math.round(t * 220)},${g},${Math.round(255 - t * 220)})`;
    }
    const cats = [...new Set(colors as string[])];
    const idx = cats.indexOf(colors[i] as string);
    const palette = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6"];
    return palette[idx % palette.length];
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Scatter Plot</CardTitle>
        <div className="flex gap-2 mt-2 flex-wrap">
          {([["X axis", colX, setColX], ["Y axis", colY, setColY]] as [string, string, (v: string) => void][]).map(([label, val, setter]) => (
            <div key={label}>
              <label className="text-xs text-slate-500 block mb-1">{label}</label>
              <Select
                value={val}
                onChange={setter}
                options={columns.map((c) => ({ value: c, label: c }))}
                size="sm"
                className="w-36"
              />
            </div>
          ))}
          <div>
            <label className="text-xs text-slate-500 block mb-1">Color (optional)</label>
            <Select
              value={target}
              onChange={setTarget}
              options={[{ value: "", label: "None" }, ...columns.map((c) => ({ value: c, label: c }))]}
              size="sm"
              className="w-36"
            />
          </div>
          <div className="flex items-end">
            <button
              type="button"
              onClick={fetchScatter}
              disabled={loading}
              className="px-3 py-1.5 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? "Loading…" : "Plot"}
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {data && (
          <svg width={W} height={H} className="w-full">
            {/* Axes */}
            <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#e2e8f0" strokeWidth={1} />
            <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#e2e8f0" strokeWidth={1} />
            <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={10} fill="#94a3b8">{colX}</text>
            <text x={10} y={H / 2} textAnchor="middle" fontSize={10} fill="#94a3b8" transform={`rotate(-90 10 ${H/2})`}>{colY}</text>
            {/* Points */}
            {xs.map((x, i) => (
              <circle key={i} cx={sx(x)} cy={sy(ys[i])} r={3} fill={dotColor(i)} fillOpacity={0.7} />
            ))}
          </svg>
        )}
        {!data && <p className="text-sm text-slate-500 py-8 text-center">Select axes and click Plot</p>}
      </CardContent>
    </Card>
  );
}

function ScatterTab({
  datasetId,
  columns,
  columnsOriginal,
  transformActive,
}: {
  datasetId: string;
  columns: string[];
  columnsOriginal: string[];
  transformActive: boolean;
}) {
  return (
    <EdaBeforeAfterTabs transformActive={transformActive}>
      {(source) => {
        const cols = source === "original" && columnsOriginal.length > 0 ? columnsOriginal : columns;
        return <ScatterInner datasetId={datasetId} columns={cols} source={source} />;
      }}
    </EdaBeforeAfterTabs>
  );
}

// ── Missing tab ───────────────────────────────────────────────────────────────
function MissingInner({ datasetId, source }: { datasetId: string; source: EdaDataSource }) {
  const [data, setData] = useState<MissingValues | null>(null);
  useEffect(() => { getMissingValues(datasetId, source).then(setData); }, [datasetId, source]);
  if (!data) return <Skeleton className="h-40 w-full" />;

  const cols = Object.entries(data).sort(([, a], [, b]) => b.pct - a.pct);
  const total = cols.reduce((s, [, v]) => s + v.count, 0);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Missing Values</CardTitle>
        {total === 0 && <Badge variant="success" className="mt-1">No missing values</Badge>}
      </CardHeader>
      <CardContent>
        {total > 0 ? (
          <div className="space-y-2.5">
            {cols.filter(([, v]) => v.count > 0).map(([col, { count, pct }]) => (
              <div key={col}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-700 font-medium">{col}</span>
                  <span className="text-slate-500 tabular-nums">{count.toLocaleString()} ({pct}%)</span>
                </div>
                <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full rounded-full bg-amber-400" style={{ width: `${pct}%` }} />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-500">All columns are complete.</p>
        )}
      </CardContent>
    </Card>
  );
}

function MissingTab({ datasetId, transformActive }: { datasetId: string; transformActive: boolean }) {
  return (
    <EdaBeforeAfterTabs transformActive={transformActive}>
      {(source) => <MissingInner datasetId={datasetId} source={source} />}
    </EdaBeforeAfterTabs>
  );
}

// ── Outliers tab ──────────────────────────────────────────────────────────────
function OutliersInner({ datasetId, source }: { datasetId: string; source: EdaDataSource }) {
  const [data, setData] = useState<OutlierInfo | null>(null);
  useEffect(() => { getOutliers(datasetId, source).then(setData); }, [datasetId, source]);
  if (!data) return <Skeleton className="h-40 w-full" />;

  const cols = Object.entries(data).sort(([, a], [, b]) => b.n_outliers - a.n_outliers);

  return (
    <Card>
      <CardHeader><CardTitle>Outliers (IQR method)</CardTitle></CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50">
                {["Feature", "Lower fence", "Upper fence", "# Outliers"].map((h) => (
                  <th key={h} className="px-4 py-2.5 text-left font-medium text-slate-600">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cols.map(([col, { lower, upper, n_outliers }]) => (
                <tr key={col} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                  <td className="px-4 py-2.5 font-medium text-slate-800">{col}</td>
                  <td className="px-4 py-2.5 tabular-nums text-slate-600">{lower.toFixed(4)}</td>
                  <td className="px-4 py-2.5 tabular-nums text-slate-600">{upper.toFixed(4)}</td>
                  <td className="px-4 py-2.5 tabular-nums">
                    {n_outliers > 0
                      ? <Badge variant="warning">{n_outliers}</Badge>
                      : <Badge variant="success">0</Badge>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function OutliersTab({ datasetId, transformActive }: { datasetId: string; transformActive: boolean }) {
  return (
    <EdaBeforeAfterTabs transformActive={transformActive}>
      {(source) => <OutliersInner datasetId={datasetId} source={source} />}
    </EdaBeforeAfterTabs>
  );
}

// ── Categorical tab ───────────────────────────────────────────────────────────
function CategoricalInner({ datasetId, source }: { datasetId: string; source: EdaDataSource }) {
  const [data, setData] = useState<CategoricalStats | null>(null);
  useEffect(() => { getCategorical(datasetId, source).then(setData); }, [datasetId, source]);
  if (!data) return <Skeleton className="h-40 w-full" />;

  const cols = Object.entries(data);
  if (!cols.length) return <p className="text-sm text-slate-500">No categorical (non-numeric) columns found.</p>;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      {cols.map(([col, { n_unique, top_values }]) => {
        const maxCount = top_values[0]?.[1] ?? 1;
        return (
          <Card key={col}>
            <CardHeader>
              <CardTitle>{col}</CardTitle>
              <Badge variant="secondary">{n_unique} unique values</Badge>
            </CardHeader>
            <CardContent className="space-y-2">
              {top_values.map(([val, count]) => (
                <div key={val}>
                  <div className="flex justify-between text-xs mb-0.5">
                    <span className="text-slate-700 truncate max-w-[180px]">{val}</span>
                    <span className="text-slate-500 tabular-nums ml-2">{count.toLocaleString()}</span>
                  </div>
                  <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-blue-500" style={{ width: `${(count / maxCount) * 100}%` }} />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

function CategoricalTab({ datasetId, transformActive }: { datasetId: string; transformActive: boolean }) {
  return (
    <EdaBeforeAfterTabs transformActive={transformActive}>
      {(source) => <CategoricalInner datasetId={datasetId} source={source} />}
    </EdaBeforeAfterTabs>
  );
}

// ── Health tab ────────────────────────────────────────────────────────────────
function HealthInner({ datasetId, source }: { datasetId: string; source: EdaDataSource }) {
  const [data, setData] = useState<HealthResult | null>(null);
  useEffect(() => { getHealth(datasetId, source).then(setData); }, [datasetId, source]);
  if (!data) return <div className="space-y-3">{[1,2,3,4].map(i=><Skeleton key={i} className="h-16 w-full"/>)}</div>;

  const completenessEntries = Object.entries(data.completeness).sort(([,a],[,b]) => a - b);

  return (
    <div className="space-y-5">
      {/* KPI row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: "Rows", value: data.n_rows.toLocaleString() },
          { label: "Columns", value: data.n_cols.toString() },
          { label: "Duplicate rows", value: data.n_duplicates.toLocaleString(), warn: data.n_duplicates > 0 },
          { label: "Memory", value: `${data.memory_mb} MB` },
        ].map(({ label, value, warn }) => (
          <Card key={label} className={warn ? "border-amber-300 bg-amber-50" : ""}>
            <CardContent className="pt-4 pb-3">
              <p className="text-xs text-slate-500">{label}</p>
              <p className={`text-2xl font-bold tabular-nums mt-0.5 ${warn ? "text-amber-700" : "text-slate-900"}`}>{value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Dtype breakdown */}
      <Card>
        <CardHeader><CardTitle className="text-sm">Column types</CardTitle></CardHeader>
        <CardContent className="flex gap-4">
          <Badge variant="secondary">{data.dtype_counts.numeric} numeric</Badge>
          <Badge variant="secondary">{data.dtype_counts.categorical} categorical</Badge>
        </CardContent>
      </Card>

      {/* Flags */}
      {(data.constant_cols.length > 0 || data.near_zero_cols.length > 0 || data.high_card_cols.length > 0) && (
        <Card className="border-amber-200">
          <CardHeader><CardTitle className="text-sm text-amber-800">Warnings</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm">
            {data.constant_cols.length > 0 && (
              <div>
                <p className="font-medium text-rose-700 mb-1">Constant columns (zero variance — useless for ML)</p>
                <div className="flex flex-wrap gap-1">{data.constant_cols.map(c=><Badge key={c} variant="warning">{c}</Badge>)}</div>
              </div>
            )}
            {data.near_zero_cols.length > 0 && (
              <div>
                <p className="font-medium text-amber-700 mb-1">Near-zero variance</p>
                <div className="flex flex-wrap gap-1">{data.near_zero_cols.map(c=><Badge key={c} variant="warning">{c}</Badge>)}</div>
              </div>
            )}
            {data.high_card_cols.length > 0 && (
              <div>
                <p className="font-medium text-amber-700 mb-1">High cardinality (&gt;50% unique — risky to one-hot encode)</p>
                <div className="flex flex-wrap gap-1">{data.high_card_cols.map(c=><Badge key={c} variant="warning">{c}</Badge>)}</div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Completeness bars */}
      <Card>
        <CardHeader><CardTitle className="text-sm">Column completeness</CardTitle></CardHeader>
        <CardContent className="space-y-2.5">
          {completenessEntries.map(([col, pct]) => (
            <div key={col}>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-700 font-medium truncate max-w-[240px]">{col}</span>
                <span className={`tabular-nums font-semibold ml-2 ${pct < 80 ? "text-rose-600" : pct < 95 ? "text-amber-600" : "text-emerald-600"}`}>{pct}%</span>
              </div>
              <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${pct < 80 ? "bg-rose-400" : pct < 95 ? "bg-amber-400" : "bg-emerald-400"}`} style={{ width: `${pct}%` }} />
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

function HealthTab({ datasetId, transformActive }: { datasetId: string; transformActive: boolean }) {
  return (
    <EdaBeforeAfterTabs transformActive={transformActive}>
      {(source) => <HealthInner datasetId={datasetId} source={source} />}
    </EdaBeforeAfterTabs>
  );
}

// ── Target tab ────────────────────────────────────────────────────────────────
function TargetInner({
  datasetId,
  columns,
  source,
}: {
  datasetId: string;
  columns: string[];
  source: EdaDataSource;
}) {
  const [sel, setSel] = useState(columns[0] ?? "");
  const [data, setData] = useState<TargetAnalysis | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setSel((prev) => (columns.includes(prev) ? prev : columns[0] ?? ""));
  }, [columns.join("\0")]);

  useEffect(() => {
    if (!sel) return;
    setLoading(true);
    setData(null);
    getTargetAnalysis(datasetId, sel, source).then(setData).finally(() => setLoading(false));
  }, [sel, datasetId, source]);

  return (
    <div className="space-y-5">
      <Select value={sel} onChange={setSel} options={columns.map(c=>({ value: c, label: c }))} />
      {loading && <div className="space-y-3">{[1,2,3].map(i=><Skeleton key={i} className="h-16 w-full"/>)}</div>}
      {data && data.is_numeric && (
        <>
          {/* Skewness badge + hint */}
          <div className="flex flex-wrap gap-2 items-center">
            <Badge variant={Math.abs(data.skewness!) < 0.5 ? "success" : Math.abs(data.skewness!) < 1 ? "warning" : "warning"}>
              skewness: {data.skewness!.toFixed(3)}
            </Badge>
            <Badge variant="secondary">kurtosis: {data.kurtosis!.toFixed(3)}</Badge>
            {data.transform_hint && (
              <span className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-0.5">
                💡 Suggested transform: {data.transform_hint}
              </span>
            )}
          </div>

          {/* Stats row */}
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
            {([["Mean", data.mean], ["Std", data.std], ["Min", data.min], ["Median", data.median], ["Max", data.max], ["N", data.n]] as [string, number][]).map(([l, v]) => (
              <Card key={l}>
                <CardContent className="pt-3 pb-2">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wide">{l}</p>
                  <p className="text-base font-semibold tabular-nums text-slate-900">{typeof v === "number" ? v.toLocaleString() : v}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Histogram */}
          {(() => {
            const maxCount = Math.max(...data.hist_values!);
            const BAR_H = 140;
            const yTicks = [0, 0.25, 0.5, 0.75, 1].map(t => Math.round(t * maxCount));
            return (
              <Card>
                <CardHeader><CardTitle className="text-sm">Distribution of {sel}</CardTitle></CardHeader>
                <CardContent>
                  <div className="flex gap-2">
                    <div className="flex flex-col-reverse justify-between text-right shrink-0" style={{ height: BAR_H }}>
                      {yTicks.map(t => <span key={t} className="text-[10px] text-slate-400 tabular-nums leading-none">{t.toLocaleString()}</span>)}
                    </div>
                    <div className="flex-1 flex flex-col gap-1">
                      <div className="flex items-end gap-px" style={{ height: BAR_H }}>
                        {data.hist_values!.map((v, i) => (
                          <div key={i} className="flex-1 bg-blue-500 rounded-t-sm opacity-80 hover:opacity-100 transition-opacity cursor-default"
                            style={{ height: `${(v / maxCount) * 100}%` }}
                            title={`${data.hist_bins![i]?.toFixed(2)} – ${data.hist_bins![i+1]?.toFixed(2)}: ${v}`} />
                        ))}
                      </div>
                      <div className="flex justify-between text-[10px] text-slate-400 tabular-nums">
                        <span>{data.hist_bins![0]?.toFixed(2)}</span>
                        <span>{data.hist_bins![Math.floor(data.hist_bins!.length/2)]?.toFixed(2)}</span>
                        <span>{data.hist_bins![data.hist_bins!.length-1]?.toFixed(2)}</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })()}
        </>
      )}

      {data && !data.is_numeric && (
        <>
          <div className="flex gap-2 items-center flex-wrap">
            <Badge variant="secondary">{data.n_classes} classes</Badge>
            <Badge variant={data.is_imbalanced ? "warning" : "success"}>
              imbalance ratio: {data.imbalance_ratio?.toFixed(2)}×{data.is_imbalanced ? " ⚠ imbalanced" : " ✓ balanced"}
            </Badge>
          </div>
          <Card>
            <CardHeader><CardTitle className="text-sm">Class distribution</CardTitle></CardHeader>
            <CardContent className="space-y-2.5">
              {data.class_counts!.map(([cls, count, pct]) => (
                <div key={cls}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="font-medium text-slate-700 truncate max-w-[200px]">{cls}</span>
                    <span className="text-slate-500 tabular-nums ml-2">{count.toLocaleString()} ({pct}%)</span>
                  </div>
                  <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full rounded-full bg-blue-500" style={{ width: `${pct}%` }} />
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function TargetTab({
  datasetId,
  columns,
  columnsOriginal,
  transformActive,
}: {
  datasetId: string;
  columns: string[];
  columnsOriginal: string[];
  transformActive: boolean;
}) {
  return (
    <EdaBeforeAfterTabs transformActive={transformActive}>
      {(source) => {
        const cols = source === "original" && columnsOriginal.length > 0 ? columnsOriginal : columns;
        return <TargetInner datasetId={datasetId} columns={cols} source={source} />;
      }}
    </EdaBeforeAfterTabs>
  );
}

// ── Skewness tab ──────────────────────────────────────────────────────────────
function SkewnessInner({ datasetId, source }: { datasetId: string; source: EdaDataSource }) {
  const [data, setData] = useState<SkewnessRow[] | null>(null);
  useEffect(() => { getSkewness(datasetId, source).then(setData); }, [datasetId, source]);
  if (!data) return <Skeleton className="h-40 w-full" />;

  const severityColor = { normal: "text-emerald-600", moderate: "text-amber-600", high: "text-rose-600" };
  const severityBg    = { normal: "bg-emerald-500",  moderate: "bg-amber-500",   high: "bg-rose-500" };
  const maxAbs = Math.max(...data.map(r => r.abs_skewness), 0.001);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Skewness & Kurtosis</CardTitle>
        <p className="text-xs text-slate-500 mt-1">
          |skew| &lt; 0.5 = normal · 0.5–1 = moderate · &gt;1 = high. Sorted by severity.
        </p>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50">
                {["Column", "Skewness", "Kurtosis", "Severity", "Suggested transform", "Visual"].map(h => (
                  <th key={h} className="px-4 py-2.5 text-left font-medium text-slate-600 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map(row => (
                <tr key={row.column} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                  <td className="px-4 py-2.5 font-medium text-slate-800">{row.column}</td>
                  <td className={`px-4 py-2.5 tabular-nums font-semibold ${severityColor[row.severity]}`}>{row.skewness.toFixed(4)}</td>
                  <td className="px-4 py-2.5 tabular-nums text-slate-600">{row.kurtosis.toFixed(4)}</td>
                  <td className="px-4 py-2.5">
                    <Badge variant={row.severity === "normal" ? "success" : "warning"}>{row.severity}</Badge>
                  </td>
                  <td className="px-4 py-2.5 text-slate-500 text-xs">{row.hint ?? "—"}</td>
                  <td className="px-4 py-2.5 w-32">
                    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full ${severityBg[row.severity]}`} style={{ width: `${(row.abs_skewness / maxAbs) * 100}%` }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function SkewnessTab({ datasetId, transformActive }: { datasetId: string; transformActive: boolean }) {
  return (
    <EdaBeforeAfterTabs transformActive={transformActive}>
      {(source) => <SkewnessInner datasetId={datasetId} source={source} />}
    </EdaBeforeAfterTabs>
  );
}

// ── Boxplot tab ───────────────────────────────────────────────────────────────
function BoxplotInner({
  datasetId,
  columns,
  source,
}: {
  datasetId: string;
  columns: string[];
  source: EdaDataSource;
}) {
  const [data, setData] = useState<BoxplotData | null>(null);
  const [sel, setSel] = useState<string[]>(columns.slice(0, 6));
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setSel((prev) => {
      const kept = prev.filter((c) => columns.includes(c));
      if (kept.length > 0) return kept.slice(0, 10);
      return columns.slice(0, 6);
    });
  }, [columns.join("\0")]);

  useEffect(() => {
    setLoading(true);
    getBoxplots(datasetId, sel.length ? sel : undefined, source).then(setData).finally(() => setLoading(false));
  }, [datasetId, sel.join(","), source]);

  const PALETTE = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4"];
  const H = 180;

  return (
    <div className="space-y-4">
      {/* Column selector chips */}
      <div className="flex flex-wrap gap-2">
        {columns.map((c, i) => {
          const active = sel.includes(c);
          return (
            <button key={c} onClick={() => setSel(prev => active ? prev.filter(x=>x!==c) : [...prev, c].slice(0,10))}
              className={`px-2.5 py-1 rounded-full text-xs border transition-colors ${active ? "text-white border-transparent" : "border-slate-200 text-slate-600 hover:border-blue-400"}`}
              style={active ? { background: PALETTE[i % PALETTE.length] } : {}}
            >{c}</button>
          );
        })}
      </div>

      {loading && <Skeleton className="h-60 w-full" />}

      {data && !loading && (
        <Card>
          <CardContent className="pt-5 overflow-x-auto">
            <div className="flex gap-6 min-w-max items-end pb-2">
              {Object.entries(data).map(([col, s], idx) => {
                const range = s.upper_fence - s.lower_fence || 1;
                const toY = (v: number) => H - ((Math.min(Math.max(v, s.lower_fence), s.upper_fence) - s.lower_fence) / range) * H;
                const q1Y = toY(s.q1), q3Y = toY(s.q3), medY = toY(s.median), loY = toY(s.lower_fence), hiY = toY(s.upper_fence);
                const color = PALETTE[idx % PALETTE.length];
                return (
                  <div key={col} className="flex flex-col items-center gap-2" style={{ width: 72 }}>
                    <svg width={72} height={H} className="overflow-visible">
                      {/* Whiskers */}
                      <line x1={36} y1={loY} x2={36} y2={q1Y} stroke={color} strokeWidth={1.5} strokeDasharray="3 2" />
                      <line x1={36} y1={q3Y} x2={36} y2={hiY} stroke={color} strokeWidth={1.5} strokeDasharray="3 2" />
                      {/* Fence caps */}
                      <line x1={24} y1={loY} x2={48} y2={loY} stroke={color} strokeWidth={1.5} />
                      <line x1={24} y1={hiY} x2={48} y2={hiY} stroke={color} strokeWidth={1.5} />
                      {/* Box */}
                      <rect x={18} y={q3Y} width={36} height={Math.max(q1Y - q3Y, 2)} fill={color} fillOpacity={0.15} stroke={color} strokeWidth={1.5} rx={2} />
                      {/* Median */}
                      <line x1={18} y1={medY} x2={54} y2={medY} stroke={color} strokeWidth={2.5} />
                      {/* Mean dot */}
                      <circle cx={36} cy={toY(s.mean)} r={3} fill={color} />
                      {/* Outliers */}
                      {s.outliers.map((v, i) => {
                        const oy = H - ((v - s.lower_fence) / range) * H;
                        return <circle key={i} cx={36 + (Math.random() * 16 - 8)} cy={oy} r={2} fill={color} fillOpacity={0.4} />;
                      })}
                    </svg>
                    <span className="text-[10px] text-slate-600 text-center truncate w-full font-medium">{col}</span>
                    <span className="text-[10px] text-slate-400 tabular-nums">med: {s.median.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
            <p className="text-[11px] text-slate-400 mt-3">Box = IQR · line = median · dot = mean · dashes = fences · scattered dots = outliers</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function BoxplotTab({
  datasetId,
  columns,
  columnsOriginal,
  transformActive,
}: {
  datasetId: string;
  columns: string[];
  columnsOriginal: string[];
  transformActive: boolean;
}) {
  return (
    <EdaBeforeAfterTabs transformActive={transformActive}>
      {(source) => {
        const cols = source === "original" && columnsOriginal.length > 0 ? columnsOriginal : columns;
        return <BoxplotInner datasetId={datasetId} columns={cols} source={source} />;
      }}
    </EdaBeforeAfterTabs>
  );
}

// ── Pairplot tab ──────────────────────────────────────────────────────────────
function PairplotInner({
  datasetId,
  columns,
  source,
}: {
  datasetId: string;
  columns: string[];
  source: EdaDataSource;
}) {
  const [sel, setSel] = useState<string[]>(columns.slice(0, 4));
  const [data, setData] = useState<PairplotData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setSel((prev) => {
      const kept = prev.filter((c) => columns.includes(c));
      if (kept.length > 0) return kept.slice(0, 6);
      return columns.slice(0, 4);
    });
  }, [columns.join("\0")]);

  useEffect(() => {
    if (!sel.length) return;
    setLoading(true);
    setData(null);
    getPairplot(datasetId, sel, source).then(setData).finally(() => setLoading(false));
  }, [datasetId, sel.join(","), source]);

  const CELL = 100;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-slate-500 self-center">Columns (max 6):</span>
        {columns.map(c => {
          const active = sel.includes(c);
          return (
            <button key={c} onClick={() => setSel(prev => active ? prev.filter(x=>x!==c) : prev.length < 6 ? [...prev, c] : prev)}
              className={`px-2.5 py-1 rounded-full text-xs border transition-colors ${active ? "bg-blue-600 text-white border-transparent" : "border-slate-200 text-slate-600 hover:border-blue-400"}`}
            >{c}</button>
          );
        })}
      </div>

      {loading && <Skeleton className="h-64 w-full" />}

      {data && !loading && (
        <Card>
          <CardContent className="pt-4 overflow-auto">
            <div style={{ display: "grid", gridTemplateColumns: `60px repeat(${data.columns.length}, ${CELL}px)` }}>
              {/* top-left empty */}
              <div />
              {/* column headers */}
              {data.columns.map(c => (
                <div key={c} className="text-[10px] text-slate-500 text-center truncate px-1 font-medium pb-1">{c}</div>
              ))}
              {data.columns.map((rowCol) => (
                <>
                  {/* row label */}
                  <div key={`lbl-${rowCol}`} className="text-[10px] text-slate-500 text-right pr-2 flex items-center justify-end truncate font-medium">{rowCol}</div>
                  {data.columns.map((colCol) => {
                    const key = `${rowCol}___${colCol}`;
                    const cell = data.pairs[key];
                    if (!cell) return <div key={key} style={{ width: CELL, height: CELL }} />;
                    if (cell.type === "hist") {
                      const maxV = Math.max(...cell.values, 1);
                      return (
                        <div key={key} style={{ width: CELL, height: CELL }} className="bg-blue-50 border border-slate-100 flex items-end gap-px p-1 rounded">
                          {cell.values.map((v, i) => (
                            <div key={i} className="flex-1 bg-blue-500 rounded-sm opacity-70" style={{ height: `${(v/maxV)*80}%` }} />
                          ))}
                        </div>
                      );
                    }
                    const xs = cell.x, ys = cell.y;
                    const minX = Math.min(...xs), maxX = Math.max(...xs) || 1;
                    const minY = Math.min(...ys), maxY = Math.max(...ys) || 1;
                    const sx = (v: number) => 4 + ((v-minX)/(maxX-minX||1))*(CELL-8);
                    const sy = (v: number) => (CELL-4) - ((v-minY)/(maxY-minY||1))*(CELL-8);
                    return (
                      <svg key={key} width={CELL} height={CELL} className="border border-slate-100 rounded bg-white">
                        {xs.map((x, i) => (
                          <circle key={i} cx={sx(x)} cy={sy(ys[i])} r={1.5} fill="#3b82f6" fillOpacity={0.4} />
                        ))}
                      </svg>
                    );
                  })}
                </>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function PairplotTab({
  datasetId,
  columns,
  columnsOriginal,
  transformActive,
}: {
  datasetId: string;
  columns: string[];
  columnsOriginal: string[];
  transformActive: boolean;
}) {
  return (
    <EdaBeforeAfterTabs transformActive={transformActive}>
      {(source) => {
        const cols = source === "original" && columnsOriginal.length > 0 ? columnsOriginal : columns;
        return <PairplotInner datasetId={datasetId} columns={cols} source={source} />;
      }}
    </EdaBeforeAfterTabs>
  );
}

// ── Feature vs Target tab ─────────────────────────────────────────────────────
const PALETTE_FT = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#ec4899","#84cc16"];

function FeatureTargetInner({
  datasetId,
  columns,
  source,
}: {
  datasetId: string;
  columns: string[];
  source: EdaDataSource;
}) {
  const [target, setTarget] = useState(columns[columns.length - 1] ?? "");
  const [data, setData]     = useState<FeatureTargetData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setTarget((prev) => (columns.includes(prev) ? prev : columns[columns.length - 1] ?? ""));
  }, [columns.join("\0")]);

  useEffect(() => {
    if (!target) return;
    setLoading(true);
    setData(null);
    getFeatureTarget(datasetId, target, source).then(setData).finally(() => setLoading(false));
  }, [datasetId, target, source]);

  const W = 280, H = 160, PAD = 28;

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <label className="text-sm font-medium text-slate-700">Target column</label>
        <Select value={target} onChange={setTarget} options={columns.map(c=>({ value: c, label: c }))} className="w-48" />
      </div>

      {loading && <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">{[1,2,3,4].map(i=><Skeleton key={i} className="h-52 w-full"/>)}</div>}

      {data && !loading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {Object.entries(data.features).map(([col, feat]) => (
            <Card key={col}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm">{col}</CardTitle>
                  {feat.type === "scatter" && (
                    <Badge variant={Math.abs(feat.correlation) > 0.5 ? "success" : "secondary"}>
                      r = {feat.correlation.toFixed(3)}
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                {feat.type === "scatter" && (() => {
                  const xs = feat.x, ys = feat.y;
                  const minX = Math.min(...xs), maxX = Math.max(...xs);
                  const minY = Math.min(...ys), maxY = Math.max(...ys);
                  const sx = (v: number) => PAD + ((v-minX)/(maxX-minX||1))*(W-PAD*2);
                  const sy = (v: number) => H-PAD - ((v-minY)/(maxY-minY||1))*(H-PAD*2);
                  return (
                    <svg width={W} height={H} className="w-full">
                      <line x1={PAD} y1={H-PAD} x2={W-PAD} y2={H-PAD} stroke="#e2e8f0" strokeWidth={1}/>
                      <line x1={PAD} y1={PAD}   x2={PAD}   y2={H-PAD} stroke="#e2e8f0" strokeWidth={1}/>
                      <text x={W/2} y={H-4} textAnchor="middle" fontSize={8} fill="#94a3b8">{col}</text>
                      <text x={8}   y={H/2} textAnchor="middle" fontSize={8} fill="#94a3b8" transform={`rotate(-90 8 ${H/2})`}>{target}</text>
                      {xs.map((x, i) => <circle key={i} cx={sx(x)} cy={sy(ys[i])} r={2} fill="#3b82f6" fillOpacity={0.5}/>)}
                      {feat.trend && (
                        <line x1={sx(feat.trend.x[0])} y1={sy(feat.trend.y[0])} x2={sx(feat.trend.x[1])} y2={sy(feat.trend.y[1])}
                          stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 2"/>
                      )}
                    </svg>
                  );
                })()}
                {feat.type === "boxgroup" && (() => {
                  const classes = Object.entries(feat.classes);
                  const allVals = classes.flatMap(([,s]) => [s.min, s.max]);
                  const minV = Math.min(...allVals), maxV = Math.max(...allVals);
                  const toY = (v: number) => H-PAD - ((v-minV)/(maxV-minV||1))*(H-PAD*2);
                  const bw = Math.min(36, Math.floor((W - PAD*2) / classes.length) - 8);
                  const cx = (i: number) => PAD + ((i+0.5) / classes.length) * (W-PAD*2);
                  return (
                    <svg width={W} height={H} className="w-full">
                      <line x1={PAD} y1={H-PAD} x2={W-PAD} y2={H-PAD} stroke="#e2e8f0" strokeWidth={1}/>
                      {classes.map(([cls, s], i) => {
                        const color = PALETTE_FT[i % PALETTE_FT.length];
                        const x = cx(i);
                        return (
                          <g key={cls}>
                            <line x1={x} y1={toY(s.min)} x2={x} y2={toY(s.max)} stroke={color} strokeWidth={1} strokeDasharray="2 2"/>
                            <rect x={x-bw/2} y={toY(s.q3)} width={bw} height={Math.max(toY(s.q1)-toY(s.q3),1)} fill={color} fillOpacity={0.2} stroke={color} strokeWidth={1.5} rx={1}/>
                            <line x1={x-bw/2} y1={toY(s.median)} x2={x+bw/2} y2={toY(s.median)} stroke={color} strokeWidth={2}/>
                            <text x={x} y={H-4} textAnchor="middle" fontSize={8} fill="#94a3b8">{cls}</text>
                          </g>
                        );
                      })}
                    </svg>
                  );
                })()}
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

function FeatureTargetTab({
  datasetId,
  columns,
  columnsOriginal,
  transformActive,
}: {
  datasetId: string;
  columns: string[];
  columnsOriginal: string[];
  transformActive: boolean;
}) {
  return (
    <EdaBeforeAfterTabs transformActive={transformActive}>
      {(source) => {
        const cols = source === "original" && columnsOriginal.length > 0 ? columnsOriginal : columns;
        return <FeatureTargetInner datasetId={datasetId} columns={cols} source={source} />;
      }}
    </EdaBeforeAfterTabs>
  );
}

// ── Column multi-picker (inline, no Transforms import) ────────────────────────

function ColPicker({ all, selected, onChange }: {
  all: string[]; selected: string[]; onChange: (v: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, [open]);
  useEffect(() => {
    if (!open) setQ("");
  }, [open]);

  const filtered = useMemo(() => {
    const n = q.trim().toLowerCase();
    if (!n) return all;
    return all.filter((c) => c.toLowerCase().includes(n));
  }, [all, q]);

  const toggle = (c: string) => onChange(selected.includes(c) ? selected.filter(x => x !== c) : [...selected, c]);
  return (
    <div className="relative" ref={ref}>
      <button type="button" onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between gap-2 border border-slate-200 rounded-lg px-3 py-2 text-sm bg-white hover:border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-400 transition-colors">
        <span className="flex items-center gap-2 min-w-0">
          <LayoutList className="w-3.5 h-3.5 text-slate-400 shrink-0" />
          <span className={cn("truncate", selected.length ? "text-slate-800 font-medium" : "text-slate-400")}>
            {selected.length ? `${selected.length} column${selected.length > 1 ? "s" : ""} selected` : "Choose columns…"}
          </span>
        </span>
        <ChevronDown className={cn("w-4 h-4 text-slate-400 shrink-0 transition-transform", open && "rotate-180")} />
      </button>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mt-2">
          {selected.map(c => (
            <span key={c} className="inline-flex items-center gap-1 pl-2 pr-1 py-0.5 rounded-md bg-blue-50 border border-blue-100 text-xs text-blue-800">
              <span className="max-w-[140px] truncate">{c}</span>
              <button type="button" onClick={() => toggle(c)} className="p-0.5 rounded hover:bg-blue-100 text-blue-600" aria-label={`Remove ${c}`}>
                <X className="w-3 h-3" />
              </button>
            </span>
          ))}
        </div>
      )}
      {open && (
        <div className="absolute z-50 mt-1 w-full min-w-[min(100%,260px)] bg-white border border-slate-200 rounded-xl shadow-lg overflow-hidden flex flex-col max-h-64">
          <div className="p-2 border-b border-slate-100 bg-slate-50/90 space-y-2">
            <input
              type="search"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Filter columns…"
              className="w-full rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500/30"
            />
            <div className="flex gap-3 px-0.5">
              <button type="button" onClick={() => onChange([...all])} className="text-xs font-medium text-blue-600 hover:underline">Select all</button>
              <button type="button" onClick={() => onChange([])} className="text-xs font-medium text-slate-500 hover:underline">Clear</button>
            </div>
          </div>
          <div className="overflow-y-auto py-1">
            {filtered.length === 0 ? (
              <p className="px-3 py-4 text-xs text-slate-400 text-center">No columns match</p>
            ) : (
              filtered.map(c => (
                <label key={c} className="flex items-center gap-2.5 px-3 py-2 hover:bg-slate-50 cursor-pointer text-sm text-slate-700">
                  <input type="checkbox" checked={selected.includes(c)} onChange={() => toggle(c)}
                    className="rounded border-slate-300 text-blue-600 focus:ring-blue-500" />
                  <span className="truncate">{c}</span>
                </label>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function edaOptChip(active: boolean, className?: string) {
  return cn(
    "rounded-lg text-xs font-medium border transition-all",
    active
      ? "bg-blue-600 border-blue-600 text-white shadow-sm"
      : "border-slate-200 text-slate-600 bg-white hover:border-slate-300 hover:bg-slate-50",
    className,
  );
}

const EDA_DT_PARTS: { value: DatetimePart; label: string }[] = [
  { value: "year", label: "Year" },
  { value: "month", label: "Mo" },
  { value: "day", label: "Day" },
  { value: "hour", label: "Hr" },
  { value: "minute", label: "Min" },
  { value: "dow", label: "DoW" },
  { value: "doy", label: "DoY" },
  { value: "week", label: "Wk" },
];

// ── Quick Transform Panel ─────────────────────────────────────────────────────

function QuickTransformPanel({
  datasetId,
  allColumns,
  onApplied,
  queueSeed,
  onQueueSeedConsumed = () => {},
}: {
  datasetId: string;
  allColumns: string[];
  onApplied: () => void;
  /** When set, replaces the queue and opens the panel in full-pipeline replace mode. */
  queueSeed: { key: number; steps: Step[] } | null;
  onQueueSeedConsumed?: () => void;
}) {
  const [open, setOpen]         = useState(false);
  const [addType, setAddType]   = useState<StepType>("impute");
  const [steps, setSteps]       = useState<Step[]>([]);
  const [applying, setApplying] = useState(false);
  /** When true, apply runs from the original CSV and replaces all saved steps. */
  const [pipelineReplaceMode, setPipelineReplaceMode] = useState(false);

  useEffect(() => {
    if (!queueSeed) return;
    setSteps(queueSeed.steps);
    setPipelineReplaceMode(true);
    setOpen(true);
    onQueueSeedConsumed();
  }, [queueSeed, onQueueSeedConsumed]);

  const addStep = () => setSteps(p => [...p, makeStep(addType)]);
  const removeStep = (id: string) => setSteps(p => p.filter(s => s.id !== id));
  const updateStep = (id: string, patch: Partial<Step>) =>
    setSteps(p => p.map(s => s.id === id ? { ...s, ...patch } as Step : s));

  const handleApply = async () => {
    if (!steps.length) { toast.error("Add at least one step."); return; }
    try {
      setApplying(true);
      const fromOriginal = pipelineReplaceMode;
      const res = await applyTransform(datasetId, steps.map(serializeStep), fromOriginal);
      await previewDataset(datasetId, 1).catch(() => null);
      toast.success(
        fromOriginal
          ? `Pipeline replaced — ${res.shape[0].toLocaleString()} × ${res.shape[1]}.`
          : `Applied ${res.steps_applied} step(s) — ${res.shape[0]} × ${res.shape[1]}.`
      );
      setSteps([]);
      if (fromOriginal) setPipelineReplaceMode(false);
      setOpen(false);
      onApplied();
    } catch (e) {
      toast.error("Apply failed: " + (e as Error).message);
    } finally {
      setApplying(false);
    }
  };

  return (
    <div className="mb-4 rounded-2xl border border-blue-200/70 bg-gradient-to-b from-blue-50/80 to-white shadow-sm ring-1 ring-blue-100/50">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="flex w-full items-start gap-3 px-4 py-3.5 text-left transition-colors hover:bg-blue-50/60 sm:items-center"
      >
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white shadow-md shadow-blue-600/25">
          <Wand2 className="h-5 w-5" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-semibold text-slate-800">Quick transform</span>
            {steps.length > 0 && (
              <span className="rounded-full bg-blue-600 px-2 py-0.5 text-[11px] font-bold tabular-nums text-white">{steps.length} in queue</span>
            )}
          </div>
          <p className="mt-0.5 text-xs leading-snug text-slate-500">
            Build a short pipeline here — charts and stats refresh after you apply.
          </p>
        </div>
        {open ? <ChevronDown className="mt-1 h-4 w-4 shrink-0 text-blue-500" /> : <ChevronRight className="mt-1 h-4 w-4 shrink-0 text-blue-500" />}
      </button>

      {open && (
        <div className="space-y-4 border-t border-blue-200/80 bg-white/80 px-4 pb-4 pt-4 backdrop-blur-[2px]">
          {pipelineReplaceMode && (
            <div className="rounded-xl border border-violet-200 bg-violet-50 px-3 py-2.5 text-[11px] leading-snug text-violet-900">
              <p className="font-semibold text-violet-950">Replacing the full pipeline</p>
              <p className="mt-0.5 text-violet-800/90">
                Apply runs from your <strong>original</strong> upload and overwrites every saved step with the queue below.
              </p>
            </div>
          )}
          <div>
            <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Add a step</label>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-stretch">
              <TransformTypePicker value={addType} onChange={setAddType} size="sm" className="min-w-0 flex-1" />
              <button
                type="button"
                onClick={addStep}
                className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-700"
              >
                <Plus className="h-4 w-4" />
                Add to queue
              </button>
            </div>
            <p className="mt-2 rounded-lg bg-slate-50 px-3 py-2 text-xs leading-relaxed text-slate-600">{STEP_META[addType].description}</p>
          </div>

          {steps.length === 0 && (
            <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50/60 px-4 py-8 text-center">
              <p className="text-sm font-medium text-slate-600">Nothing in the queue yet</p>
              <p className="mt-1 text-xs text-slate-400">
                Pick a transform type, then tap <span className="font-medium text-slate-500">Add to queue</span>.
              </p>
            </div>
          )}

          <div className="space-y-3">
            {steps.map((step, idx) => (
              <div key={step.id} className="rounded-xl border border-slate-200/90 bg-white shadow-sm ring-1 ring-slate-100">
                <div className="flex items-start gap-3 border-b border-slate-100 bg-gradient-to-r from-slate-50/90 to-white px-3 py-2.5">
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-xs font-bold text-white">{idx + 1}</span>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-slate-800">{STEP_META[step.type].label}</p>
                    <p className="line-clamp-2 text-[11px] leading-snug text-slate-500">{STEP_META[step.type].description}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeStep(step.id)}
                    className="shrink-0 rounded-lg p-2 text-slate-300 transition-colors hover:bg-rose-50 hover:text-rose-600"
                    aria-label="Remove step from queue"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>

                <div className="space-y-3 p-3">
                  {step.type === "rename_columns" ? (
                    <RenameEditor
                      allColumns={allColumns}
                      mapping={(step as RenameStep).mapping}
                      onChange={(mapping) => updateStep(step.id, { mapping } as Partial<Step>)}
                    />
                  ) : (
                    <>
                      {step.type !== "tfidf_column" && step.type !== "derive_numeric" && step.type !== "train_test_split" && (
                        <div>
                          <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Columns</label>
                          <ColPicker
                            all={allColumns}
                            selected={(step as { columns: string[] }).columns ?? []}
                            onChange={(cols) => updateStep(step.id, { columns: cols } as Partial<Step>)}
                          />
                        </div>
                      )}

                      {step.type === "impute" && (
                        <div>
                          <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Fill strategy</label>
                          <Select
                            size="sm"
                            value={(step as ImputeStep).strategy}
                            onChange={(v) => updateStep(step.id, { strategy: v } as Partial<Step>)}
                            options={[
                              { value: "mean", label: "Mean" },
                              { value: "median", label: "Median" },
                              { value: "mode", label: "Mode" },
                              { value: "zero", label: "Zero" },
                            ]}
                          />
                        </div>
                      )}
                      {step.type === "clip_outliers" && (
                        <div>
                          <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Detection method</label>
                          <div className="flex gap-1.5">
                            {(["iqr", "zscore"] as ClipStep["method"][]).map((m) => (
                              <button
                                key={m}
                                type="button"
                                onClick={() => updateStep(step.id, { method: m } as Partial<Step>)}
                                className={edaOptChip((step as ClipStep).method === m, "flex-1 py-2")}
                              >
                                {m === "iqr" ? "IQR" : "Z-score"}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      {step.type === "scale" && (
                        <div>
                          <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Scaler</label>
                          <div className="flex gap-1.5">
                            {(["standard", "minmax", "robust"] as ScaleStep["method"][]).map((m) => (
                              <button
                                key={m}
                                type="button"
                                onClick={() => updateStep(step.id, { method: m } as Partial<Step>)}
                                className={edaOptChip((step as ScaleStep).method === m, "flex-1 py-2")}
                              >
                                {m === "standard" ? "Standard" : m === "minmax" ? "Min-max" : "Robust"}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      {step.type === "math_transform" && (
                        <div>
                          <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Function</label>
                          <div className="flex flex-wrap gap-1.5">
                            {(["log1p", "sqrt", "square", "reciprocal", "abs"] as MathStep["method"][]).map((m) => (
                              <button
                                key={m}
                                type="button"
                                onClick={() => updateStep(step.id, { method: m } as Partial<Step>)}
                                className={edaOptChip((step as MathStep).method === m, "px-2.5 py-2")}
                              >
                                {m === "log1p" ? "log1p" : m === "sqrt" ? "√x" : m === "square" ? "x²" : m === "reciprocal" ? "1/x" : "|x|"}
                              </button>
                            ))}
                          </div>
                          <p className="mt-1.5 text-[11px] text-slate-400">Applied to selected numeric columns. log1p and √x treat negatives as 0.</p>
                        </div>
                      )}
                      {step.type === "fix_skewness" && (
                        <div className="space-y-2">
                          <div>
                            <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Method</label>
                            <div className="flex flex-wrap gap-1.5">
                              {(["auto", "log1p", "sqrt", "box_cox", "yeo_johnson"] as FixSkewStep["method"][]).map((m) => (
                                <button
                                  key={m}
                                  type="button"
                                  onClick={() => updateStep(step.id, { method: m } as Partial<Step>)}
                                  className={edaOptChip((step as FixSkewStep).method === m, "px-2 py-2")}
                                >
                                  {m === "auto" ? "Auto" : m === "box_cox" ? "Box-Cox" : m === "yeo_johnson" ? "Yeo-Johnson" : m}
                                </button>
                              ))}
                            </div>
                          </div>
                          <div className="flex flex-wrap items-center gap-2 rounded-lg bg-slate-50 px-2 py-2">
                            <label className="text-xs text-slate-600" htmlFor={`skew-th-${step.id}`}>Skip if |skew| &lt;</label>
                            <input
                              id={`skew-th-${step.id}`}
                              type="number"
                              min={0}
                              max={3}
                              step={0.1}
                              value={(step as FixSkewStep).threshold}
                              onChange={(e) => updateStep(step.id, { threshold: parseFloat(e.target.value) || 0.5 } as Partial<Step>)}
                              className="w-16 rounded-md border border-slate-200 px-2 py-1 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500/30"
                            />
                          </div>
                        </div>
                      )}
                      {step.type === "cast_dtype" && (
                        <div>
                          <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Target type</label>
                          <div className="flex gap-1.5">
                            {(["float", "int", "str"] as CastStep["dtype"][]).map((d) => (
                              <button
                                key={d}
                                type="button"
                                onClick={() => updateStep(step.id, { dtype: d } as Partial<Step>)}
                                className={edaOptChip((step as CastStep).dtype === d, "flex-1 py-2 capitalize")}
                              >
                                {d}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      {step.type === "drop_nulls" && (
                        <div>
                          <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Drop when</label>
                          <div className="flex gap-1.5">
                            {(["any", "all"] as DropNullStep["how"][]).map((h) => (
                              <button
                                key={h}
                                type="button"
                                onClick={() => updateStep(step.id, { how: h } as Partial<Step>)}
                                className={edaOptChip((step as DropNullStep).how === h, "flex-1 py-2")}
                              >
                                {h === "any" ? "Any column null" : "All selected null"}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      {step.type === "drop_duplicates" && (
                        <div>
                          <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Which duplicate to keep</label>
                          <div className="flex gap-1.5">
                            {(["first", "last", "none"] as DropDupStep["keep"][]).map((k) => (
                              <button
                                key={k}
                                type="button"
                                onClick={() => updateStep(step.id, { keep: k } as Partial<Step>)}
                                className={edaOptChip((step as DropDupStep).keep === k, "flex-1 py-2")}
                              >
                                {k === "first" ? "First" : k === "last" ? "Last" : "Drop all"}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      {step.type === "frequency_encode" && (
                        <div>
                          <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Value</label>
                          <div className="flex gap-1.5">
                            {([true, false] as boolean[]).map((n) => (
                              <button
                                key={String(n)}
                                type="button"
                                onClick={() => updateStep(step.id, { normalize: n } as Partial<Step>)}
                                className={edaOptChip((step as FreqEncStep).normalize === n, "flex-1 py-2")}
                              >
                                {n ? "Proportion" : "Raw count"}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                      {step.type === "bin_numeric" && (
                        <div className="space-y-2">
                          <div className="flex flex-wrap items-end gap-2">
                            <div>
                              <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Bins</label>
                              <input
                                type="number"
                                min={2}
                                max={50}
                                value={(step as BinStep).n_bins}
                                onChange={(e) => updateStep(step.id, { n_bins: Math.max(2, parseInt(e.target.value, 10) || 5) } as Partial<Step>)}
                                className="w-20 rounded-lg border border-slate-200 px-2.5 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30"
                              />
                            </div>
                          </div>
                          <div>
                            <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-wide text-slate-500">Bin edges</label>
                            <div className="flex gap-1.5">
                              {(["equal_width", "quantile"] as BinStep["strategy"][]).map((s) => (
                                <button
                                  key={s}
                                  type="button"
                                  onClick={() => updateStep(step.id, { strategy: s } as Partial<Step>)}
                                  className={edaOptChip((step as BinStep).strategy === s, "flex-1 py-2")}
                                >
                                  {s === "equal_width" ? "Equal width" : "Quantile"}
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                      {step.type === "polynomial_features" && (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <label className="text-[11px] text-slate-500">Degree</label>
                            <input
                              type="number"
                              min={2}
                              max={4}
                              value={(step as { degree: number }).degree}
                              onChange={(e) => updateStep(step.id, { degree: Math.min(4, Math.max(2, parseInt(e.target.value, 10) || 2)) } as Partial<Step>)}
                              className="w-14 rounded border border-slate-200 px-2 py-1 text-xs"
                            />
                          </div>
                          <label className="flex items-center gap-2 text-[11px] text-slate-600">
                            <input
                              type="checkbox"
                              checked={(step as { interaction_only: boolean }).interaction_only}
                              onChange={(e) => updateStep(step.id, { interaction_only: e.target.checked } as Partial<Step>)}
                            />
                            Interactions only
                          </label>
                          <label className="flex items-center gap-2 text-[11px] text-slate-600">
                            <input
                              type="checkbox"
                              checked={(step as { include_bias: boolean }).include_bias}
                              onChange={(e) => updateStep(step.id, { include_bias: e.target.checked } as Partial<Step>)}
                            />
                            Bias
                          </label>
                        </div>
                      )}
                      {step.type === "extract_datetime" && (
                        <div className="flex flex-wrap gap-1">
                          {EDA_DT_PARTS.map(({ value, label }) => {
                            const parts = (step as { parts: DatetimePart[] }).parts ?? [];
                            const on = parts.includes(value);
                            return (
                              <button
                                key={value}
                                type="button"
                                onClick={() => {
                                  const next = on ? parts.filter((p) => p !== value) : [...parts, value];
                                  updateStep(step.id, { parts: next.length ? next : ["year"] } as Partial<Step>);
                                }}
                                className={edaOptChip(on, "px-2 py-1")}
                              >
                                {label}
                              </button>
                            );
                          })}
                        </div>
                      )}
                      {step.type === "pca_projection" && (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <label className="text-[11px] text-slate-500">Components</label>
                            <input
                              type="number"
                              min={1}
                              max={50}
                              value={(step as { n_components: number }).n_components}
                              onChange={(e) => updateStep(step.id, { n_components: Math.max(1, parseInt(e.target.value, 10) || 3) } as Partial<Step>)}
                              className="w-14 rounded border border-slate-200 px-2 py-1 text-xs"
                            />
                          </div>
                          <input
                            type="text"
                            value={(step as { prefix: string }).prefix}
                            onChange={(e) => updateStep(step.id, { prefix: e.target.value || "PC_" } as Partial<Step>)}
                            placeholder="Prefix"
                            className="w-full rounded border border-slate-200 px-2 py-1 text-xs"
                          />
                          <label className="flex items-center gap-2 text-[11px] text-slate-600">
                            <input
                              type="checkbox"
                              checked={(step as { drop_original: boolean }).drop_original}
                              onChange={(e) => updateStep(step.id, { drop_original: e.target.checked } as Partial<Step>)}
                            />
                            Drop originals
                          </label>
                        </div>
                      )}
                      {step.type === "tfidf_column" && (
                        <div className="space-y-2">
                          <Select
                            size="sm"
                            value={(step as { column: string }).column || allColumns[0] || ""}
                            onChange={(v) => updateStep(step.id, { column: v } as Partial<Step>)}
                            options={allColumns.map((c) => ({ value: c, label: c }))}
                          />
                          <div className="flex gap-2">
                            <input
                              type="number"
                              min={5}
                              max={200}
                              value={(step as { max_features: number }).max_features}
                              onChange={(e) => updateStep(step.id, { max_features: Math.min(200, Math.max(5, parseInt(e.target.value, 10) || 50)) } as Partial<Step>)}
                              className="w-16 rounded border border-slate-200 px-2 py-1 text-xs"
                            />
                            <span className="text-[10px] text-slate-400 self-center">max feat.</span>
                            <input
                              type="number"
                              min={1}
                              max={3}
                              value={(step as { ngram_max: number }).ngram_max}
                              onChange={(e) => updateStep(step.id, { ngram_max: Math.min(3, Math.max(1, parseInt(e.target.value, 10) || 1)) } as Partial<Step>)}
                              className="w-12 rounded border border-slate-200 px-2 py-1 text-xs"
                            />
                            <span className="text-[10px] text-slate-400 self-center">ngram</span>
                          </div>
                        </div>
                      )}
                      {step.type === "derive_numeric" && (
                        <div className="space-y-2">
                          <div className="grid grid-cols-2 gap-2">
                            <Select
                              size="sm"
                              value={(step as DeriveStep).column_a || allColumns[0] || ""}
                              onChange={(v) => updateStep(step.id, { column_a: v } as Partial<Step>)}
                              options={allColumns.map((c) => ({ value: c, label: c }))}
                            />
                            <Select
                              size="sm"
                              value={(step as DeriveStep).column_b || allColumns[0] || ""}
                              onChange={(v) => updateStep(step.id, { column_b: v } as Partial<Step>)}
                              options={allColumns.map((c) => ({ value: c, label: c }))}
                            />
                          </div>
                          <div className="flex gap-1">
                            {(["add", "subtract", "multiply", "divide"] as DeriveStep["op"][]).map((op) => (
                              <button
                                key={op}
                                type="button"
                                onClick={() => updateStep(step.id, { op } as Partial<Step>)}
                                className={edaOptChip((step as DeriveStep).op === op, "flex-1 py-1 text-[10px]")}
                              >
                                {op === "add" ? "+" : op === "subtract" ? "−" : op === "multiply" ? "×" : "÷"}
                              </button>
                            ))}
                          </div>
                          <input
                            type="text"
                            value={(step as DeriveStep).output_column}
                            onChange={(e) => updateStep(step.id, { output_column: e.target.value || "derived" } as Partial<Step>)}
                            placeholder="Output name"
                            className="w-full rounded border border-slate-200 px-2 py-1 text-xs"
                          />
                        </div>
                      )}
                      {step.type === "target_encode_dataset" && (
                        <div className="space-y-2">
                          <Select
                            size="sm"
                            value={(step as { target_column: string }).target_column || allColumns[0] || ""}
                            onChange={(v) => updateStep(step.id, { target_column: v } as Partial<Step>)}
                            options={allColumns.map((c) => ({ value: c, label: c }))}
                          />
                          <p className="text-[10px] text-amber-700">Full-data means can leak into modeling.</p>
                        </div>
                      )}
                      {step.type === "train_test_split" && (
                        <div className="space-y-2">
                          <Select
                            size="sm"
                            value={(step as { target_column: string }).target_column || allColumns[0] || ""}
                            onChange={(v) => updateStep(step.id, { target_column: v } as Partial<Step>)}
                            options={allColumns.map((c) => ({ value: c, label: c }))}
                          />
                          <input
                            type="range"
                            min={0.05}
                            max={0.5}
                            step={0.05}
                            value={(step as { test_size: number }).test_size ?? 0.2}
                            onChange={(e) => updateStep(step.id, { test_size: parseFloat(e.target.value) } as Partial<Step>)}
                            className="w-full"
                          />
                          <p className="text-[10px] text-slate-500">Adds __ml_split__. Reorder in Transforms like other steps.</p>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>

          {steps.length > 0 && (
            <div className="flex flex-col-reverse gap-2 border-t border-slate-200/80 pt-3 sm:flex-row sm:items-center sm:justify-end">
              <button
                type="button"
                onClick={() => {
                  setSteps([]);
                  if (pipelineReplaceMode) setPipelineReplaceMode(false);
                }}
                className="rounded-xl border border-slate-200 px-4 py-2.5 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-50 sm:mr-auto"
              >
                {pipelineReplaceMode ? "Cancel pipeline edit" : "Clear queue"}
              </button>
              <button
                type="button"
                onClick={handleApply}
                disabled={applying}
                className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-blue-600 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-blue-700 disabled:opacity-50 sm:flex-none sm:min-w-[220px]"
              >
                {applying && <Loader2 className="h-4 w-4 animate-spin" />}
                {applying
                  ? "Applying…"
                  : pipelineReplaceMode
                    ? `Replace pipeline (${steps.length} step${steps.length > 1 ? "s" : ""})`
                    : `Apply ${steps.length} step${steps.length > 1 ? "s" : ""}`}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

function stepSummary(step: Record<string, unknown>): string {
  const type = step.type as string;
  const cols = (step.columns as string[] | undefined) ?? [];
  const colStr = cols.length ? cols.join(", ") : "";
  switch (type) {
    case "drop_columns":    return colStr;
    case "rename_columns":  return Object.entries(step.mapping as Record<string,string>).map(([k,v]) => `${k} → ${v}`).join(", ");
    case "impute":          return `${colStr} (${step.strategy})`;
    case "one_hot_encode":  return colStr;
    case "label_encode":    return colStr;
    case "clip_outliers":   return `${colStr} (${step.method})`;
    case "scale":           return `${colStr} (${step.method})`;
    case "math_transform":  return `${colStr} (${step.method})`;
    case "fix_skewness":    return `${colStr} (${step.method}, |skew|≥${step.threshold})`;
    case "bin_numeric":     return `${colStr} (${step.n_bins} bins, ${step.strategy})`;
    case "drop_duplicates": return cols.length ? colStr : "all columns";
    case "drop_nulls":      return `${colStr} (${step.how})`;
    case "frequency_encode":return colStr;
    case "cast_dtype":      return `${colStr} → ${step.dtype}`;
    case "polynomial_features":
      return `${colStr} (deg ${step.degree})`;
    case "extract_datetime":
      return `${colStr} (${(step.parts as string[] | undefined)?.join(",")})`;
    case "pca_projection":
      return `${colStr} (${step.n_components} PC)`;
    case "tfidf_column":
      return String(step.column ?? "");
    case "derive_numeric":
      return `${step.column_a} ${step.op} ${step.column_b} → ${step.output_column}`;
    case "target_encode_dataset":
      return `${colStr} | y=${step.target_column}`;
    case "train_test_split":
      return `y=${step.target_column} · test ${Math.round(Number(step.test_size ?? 0.2) * 100)}%`;
    default:                return colStr;
  }
}

function TransformBanner({
  datasetId,
  history,
  onReverted,
  allColumns,
  onRequestEditPipeline,
}: {
  datasetId: string;
  history: TransformHistory;
  onReverted: () => void;
  allColumns: string[];
  onRequestEditPipeline: (steps: Step[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const [revertingAll, setRevertingAll] = useState(false);
  const [revertingIdx, setRevertingIdx] = useState<number | null>(null);
  const [inlineEditIdx, setInlineEditIdx] = useState<number | null>(null);
  const [inlineEditDraft, setInlineEditDraft] = useState<Step | null>(null);
  const [savingInlineEdit, setSavingInlineEdit] = useState(false);

  const appliedStepsTyped = useMemo(() => {
    if (!history.active || !history.steps.length) return [] as Step[];
    return (history.steps as object[]).map((raw) => deserializeStep(raw as Record<string, unknown>));
  }, [history.active, history.steps]);

  const visible = history.active && history.steps.length > 0;
  const steps = (history.steps ?? []) as Record<string, unknown>[];

  const cancelInlineEdit = () => {
    setInlineEditIdx(null);
    setInlineEditDraft(null);
  };

  const appliedAt = history.applied_at
    ? new Date(history.applied_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
    : null;

  const handleRevertAll = async () => {
    cancelInlineEdit();
    const ok = window.confirm(
      "Revert all transforms back to the original dataset?\n\nThis will remove all applied steps."
    );
    if (!ok) return;
    try {
      setRevertingAll(true);
      await resetTransform(datasetId);
      toast.success("Reverted all transforms to the original dataset.");
      onReverted();
    } finally {
      setRevertingAll(false);
    }
  };

  const handleRevertStep = async (idx: number, e: React.MouseEvent) => {
    e.stopPropagation();
    cancelInlineEdit();
    try {
      setRevertingIdx(idx);
      const remaining = steps.filter((_, i) => i !== idx);
      if (remaining.length === 0) {
        await resetTransform(datasetId);
      } else {
        await applyTransform(datasetId, remaining, true /* from_original */);
      }
      toast.success(`Removed step ${idx + 1}.`);
      onReverted();
    } catch (err) {
      toast.error("Revert failed: " + (err as Error).message);
    } finally {
      setRevertingIdx(null);
    }
  };

  const startInlineEdit = (idx: number) => {
    if (revertingIdx !== null || revertingAll || appliedStepsTyped[idx] == null) return;
    setInlineEditDraft(cloneStepsForEdit([appliedStepsTyped[idx]])[0]);
    setInlineEditIdx(idx);
  };

  const saveInlineEdit = async () => {
    if (inlineEditIdx === null || !inlineEditDraft) return;
    const newPipeline = appliedStepsTyped.map((s, i) => (i === inlineEditIdx ? inlineEditDraft : s));
    try {
      setSavingInlineEdit(true);
      await applyTransform(datasetId, newPipeline.map(serializeStep), true);
      toast.success("Step updated.");
      cancelInlineEdit();
      onReverted();
    } catch (err) {
      toast.error("Save failed: " + (err as Error).message);
    } finally {
      setSavingInlineEdit(false);
    }
  };

  if (!visible) return null;

  return (
    <div className="mb-4 rounded-2xl border border-amber-200/80 bg-gradient-to-b from-amber-50/90 to-amber-50/40 shadow-sm ring-1 ring-amber-100/60">
      <div className="flex flex-wrap items-stretch gap-2 border-b border-amber-200/50 bg-amber-50/90 px-3 py-2.5 sm:px-4">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="flex min-w-0 flex-1 items-center gap-2 rounded-lg py-1 text-left transition-colors hover:bg-amber-100/60 sm:py-0"
        >
          {open ? <ChevronDown className="h-4 w-4 shrink-0 text-amber-700" /> : <ChevronRight className="h-4 w-4 shrink-0 text-amber-700" />}
          <div className="min-w-0">
            <p className="text-sm font-semibold text-amber-950">
              {steps.length} saved step{steps.length !== 1 ? "s" : ""} on this dataset
            </p>
            {appliedAt && <p className="text-xs text-amber-700/80">Last applied {appliedAt}</p>}
          </div>
        </button>
        <button
          type="button"
          onClick={() => {
            toast.info("Quick transform opened with your full pipeline.");
            onRequestEditPipeline(cloneStepsForEdit(appliedStepsTyped));
          }}
          disabled={inlineEditIdx !== null || revertingAll || revertingIdx !== null}
          title="Load all steps into Quick transform to edit the full pipeline"
          className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-xl border border-amber-300 bg-white px-3 py-2 text-xs font-semibold text-amber-900 shadow-sm transition-colors hover:bg-amber-50 disabled:opacity-50"
        >
          <Pencil className="h-3.5 w-3.5" />
          Edit pipeline
        </button>
        <button
          type="button"
          onClick={handleRevertAll}
          disabled={revertingAll}
          className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-xl border border-amber-300 bg-white px-3 py-2 text-xs font-semibold text-amber-900 shadow-sm transition-colors hover:bg-amber-50 disabled:opacity-50"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          {revertingAll ? "Reverting…" : "Revert all"}
        </button>
      </div>
      {open && (
        <div className="border-t border-amber-100/80 bg-white/60 px-3 py-3 sm:px-4">
          <p className="mb-2 text-[11px] leading-snug text-slate-500">
            Steps run in order. Edit a step or remove one — the pipeline is rebuilt from your original upload when you save or remove.
          </p>
          <div className="max-h-[min(28rem,70vh)] space-y-2 overflow-y-auto overflow-x-hidden pr-0.5">
            {steps.map((step, i) => (
              <div key={`${i}-${String(step.type)}`} className="space-y-2">
                {inlineEditIdx === i && inlineEditDraft ? (
                  <>
                    <StepCard
                      step={inlineEditDraft}
                      index={i}
                      allColumns={allColumns}
                      hideRemove
                      onUpdate={(patch) =>
                        setInlineEditDraft((d) => (d ? { ...d, ...patch } as Step : d))
                      }
                      onRemove={() => {}}
                    />
                    <div className="flex flex-wrap gap-2 px-0.5">
                      <button
                        type="button"
                        onClick={saveInlineEdit}
                        disabled={savingInlineEdit}
                        className="rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
                      >
                        {savingInlineEdit ? "Saving…" : "Save step"}
                      </button>
                      <button
                        type="button"
                        onClick={cancelInlineEdit}
                        disabled={savingInlineEdit}
                        className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                      >
                        Cancel
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="flex items-start gap-3 rounded-xl border border-amber-200/60 bg-white px-3 py-2.5 shadow-sm">
                    <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-amber-100 text-[11px] font-bold text-amber-900">
                      {i + 1}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-semibold text-slate-800">
                        {STEP_META[step.type as StepType]?.label ?? String(step.type)}
                      </p>
                      <p className="mt-0.5 break-words text-xs leading-relaxed text-slate-500">{stepSummary(step) || "—"}</p>
                    </div>
                    <div className="flex shrink-0 flex-col gap-1 sm:flex-row sm:items-center">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          startInlineEdit(i);
                        }}
                        disabled={revertingAll || revertingIdx !== null || inlineEditIdx !== null}
                        title="Edit this step"
                        className="inline-flex items-center justify-center gap-1 rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5 text-[11px] font-semibold text-slate-800 transition-colors hover:border-amber-300 hover:bg-amber-50 hover:text-amber-900 disabled:opacity-50"
                      >
                        <Pencil className="h-3 w-3" />
                        Edit
                      </button>
                      <button
                        type="button"
                        onClick={(e) => handleRevertStep(i, e)}
                        disabled={revertingAll || revertingIdx !== null || inlineEditIdx !== null}
                        title="Remove this step and rebuild the remaining pipeline"
                        className="inline-flex items-center justify-center gap-1 rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5 text-[11px] font-semibold text-slate-700 transition-colors hover:border-amber-300 hover:bg-amber-50 hover:text-amber-900 disabled:opacity-50"
                      >
                        <RotateCcw className="h-3 w-3" />
                        {revertingIdx === i ? "…" : "Remove"}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function EDA({ datasetId, transformSyncKey = 0, onTransformsMutated }: Props) {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = searchParams.get("tab") ?? "health";

  const [eda, setEda] = useState<EDAResult | null>(null);
  const [edaOriginal, setEdaOriginal] = useState<EDAResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [transformHistory, setTransformHistory] = useState<TransformHistory | null>(null);
  const [allColumns, setAllColumns] = useState<string[]>([]);
  const [quickQueueSeed, setQuickQueueSeed] = useState<{ key: number; steps: Step[] } | null>(null);
  const clearQuickQueueSeed = useCallback(() => setQuickQueueSeed(null), []);
  const prevDatasetRef = useRef<string | null>(null);

  const fetchEdaBundle = useCallback(() => {
    if (!datasetId) return;
    const dsChanged = prevDatasetRef.current !== datasetId;
    prevDatasetRef.current = datasetId;
    if (dsChanged) setLoading(true);
    getEDA(datasetId).then(setEda).finally(() => { if (dsChanged) setLoading(false); });
    getTransformHistory(datasetId)
      .then(async (h) => {
        setTransformHistory(h);
        if (h.active) {
          try {
            setEdaOriginal(await getEDA(datasetId, "original"));
          } catch {
            setEdaOriginal(null);
          }
        } else {
          setEdaOriginal(null);
        }
      })
      .catch(() => {
        setTransformHistory(null);
        setEdaOriginal(null);
      });
    previewDataset(datasetId, 1).then((r) => setAllColumns(r.columns)).catch(() => null);
  }, [datasetId]);

  useEffect(() => {
    if (!datasetId) {
      prevDatasetRef.current = null;
      return;
    }
    fetchEdaBundle();
  }, [datasetId, transformSyncKey, fetchEdaBundle]);

  const appliedStepsForWarnings = useMemo((): Step[] => {
    if (!transformHistory?.active || !transformHistory.steps?.length) return [];
    return transformHistory.steps.map((raw) => deserializeStep(raw as Record<string, unknown>));
  }, [transformHistory]);

  if (!datasetId) return <PageShell title="EDA"><p className="text-sm text-slate-500">Upload a dataset first.</p></PageShell>;
  if (loading) return (
    <PageShell title="EDA">
      <div className="space-y-3">{[1,2,3].map(i=><Skeleton key={i} className="h-12 w-full"/>)}</div>
    </PageShell>
  );
  if (!eda) return <PageShell title="EDA"><p className="text-sm text-slate-500">No data available.</p></PageShell>;

  const transformActive = Boolean(transformHistory?.active);
  const columnsOriginal = edaOriginal?.columns ?? [];

  return (
    <PageShell
      title="EDA"
      description={`${eda.shape[0].toLocaleString()} rows × ${eda.columns.length} numeric columns`}
    >
      <p className="mb-3 max-w-3xl text-xs leading-relaxed text-slate-500">
        Optional transforms update the <span className="font-medium text-slate-600">working dataset</span> used for every tab below and for training, until you revert.
      </p>
      <QuickTransformPanel
        datasetId={datasetId}
        allColumns={allColumns}
        onApplied={() => onTransformsMutated?.()}
        queueSeed={quickQueueSeed}
        onQueueSeedConsumed={clearQuickQueueSeed}
      />
      {transformHistory && (
        <TransformBanner
          history={transformHistory}
          datasetId={datasetId}
          onReverted={() => onTransformsMutated?.()}
          allColumns={allColumns}
          onRequestEditPipeline={(steps) => setQuickQueueSeed({ key: Date.now(), steps })}
        />
      )}
      <Tabs value={activeTab} onValueChange={(t) => setSearchParams((p) => { p.set("tab", t); return p; }, { replace: true })}>
        <TabsList className="flex-wrap h-auto">
          <TabsTrigger value="health">Health</TabsTrigger>
          <TabsTrigger value="target">Target</TabsTrigger>
          <TabsTrigger value="skewness">Skewness</TabsTrigger>
          <TabsTrigger value="boxplot">Box Plots</TabsTrigger>
          <TabsTrigger value="pairplot">Pairplot</TabsTrigger>
          <TabsTrigger value="feature-target">Feature vs Target</TabsTrigger>
          <TabsTrigger value="stats">Stats</TabsTrigger>
          <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
          <TabsTrigger value="scatter">Scatter</TabsTrigger>
          <TabsTrigger value="missing">Missing</TabsTrigger>
          <TabsTrigger value="outliers">Outliers</TabsTrigger>
          <TabsTrigger value="categorical">Categorical</TabsTrigger>
          <TabsTrigger value="transform-warnings" className="gap-1">
            Transform warnings
          </TabsTrigger>
        </TabsList>

        <TabsContent value="health"><HealthTab datasetId={datasetId} transformActive={transformActive} /></TabsContent>
        <TabsContent value="target">
          <TargetTab
            datasetId={datasetId}
            columns={eda.columns}
            columnsOriginal={columnsOriginal}
            transformActive={transformActive}
          />
        </TabsContent>
        <TabsContent value="skewness"><SkewnessTab datasetId={datasetId} transformActive={transformActive} /></TabsContent>
        <TabsContent value="boxplot">
          <BoxplotTab
            datasetId={datasetId}
            columns={eda.columns}
            columnsOriginal={columnsOriginal}
            transformActive={transformActive}
          />
        </TabsContent>
        <TabsContent value="pairplot">
          <PairplotTab
            datasetId={datasetId}
            columns={eda.columns}
            columnsOriginal={columnsOriginal}
            transformActive={transformActive}
          />
        </TabsContent>
        <TabsContent value="feature-target">
          <FeatureTargetTab
            datasetId={datasetId}
            columns={eda.columns}
            columnsOriginal={columnsOriginal}
            transformActive={transformActive}
          />
        </TabsContent>
        <TabsContent value="stats">
          <StatsTab eda={eda} edaOriginal={edaOriginal} transformActive={transformActive} />
        </TabsContent>
        <TabsContent value="heatmap"><HeatmapTab datasetId={datasetId} transformActive={transformActive} /></TabsContent>
        <TabsContent value="scatter">
          <ScatterTab
            datasetId={datasetId}
            columns={eda.columns}
            columnsOriginal={columnsOriginal}
            transformActive={transformActive}
          />
        </TabsContent>
        <TabsContent value="missing"><MissingTab datasetId={datasetId} transformActive={transformActive} /></TabsContent>
        <TabsContent value="outliers"><OutliersTab datasetId={datasetId} transformActive={transformActive} /></TabsContent>
        <TabsContent value="categorical"><CategoricalTab datasetId={datasetId} transformActive={transformActive} /></TabsContent>
        <TabsContent value="transform-warnings">
          <EdaTransformWarningsTab steps={appliedStepsForWarnings} allColumns={allColumns} />
        </TabsContent>
      </Tabs>
    </PageShell>
  );
}
