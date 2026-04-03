import { AlertTriangle } from "lucide-react";
import { useMemo, useState } from "react";
import { Badge } from "./components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { cn } from "./lib/utils";
import {
  ML_PIPELINE_SPLIT_COLUMN,
  STEP_META,
  type Step,
} from "./transformTypes";
import { TRANSFORM_WARNINGS, type TransformWarningItem } from "./transformWarnings";

/** Whether this step is considered to affect `col` for EDA warning attribution (best-effort). */
export function stepTouchesColumn(step: Step, col: string): boolean {
  switch (step.type) {
    case "rename_columns": {
      const m = step.mapping;
      return Object.entries(m).some(([from, to]) => col === from || col === to);
    }
    case "tfidf_column": {
      const base = step.column;
      if (!base) return false;
      return col === base || col.startsWith(`tfidf_${base}_`);
    }
    case "derive_numeric":
      return col === step.output_column || col === step.column_a || col === step.column_b;
    case "train_test_split":
      if (col === ML_PIPELINE_SPLIT_COLUMN) return true;
      return Boolean(step.target_column && col === step.target_column);
    case "target_encode_dataset": {
      if (step.columns.includes(col)) return true;
      return step.columns.some((c) => col === `${c}_tgtenc`);
    }
    case "bin_numeric": {
      if (step.columns.includes(col)) return true;
      return step.columns.some((c) => col === `${c}_bin`);
    }
    case "frequency_encode": {
      if (step.columns.includes(col)) return true;
      return step.columns.some((c) => col === `${c}_freq`);
    }
    case "polynomial_features": {
      if (step.columns.includes(col)) return true;
      return col.startsWith("poly_");
    }
    case "extract_datetime": {
      if (step.columns.includes(col)) return true;
      return step.columns.some(
        (c) => col !== c && (col.startsWith(`${c}_`) || col.startsWith(`${c}__`)),
      );
    }
    case "pca_projection": {
      if (step.columns.includes(col)) return true;
      const p = (step.prefix || "PC_").trim();
      return p ? col.startsWith(p) : false;
    }
    case "drop_duplicates":
    case "drop_nulls":
      if (!step.columns.length) return true;
      return step.columns.includes(col);
    default:
      if ("columns" in step && Array.isArray(step.columns)) {
        return step.columns.includes(col);
      }
      return false;
  }
}

/** Short human-readable scope string for a step (columns touched or derived). */
export function formatStepColumnScope(step: Step): string {
  switch (step.type) {
    case "rename_columns": {
      const e = Object.entries(step.mapping);
      if (!e.length) return "—";
      const head = e.slice(0, 3).map(([a, b]) => `${a}→${b}`).join(", ");
      return e.length > 3 ? `${head} +${e.length - 3}` : head;
    }
    case "tfidf_column":
      return step.column ? `${step.column} → tfidf_*` : "—";
    case "derive_numeric":
      return `${step.column_a || "?"} ${step.op} ${step.column_b || "?"} → ${step.output_column || "?"}`;
    case "train_test_split":
      return `target ${step.target_column || "—"} · ${ML_PIPELINE_SPLIT_COLUMN}`;
    case "target_encode_dataset":
      return step.columns.length
        ? `${step.columns.slice(0, 3).join(", ")}${step.columns.length > 3 ? ` +${step.columns.length - 3}` : ""} → *_tgtenc`
        : "—";
    case "polynomial_features":
      return step.columns.length
        ? `${step.columns.slice(0, 3).join(", ")}${step.columns.length > 3 ? ` +${step.columns.length - 3}` : ""} → poly_*`
        : "—";
    case "extract_datetime":
      return step.columns.length
        ? `${step.columns.slice(0, 3).join(", ")} → *_year, *_month, …`
        : "—";
    case "pca_projection":
      return step.columns.length
        ? `${step.columns.slice(0, 3).join(", ")}${step.columns.length > 3 ? ` +${step.columns.length - 3}` : ""} → ${(step.prefix || "PC_").trim()}*`
        : "—";
    case "drop_duplicates":
    case "drop_nulls":
      return step.columns.length ? step.columns.join(", ") : "All columns (row filter)";
    default:
      if ("columns" in step && Array.isArray(step.columns)) {
        const c = step.columns;
        if (!c.length) return "—";
        if (c.length <= 4) return c.join(", ");
        return `${c.slice(0, 4).join(", ")} +${c.length - 4}`;
      }
      return "—";
  }
}

function WarningBlock({ w }: { w: TransformWarningItem }) {
  return (
    <div className="rounded-lg border border-amber-200/90 bg-amber-50/80 px-3 py-2.5">
      <div className="flex gap-2">
        <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" aria-hidden />
        <div className="min-w-0">
          <p className="text-[11px] font-semibold text-amber-950">{w.title}</p>
          <p className="mt-1 text-[11px] leading-relaxed text-amber-950/85">{w.reason}</p>
        </div>
      </div>
    </div>
  );
}

export function EdaTransformWarningsTab({ steps, allColumns }: { steps: Step[]; allColumns: string[] }) {
  const [colOpen, setColOpen] = useState<string | null>(null);

  const byColumn = useMemo(() => {
    const sorted = [...allColumns].sort((a, b) => a.localeCompare(b));
    return sorted.map((col) => {
      const hits = steps
        .map((step, idx) => ({ step, idx }))
        .filter(({ step }) => stepTouchesColumn(step, col));
      return { col, hits };
    });
  }, [allColumns, steps]);

  if (!steps.length) {
    return (
      <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50/50 px-5 py-8 text-center">
        <p className="text-sm text-slate-600">No transforms applied yet.</p>
        <p className="mt-2 text-xs text-slate-500">
          After you apply steps on Transforms or Quick transform, this tab lists cautions for each step and each column they may affect.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <p className="max-w-3xl text-xs leading-relaxed text-slate-600">
        Warnings mirror the Transforms reference: each <strong className="text-slate-800">pipeline step</strong> has general cautions, and{" "}
        <strong className="text-slate-800">by column</strong> repeats the same notes for every column that step likely touched (heuristic; renamed or derived names may differ).
      </p>

      <Tabs defaultValue="pipeline" className="w-full">
        <TabsList className="mb-4 h-auto w-full flex-wrap justify-start gap-1 p-1 sm:w-auto">
          <TabsTrigger value="pipeline" className="text-xs sm:text-sm">
            By pipeline step
          </TabsTrigger>
          <TabsTrigger value="columns" className="text-xs sm:text-sm">
            By column ({allColumns.length})
          </TabsTrigger>
        </TabsList>

        <TabsContent value="pipeline" className="mt-0 space-y-4 outline-none">
          {steps.map((step, i) => {
            const warnings = TRANSFORM_WARNINGS[step.type];
            return (
              <Card key={`${step.id}-${i}`} className="border-slate-200/90 shadow-sm">
                <CardHeader className="space-y-2 pb-3 pt-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="secondary" className="font-mono text-[10px]">
                      {i + 1}
                    </Badge>
                    <CardTitle className="text-sm font-semibold text-slate-900">{STEP_META[step.type].label}</CardTitle>
                  </div>
                  <p className="text-[11px] leading-relaxed text-slate-500">
                    <span className="font-medium text-slate-600">Columns / scope:</span>{" "}
                    <code className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-800">{formatStepColumnScope(step)}</code>
                  </p>
                </CardHeader>
                <CardContent className="space-y-2 pb-4 pt-0">
                  {warnings.map((w, j) => (
                    <WarningBlock key={j} w={w} />
                  ))}
                </CardContent>
              </Card>
            );
          })}
        </TabsContent>

        <TabsContent value="columns" className="mt-0 outline-none">
          <div className="max-h-[min(70vh,720px)] space-y-2 overflow-y-auto pr-1">
            {byColumn.map(({ col, hits }) => {
              const open = colOpen === col;
              const totalWarnings = hits.reduce((n, { step }) => n + TRANSFORM_WARNINGS[step.type].length, 0);
              return (
                <div key={col} className="rounded-xl border border-slate-200 bg-white shadow-sm">
                  <button
                    type="button"
                    onClick={() => setColOpen((c) => (c === col ? null : col))}
                    className={cn(
                      "flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition-colors",
                      open ? "bg-slate-50" : "hover:bg-slate-50/80",
                    )}
                  >
                    <div className="min-w-0">
                      <code className="text-xs font-semibold text-slate-900">{col}</code>
                      <p className="mt-0.5 text-[10px] text-slate-500">
                        {hits.length} step{hits.length !== 1 ? "s" : ""} · {totalWarnings} warning{totalWarnings !== 1 ? "s" : ""}
                      </p>
                    </div>
                    <span className="shrink-0 text-slate-400">{open ? "−" : "+"}</span>
                  </button>
                  {open && (
                    <div className="space-y-4 border-t border-slate-100 px-4 py-3">
                      {hits.length === 0 ? (
                        <p className="text-xs text-slate-500">No specific step matched this column name (it may only be affected indirectly).</p>
                      ) : (
                        hits.map(({ step, idx }) => (
                          <div key={`${col}-${step.id}-${idx}`} className="space-y-2">
                            <div className="flex flex-wrap items-center gap-2">
                              <Badge variant="secondary" className="font-mono text-[10px]">
                                Step {idx + 1}
                              </Badge>
                              <span className="text-xs font-medium text-slate-800">{STEP_META[step.type].label}</span>
                            </div>
                            <p className="text-[10px] text-slate-500">
                              Scope: <code className="text-slate-700">{formatStepColumnScope(step)}</code>
                            </p>
                            <div className="space-y-2">
                              {TRANSFORM_WARNINGS[step.type].map((w, j) => (
                                <WarningBlock key={j} w={w} />
                              ))}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
