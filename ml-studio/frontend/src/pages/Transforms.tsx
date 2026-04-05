import { AlertTriangle, BookOpen, ChevronDown, GripVertical, History, Loader2, Pencil, Plus, RotateCcw, Save, Trash2, X } from "lucide-react";
import { Select } from "../components/ui/select";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  applyTransform,
  getHoldoutSplitStatus,
  getTransformHistory,
  previewDataset,
  previewTransform,
  resetTransform,
  type HoldoutSplitStatus,
  type TransformApplyResult,
  type TransformPreview,
} from "../api";
import { DeriveTemplateQueue } from "../components/DeriveTemplateQueue";
import { LoadingState } from "../components/LoadingState";
import { RenameEditor } from "../components/RenameEditor";
import { TransformTypePicker } from "../components/TransformTypePicker";
import { PageShell } from "../components/PageShell";
import { cn } from "../lib/utils";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { TransformsWarningsGuide } from "../transformWarnings";
import {
  type BinStep,
  type CastStep,
  type ClipStep,
  type DatetimePart,
  type DeriveStep,
  type DropDupStep,
  type DropNullStep,
  type FixSkewStep,
  type FreqEncStep,
  type ImputeStep,
  type LabelEncStep,
  type MathStep,
  type OneHotStep,
  type RenameStep,
  type ScaleStep,
  type Step,
  type StepType,
  STEP_META,
  cloneStepsForEdit,
  deserializeStep,
  makeStep,
  serializeStep,
} from "../transformTypes";

const DATETIME_PART_OPTIONS: { value: DatetimePart; label: string }[] = [
  { value: "year", label: "Year" },
  { value: "month", label: "Month" },
  { value: "day", label: "Day" },
  { value: "hour", label: "Hour" },
  { value: "minute", label: "Minute" },
  { value: "dow", label: "Day of week" },
  { value: "doy", label: "Day of year" },
  { value: "week", label: "ISO week" },
];

function _colsPreview(cols: string[], max = 4): string {
  if (!cols.length) return "all columns";
  if (cols.length <= max) return cols.join(", ");
  return `${cols.slice(0, max).join(", ")} +${cols.length - max}`;
}

/** Move item from `fromIndex` to `toIndex` in a new array (order used by the backend pipeline). */
function reorderStepList<T>(list: T[], fromIndex: number, toIndex: number): T[] {
  if (fromIndex === toIndex) return list;
  const next = [...list];
  const [removed] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, removed);
  return next;
}

function stepSummary(step: Step): string {
  switch (step.type) {
    case "rename_columns":
      return Object.entries(step.mapping).map(([a, b]) => `${a}→${b}`).join(", ") || "—";
    case "impute":
      return `${_colsPreview(step.columns)} · ${step.strategy}`;
    case "clip_outliers": {
      const c = step as ClipStep;
      if (c.method === "percentile") {
        const lo = c.p_low ?? 0.01;
        const hi = c.p_high ?? 0.99;
        return `${_colsPreview(step.columns)} · p${Math.round(lo * 100)}–p${Math.round(hi * 100)}`;
      }
      return `${_colsPreview(step.columns)} · ${step.method}`;
    }
    case "scale":
      return `${_colsPreview(step.columns)} · ${step.method}`;
    case "math_transform": {
      const m = step as MathStep;
      const mode = m.output_mode === "new_columns" ? "new cols" : "replace";
      return `${_colsPreview(step.columns)} · ${step.method} · ${mode}`;
    }
    case "fix_skewness":
      return `${_colsPreview(step.columns)} · ${step.method} (thr ${step.threshold})`;
    case "bin_numeric":
      return `${_colsPreview(step.columns)} · ${step.n_bins} bins ${step.strategy}`;
    case "drop_duplicates":
      return `${_colsPreview(step.columns)} · keep ${step.keep}`;
    case "drop_nulls":
      return `${_colsPreview(step.columns)} · ${step.how}`;
    case "frequency_encode":
      return `${_colsPreview(step.columns)} · norm ${step.normalize}`;
    case "cast_dtype":
      return `${_colsPreview(step.columns)} · ${step.dtype}`;
    case "polynomial_features":
      return `${_colsPreview(step.columns)} · deg ${step.degree}${step.interaction_only ? " · interactions" : ""}`;
    case "extract_datetime":
      return `${_colsPreview(step.columns)} · ${(step.parts ?? []).join(",")}`;
    case "pca_projection":
      return `${_colsPreview(step.columns)} · ${step.n_components} comps${step.drop_original ? " · drop src" : ""}`;
    case "tfidf_column":
      return `${step.column || "—"} · ${step.max_features} feats · ngram≤${step.ngram_max}`;
    case "derive_numeric":
      return `${step.column_a || "?"} ${step.op} ${step.column_b || "?"} → ${step.output_column}`;
    case "target_encode_dataset":
      return `${_colsPreview(step.columns)} · target ${step.target_column || "—"}`;
    case "train_test_split":
      return `${step.target_column || "—"} · test ${Math.round((step.test_size ?? 0.2) * 100)}% · seed ${step.random_state ?? 42}${step.stratify ? " · stratify" : ""}`;
    default:
      return _colsPreview("columns" in step ? step.columns : []);
  }
}

interface Props {
  datasetId: string | null;
  /** When this changes, refetch applied pipeline (keeps Transforms in sync with EDA). */
  transformSyncKey?: number;
  /** Call after any successful transform API mutation so the other tab refetches. */
  onTransformsMutated?: () => void;
}

export default function Transforms({ datasetId, transformSyncKey = 0, onTransformsMutated }: Props) {
  // Applied pipeline from server (source of truth)
  const [appliedSteps, setAppliedSteps] = useState<Step[]>([]);
  const [appliedAt, setAppliedAt]       = useState<string | null>(null);
  const [hasActive, setHasActive]       = useState(false);

  // Pending queue (new steps to add on top)
  const [pendingSteps, setPendingSteps] = useState<Step[]>([]);
  const [preview, setPreview]           = useState<TransformPreview | null>(null);
  const [result, setResult]             = useState<TransformApplyResult | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [loadingApply, setLoadingApply]     = useState(false);
  const [loadingReset, setLoadingReset]     = useState(false);
  const [revertingIdx, setRevertingIdx]     = useState<number | null>(null);
  const [addType, setAddType]               = useState<StepType>("impute");
  const [columns, setColumns]               = useState<string[]>([]);
  /** When true, pending steps replace the whole pipeline; preview/apply run from the original CSV. */
  const [pipelineReplaceMode, setPipelineReplaceMode] = useState(false);
  const [inlineEditIdx, setInlineEditIdx]   = useState<number | null>(null);
  const [inlineEditDraft, setInlineEditDraft] = useState<Step | null>(null);
  const [savingInlineEdit, setSavingInlineEdit] = useState(false);
  const [reordering, setReordering] = useState(false);
  const [dragOverIdx, setDragOverIdx] = useState<number | null>(null);
  const [draggingAppliedIdx, setDraggingAppliedIdx] = useState<number | null>(null);
  const dragAppliedFromRef = useRef<number | null>(null);
  const [pipelineReady, setPipelineReady] = useState(false);
  const [holdoutStatus, setHoldoutStatus] = useState<HoldoutSplitStatus | null>(null);

  useEffect(() => { setResult(null); setPreview(null); }, [pendingSteps]);

  const refreshHistory = useCallback(() => {
    if (!datasetId) return;
    setPipelineReady(false);
    Promise.all([
      getTransformHistory(datasetId)
        .then((h) => {
          setHasActive(h.active);
          setAppliedAt(h.applied_at);
          setAppliedSteps(h.steps.map((s) => deserializeStep(s as Record<string, unknown>)));
        })
        .catch(() => {}),
      previewDataset(datasetId, 1)
        .then((r) => setColumns(r.columns))
        .catch(() => setColumns([])),
      getHoldoutSplitStatus(datasetId)
        .then(setHoldoutStatus)
        .catch(() => setHoldoutStatus(null)),
    ]).finally(() => setPipelineReady(true));
  }, [datasetId]);

  useEffect(() => {
    if (!datasetId) {
      setPipelineReady(false);
      setAppliedSteps([]);
      setColumns([]);
      return;
    }
    refreshHistory();
  }, [datasetId, transformSyncKey, refreshHistory]);

  if (!datasetId) {
    return (
      <PageShell
        title="Transforms"
        description="Build a preprocessing pipeline. Apply it before training to produce a cleaned dataset."
      >
        <Tabs defaultValue="pipeline" className="w-full">
          <TabsList className="mb-5 h-auto flex-wrap gap-1 p-1">
            <TabsTrigger value="pipeline" className="gap-1.5">
              <BookOpen className="h-3.5 w-3.5 opacity-70" />
              Pipeline
            </TabsTrigger>
            <TabsTrigger value="warnings" className="gap-1.5">
              <AlertTriangle className="h-3.5 w-3.5 opacity-70" />
              Warnings &amp; reasons
            </TabsTrigger>
          </TabsList>
          <TabsContent value="pipeline" className="mt-0 focus-visible:outline-none">
            <p className="text-sm text-slate-500">Upload a dataset first, then build and apply steps here.</p>
          </TabsContent>
          <TabsContent value="warnings" className="mt-0 focus-visible:outline-none">
            <TransformsWarningsGuide />
          </TabsContent>
        </Tabs>
      </PageShell>
    );
  }

  if (!pipelineReady) {
    return (
      <PageShell
        title="Transforms"
        description="Build a preprocessing pipeline. Apply it before training to produce a cleaned dataset."
      >
        <LoadingState variant="page" message="Loading pipeline and columns…" />
      </PageShell>
    );
  }

  /** New uploads must use the Split tab first; older datasets with a working CSV but no __ml_split__ still work. */
  const transformsNeedSplitFirst = !holdoutStatus?.column_present && !hasActive;

  // ── Pending step CRUD ──
  const addStep   = () => setPendingSteps((p) => [...p, makeStep(addType)]);
  const removeStep = (id: string) => setPendingSteps((p) => p.filter((s) => s.id !== id));
  const updateStep = (id: string, patch: Partial<Step>) =>
    setPendingSteps((p) => p.map((s) => (s.id === id ? { ...s, ...patch } as Step : s)));

  const loadPipelineIntoBuilder = () => {
    if (inlineEditIdx !== null) {
      setInlineEditIdx(null);
      setInlineEditDraft(null);
    }
    if (!appliedSteps.length) {
      toast.error("No applied steps to edit.");
      return;
    }
    setPendingSteps(cloneStepsForEdit(appliedSteps));
    setPipelineReplaceMode(true);
    setPreview(null);
    setResult(null);
    toast.info("Editing full pipeline — preview runs from the original upload.");
  };

  const cancelPipelineReplace = () => {
    setPipelineReplaceMode(false);
    setPendingSteps([]);
    setPreview(null);
  };

  const startInlineEdit = (idx: number) => {
    if (pipelineReplaceMode || revertingIdx !== null) return;
    const s = appliedSteps[idx];
    setInlineEditDraft(cloneStepsForEdit([s])[0]);
    setInlineEditIdx(idx);
  };

  const cancelInlineEdit = () => {
    setInlineEditIdx(null);
    setInlineEditDraft(null);
  };

  const saveInlineEdit = async () => {
    if (inlineEditIdx === null || !inlineEditDraft || !datasetId) return;
    const newPipeline = appliedSteps.map((s, i) => (i === inlineEditIdx ? inlineEditDraft : s));
    try {
      setSavingInlineEdit(true);
      await applyTransform(datasetId, newPipeline.map(serializeStep), true);
      toast.success("Step updated.");
      cancelInlineEdit();
      setPreview(null);
      setResult(null);
      onTransformsMutated?.();
    } catch (e) {
      toast.error("Save failed: " + (e as Error).message);
    } finally {
      setSavingInlineEdit(false);
    }
  };

  // ── API actions ──
  const handlePreview = async () => {
    try {
      setLoadingPreview(true);
      const res = await previewTransform(
        datasetId,
        pendingSteps.map(serializeStep),
        5,
        pipelineReplaceMode
      );
      setPreview(res);
    } catch (e) { toast.error("Preview failed: " + (e as Error).message); }
    finally { setLoadingPreview(false); }
  };

  const handleApply = async () => {
    if (!pendingSteps.length) { toast.error("Add at least one step."); return; }
    try {
      setLoadingApply(true);
      const fromOriginal = pipelineReplaceMode;
      const res = await applyTransform(datasetId, pendingSteps.map(serializeStep), fromOriginal);
      setResult(res);
      setPendingSteps([]);
      setPreview(null);
      if (pipelineReplaceMode) setPipelineReplaceMode(false);
      toast.success(
        fromOriginal
          ? `Pipeline replaced — ${res.shape[0].toLocaleString()} × ${res.shape[1]}.`
          : `Applied — dataset is now ${res.shape[0]} × ${res.shape[1]}.`
      );
      onTransformsMutated?.();
    } catch (e) { toast.error("Apply failed: " + (e as Error).message); }
    finally { setLoadingApply(false); }
  };

  const handleReset = async () => {
    try {
      setLoadingReset(true);
      await resetTransform(datasetId);
      setPendingSteps([]);
      setPreview(null);
      setResult(null);
      setPipelineReplaceMode(false);
      setInlineEditIdx(null);
      setInlineEditDraft(null);
      toast.success("Dataset reset to original.");
      onTransformsMutated?.();
    } catch (e) { toast.error("Reset failed: " + (e as Error).message); }
    finally { setLoadingReset(false); }
  };

  // Remove one step from the applied pipeline by replaying the rest from original
  const handleAppliedReorder = async (fromIdx: number, toIdx: number) => {
    if (fromIdx === toIdx || !datasetId) return;
    const next = reorderStepList(appliedSteps, fromIdx, toIdx);
    try {
      setReordering(true);
      await applyTransform(datasetId, next.map(serializeStep), true);
      toast.success("Pipeline order updated.");
      onTransformsMutated?.();
    } catch (e) {
      toast.error("Reorder failed: " + (e as Error).message);
    } finally {
      setReordering(false);
    }
  };

  const canDragApplied =
    !reordering &&
    !pipelineReplaceMode &&
    inlineEditIdx === null &&
    revertingIdx === null;

  const handleGripDragStart = (idx: number) => (e: React.DragEvent) => {
    if (!canDragApplied) {
      e.preventDefault();
      return;
    }
    dragAppliedFromRef.current = idx;
    setDraggingAppliedIdx(idx);
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", String(idx));
  };

  const handleRowDragOver = (idx: number) => (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    if (dragAppliedFromRef.current !== null) setDragOverIdx(idx);
  };

  const handleRowDragLeave = (e: React.DragEvent) => {
    const rel = e.relatedTarget as Node | null;
    if (!rel || !e.currentTarget.contains(rel)) setDragOverIdx(null);
  };

  const handleRowDrop = (toIdx: number) => (e: React.DragEvent) => {
    e.preventDefault();
    const from = dragAppliedFromRef.current;
    dragAppliedFromRef.current = null;
    setDragOverIdx(null);
    setDraggingAppliedIdx(null);
    if (from === null || from === toIdx) return;
    void handleAppliedReorder(from, toIdx);
  };

  const handleAppliedDragEnd = () => {
    dragAppliedFromRef.current = null;
    setDragOverIdx(null);
    setDraggingAppliedIdx(null);
  };

  const handleRevertStep = async (idx: number) => {
    cancelInlineEdit();
    const remaining = appliedSteps.filter((_, i) => i !== idx).map(serializeStep);
    try {
      setRevertingIdx(idx);
      if (remaining.length === 0) {
        await resetTransform(datasetId);
      } else {
        await applyTransform(datasetId, remaining, true /* from_original */);
      }
      toast.success(`Step ${idx + 1} removed.`);
      onTransformsMutated?.();
    } catch (e) { toast.error("Revert failed: " + (e as Error).message); }
    finally { setRevertingIdx(null); }
  };

  const fmtDate = (s: string | null) =>
    s ? new Date(s).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : null;

  return (
    <PageShell
      title="Transforms"
      description="Build a preprocessing pipeline. Apply it before training to produce a cleaned dataset."
    >
      <Tabs defaultValue="pipeline" className="w-full">
        <TabsList className="mb-5 h-auto flex-wrap gap-1 p-1">
          <TabsTrigger value="pipeline" className="gap-1.5">
            <BookOpen className="h-3.5 w-3.5 opacity-70" />
            Pipeline
          </TabsTrigger>
          <TabsTrigger value="warnings" className="gap-1.5">
            <AlertTriangle className="h-3.5 w-3.5 opacity-70" />
            Warnings &amp; reasons
          </TabsTrigger>
        </TabsList>

        <TabsContent value="pipeline" className="mt-0 focus-visible:outline-none">
      {transformsNeedSplitFirst && (
        <div className="mb-5 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-950">
          <p className="font-semibold">Hold-out split required first</p>
          <p className="mt-1 text-xs leading-relaxed text-amber-900/95">
            Open the <strong>Split</strong> tab and save train/test labels. Transforms run only after{" "}
            <code className="rounded bg-white/80 px-1 text-[11px]">__ml_split__</code> exists on the working dataset.
          </p>
        </div>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-5">
        {/* ── Left ── */}
        <div className="space-y-5">

          {/* ── Applied pipeline (from server) ── */}
          <div>
            <div className="flex items-center gap-2 mb-3 flex-wrap">
              <History className="w-4 h-4 text-amber-500" />
              <h3 className="text-sm font-semibold text-slate-700">Applied pipeline</h3>
              {appliedSteps.length > 0 && (
                <Badge variant="secondary">{appliedSteps.length} step{appliedSteps.length !== 1 ? "s" : ""}</Badge>
              )}
              {appliedSteps.length > 0 && (
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 text-[11px] gap-1"
                  disabled={inlineEditIdx !== null || pipelineReplaceMode || reordering}
                  onClick={loadPipelineIntoBuilder}
                  title={pipelineReplaceMode ? "Finish or cancel builder edit first" : "Load all steps into the builder to edit the full pipeline"}
                >
                  <Pencil className="w-3 h-3" />
                  Edit pipeline
                </Button>
              )}
              {appliedAt && <span className="ml-auto text-[10px] text-slate-400">{fmtDate(appliedAt)}</span>}
            </div>

            {appliedSteps.length === 0 ? (
              <div className="border-2 border-dashed border-slate-200 rounded-xl p-6 text-center">
                <p className="text-sm text-slate-400">No transforms applied yet.</p>
              </div>
            ) : (
              <div className="space-y-2" role="list" aria-label="Applied transform steps">
                {appliedSteps.map((step, idx) => (
                  <div key={`${step.id}-${idx}`} className="space-y-2">
                    {step.type === "train_test_split" && (
                      <div className="rounded-lg border border-dashed border-amber-300 bg-amber-50/90 px-3 py-2.5 text-[11px] text-amber-950 leading-relaxed">
                        <p className="font-semibold text-amber-950">Legacy train/test step</p>
                        <p className="mt-1 text-amber-900/95">
                          New pipelines assign hold-out on the <strong>Split</strong> tab first. This row is from an older saved pipeline; you can remove it after switching.
                        </p>
                      </div>
                    )}
                    {inlineEditIdx === idx && inlineEditDraft ? (
                      <>
                        <StepCard
                          step={inlineEditDraft}
                          index={idx}
                          allColumns={columns}
                          hideRemove
                          onUpdate={(patch) =>
                            setInlineEditDraft((d) => (d ? { ...d, ...patch } as Step : d))
                          }
                          onRemove={() => {}}
                        />
                        <div className="flex flex-wrap gap-2 px-1">
                          <Button type="button" size="sm" onClick={saveInlineEdit} disabled={savingInlineEdit}>
                            {savingInlineEdit ? "Saving…" : "Save step"}
                          </Button>
                          <Button type="button" size="sm" variant="ghost" onClick={cancelInlineEdit} disabled={savingInlineEdit}>
                            Cancel
                          </Button>
                        </div>
                      </>
                    ) : (
                      <div
                        role="listitem"
                        onDragOver={handleRowDragOver(idx)}
                        onDragLeave={handleRowDragLeave}
                        onDrop={handleRowDrop(idx)}
                        className={cn(
                          "flex items-start gap-3 px-4 py-3 rounded-xl border bg-amber-50 transition-shadow",
                          dragOverIdx === idx && draggingAppliedIdx !== null && draggingAppliedIdx !== idx
                            ? "border-blue-400 ring-2 ring-blue-300/60"
                            : "border-amber-200",
                          draggingAppliedIdx === idx && "opacity-60",
                        )}
                      >
                        <div className="flex shrink-0 items-center gap-1">
                          <span
                            title={canDragApplied ? "Drag to reorder" : undefined}
                            draggable={canDragApplied}
                            onDragStart={handleGripDragStart(idx)}
                            onDragEnd={handleAppliedDragEnd}
                            className={cn(
                              "flex h-8 w-8 cursor-grab touch-none items-center justify-center rounded-lg border border-amber-200/80 bg-white text-amber-700 active:cursor-grabbing",
                              !canDragApplied && "cursor-not-allowed opacity-40",
                            )}
                            aria-grabbed={draggingAppliedIdx === idx}
                          >
                            <GripVertical className="h-4 w-4" />
                          </span>
                          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-amber-200 text-amber-800 text-[11px] font-semibold">
                            {idx + 1}
                          </span>
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-semibold text-amber-900">{STEP_META[step.type].label}</p>
                          <p className="text-[10px] text-amber-600 font-mono mt-0.5 truncate">
                            {stepSummary(step)}
                          </p>
                        </div>
                        <div className="shrink-0 flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => startInlineEdit(idx)}
                            disabled={
                              revertingIdx !== null || inlineEditIdx !== null || pipelineReplaceMode || reordering
                            }
                            title="Edit this step"
                            className="flex items-center gap-1 px-2 py-1 rounded-lg border border-amber-300 bg-white text-amber-800 text-[10px] font-medium hover:bg-amber-100 transition-colors disabled:opacity-40"
                          >
                            <Pencil className="w-3 h-3" />
                            Edit
                          </button>
                          <button
                            type="button"
                            onClick={() => handleRevertStep(idx)}
                            disabled={revertingIdx !== null || inlineEditIdx !== null || reordering}
                            title="Remove this step"
                            className="flex items-center gap-1 px-2 py-1 rounded-lg border border-amber-300 bg-white text-amber-700 text-[10px] font-medium hover:bg-amber-100 transition-colors disabled:opacity-40"
                          >
                            {revertingIdx === idx ? <RotateCcw className="w-3 h-3 animate-spin" /> : <X className="w-3 h-3" />}
                            Revert
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
                <Button onClick={handleReset} disabled={loadingReset} variant="ghost" size="sm" className="text-slate-500 w-full mt-1">
                  {loadingReset ? (
                    <><Loader2 className="w-3.5 h-3.5 animate-spin" />Resetting…</>
                  ) : (
                    <><RotateCcw className="w-3.5 h-3.5" />Reset all to original</>
                  )}
                </Button>
              </div>
            )}
          </div>

          {/* ── Pending queue ── */}
          <div>
            {pipelineReplaceMode && (
              <div className="mb-3 rounded-xl border border-violet-200 bg-violet-50 px-4 py-3 text-xs text-violet-900">
                <p className="font-semibold text-violet-950">Replacing the full pipeline</p>
                <p className="mt-1 text-violet-800/90">
                  Preview and apply run from the <strong>original</strong> upload. Saving overwrites all applied steps with the list below.
                </p>
              </div>
            )}
            <div className="flex items-center gap-2 mb-3">
              <Plus className="w-4 h-4 text-blue-500" />
              <h3 className="text-sm font-semibold text-slate-700">
                {pipelineReplaceMode ? "Pipeline draft" : "Add new steps"}
              </h3>
              {pendingSteps.length > 0 && (
                <Badge variant="default">{pendingSteps.length} pending</Badge>
              )}
            </div>

            <Card className="mb-4">
              <CardContent className="pt-4">
                <div className="flex gap-2">
                  <TransformTypePicker
                    value={addType}
                    onChange={setAddType}
                    className="flex-1 min-w-0"
                    disabled={inlineEditIdx !== null}
                  />
                  <Button onClick={addStep} size="sm" disabled={inlineEditIdx !== null}>
                    <Plus className="w-4 h-4" />Add
                  </Button>
                </div>
                <p className="text-xs text-slate-400 mt-2">{STEP_META[addType].description}</p>
              </CardContent>
            </Card>

            <DeriveTemplateQueue
              allColumns={columns}
              disabled={inlineEditIdx !== null || transformsNeedSplitFirst}
              onQueueSteps={(steps) => setPendingSteps((p) => [...p, ...steps])}
            />

            {pendingSteps.length === 0 && (
              <div className="border-2 border-dashed border-blue-100 rounded-xl p-6 text-center">
                <p className="text-sm text-slate-400">Choose a step type above and click Add.</p>
              </div>
            )}

            {pendingSteps.map((step, idx) => (
              <div className="mb-3" key={step.id}>
                <StepCard
                  step={step}
                  index={pipelineReplaceMode ? idx : appliedSteps.length + idx}
                  allColumns={columns}
                  onUpdate={(patch) => updateStep(step.id, patch)}
                  onRemove={() => removeStep(step.id)}
                />
              </div>
            ))}
          </div>
        </div>

        {/* ── Right: Actions + results ── */}
        <div className="space-y-4">
          <Card>
            <CardContent className="pt-5 space-y-3">
              <Button
                onClick={handlePreview}
                disabled={loadingPreview || transformsNeedSplitFirst}
                className="w-full"
                variant="secondary"
              >
                {loadingPreview ? (
                  <><Loader2 className="w-4 h-4 animate-spin" />Loading preview…</>
                ) : (
                  "Preview (5 rows)"
                )}
              </Button>
              <Button
                onClick={handleApply}
                disabled={loadingApply || !pendingSteps.length || inlineEditIdx !== null || transformsNeedSplitFirst}
                className="w-full"
              >
                {loadingApply ? (
                  <><Loader2 className="w-4 h-4 animate-spin" />Applying…</>
                ) : pipelineReplaceMode ? (
                  <><Save className="w-4 h-4" />Replace pipeline</>
                ) : (
                  <><Save className="w-4 h-4" />Apply &amp; Save</>
                )}
              </Button>
              {pipelineReplaceMode && (
                <Button type="button" onClick={cancelPipelineReplace} variant="ghost" size="sm" className="w-full text-slate-600">
                  Cancel pipeline edit
                </Button>
              )}
              {!hasActive && (
                <Button onClick={handleReset} disabled={loadingReset} variant="ghost" size="sm" className="w-full text-slate-500">
                  {loadingReset ? (
                    <><Loader2 className="w-3.5 h-3.5 animate-spin" />Resetting…</>
                  ) : (
                    <><RotateCcw className="w-3.5 h-3.5" />Reset to original</>
                  )}
                </Button>
              )}
            </CardContent>
          </Card>

          {result && (
            <Card className="border-emerald-200 bg-emerald-50">
              <CardContent className="pt-5 space-y-2">
                <p className="text-sm font-semibold text-emerald-800">Applied ✓</p>
                <div className="flex gap-2 flex-wrap">
                  <Badge variant="success">{result.steps_applied} total step{result.steps_applied !== 1 ? "s" : ""}</Badge>
                  <Badge variant="secondary">{result.shape[0].toLocaleString()} rows</Badge>
                  <Badge variant="secondary">{result.shape[1]} cols</Badge>
                </div>
                <p className="text-xs text-emerald-700">Training will use the transformed dataset.</p>
              </CardContent>
            </Card>
          )}

          {pendingSteps.length > 0 && !result && (
            <Card>
              <CardContent className="pt-4">
                <p className="text-xs text-slate-500">
                  <strong className="text-slate-700">{pendingSteps.length}</strong> step{pendingSteps.length !== 1 ? "s" : ""}{" "}
                  {pipelineReplaceMode ? "in draft (replaces pipeline)." : "pending."}{" "}
                  {inlineEditIdx !== null
                    ? "Finish or cancel the step editor above to apply pending steps."
                    : pipelineReplaceMode
                      ? "Preview or Replace pipeline to commit."
                      : "Preview or Apply & Save to commit."}
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* ── Preview table ── */}
      {preview && (
        <div className="mt-5 grid grid-cols-1 xl:grid-cols-2 gap-5">
          <PreviewTable title="Before" data={preview.before} />
          <PreviewTable title="After" data={preview.after} />
        </div>
      )}
        </TabsContent>

        <TabsContent value="warnings" className="mt-0 focus-visible:outline-none">
          <TransformsWarningsGuide />
        </TabsContent>
      </Tabs>
    </PageShell>
  );
}

// ── ColumnPicker ──────────────────────────────────────────────────────────────

interface ColumnPickerProps {
  allColumns: string[];
  selected: string[];
  onChange: (cols: string[]) => void;
  placeholder?: string;
}

function ColumnPicker({ allColumns, selected, onChange, placeholder = "Select columns…" }: ColumnPickerProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const toggle = (col: string) => {
    onChange(selected.includes(col) ? selected.filter((c) => c !== col) : [...selected, col]);
  };

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white hover:border-slate-300 transition-colors"
      >
        <span className={selected.length ? "text-slate-800" : "text-slate-400"}>
          {selected.length ? `${selected.length} column${selected.length > 1 ? "s" : ""} selected` : placeholder}
        </span>
        <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {/* Selected badges */}
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1.5">
          {selected.map((col) => (
            <span
              key={col}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-blue-50 border border-blue-200 text-xs text-blue-700"
            >
              {col}
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); toggle(col); }}
                className="hover:text-blue-900"
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Dropdown */}
      {open && (
        <div className="absolute z-20 mt-1 w-full bg-white border border-slate-200 rounded-lg shadow-lg max-h-52 overflow-y-auto">
          {allColumns.length === 0 && (
            <p className="px-3 py-2 text-xs text-slate-400">No columns available</p>
          )}
          <div className="py-1">
            {/* Select all / none */}
            {allColumns.length > 0 && (
              <div className="flex gap-3 px-3 py-1.5 border-b border-slate-100">
                <button
                  type="button"
                  onClick={() => onChange([...allColumns])}
                  className="text-xs text-blue-600 hover:underline"
                >
                  All
                </button>
                <button
                  type="button"
                  onClick={() => onChange([])}
                  className="text-xs text-slate-500 hover:underline"
                >
                  None
                </button>
              </div>
            )}
            {allColumns.map((col) => (
              <label
                key={col}
                className="flex items-center gap-2 px-3 py-1.5 hover:bg-slate-50 cursor-pointer text-sm text-slate-700"
              >
                <input
                  type="checkbox"
                  checked={selected.includes(col)}
                  onChange={() => toggle(col)}
                  className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                />
                {col}
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── StepCard ─────────────────────────────────────────────────────────────────

export interface StepCardProps {
  step: Step;
  index: number;
  allColumns: string[];
  onUpdate: (patch: Partial<Step>) => void;
  onRemove: () => void;
  /** Hide delete (e.g. inline edit of an applied step). */
  hideRemove?: boolean;
}

export function StepCard({ step, index, allColumns, onUpdate, onRemove, hideRemove }: StepCardProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          {!hideRemove && <GripVertical className="w-4 h-4 text-slate-300 shrink-0" />}
          {hideRemove && <span className="w-4 shrink-0" aria-hidden />}
          <span className="text-xs font-semibold text-slate-400">Step {index + 1}</span>
          <Badge variant="secondary" className="text-[11px]">{STEP_META[step.type].label}</Badge>
          {!hideRemove && (
            <button type="button" onClick={onRemove} className="ml-auto p-1 rounded hover:bg-rose-50 text-slate-400 hover:text-rose-500 transition-colors">
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">

        {/* Column picker — hidden for rename, tfidf (single col), derive (two cols) */}
        {step.type !== "rename_columns" && step.type !== "tfidf_column" && step.type !== "derive_numeric" && step.type !== "train_test_split" && (
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">
              Columns
              {(step.type === "drop_duplicates" || step.type === "drop_nulls") && (
                <span className="text-slate-400 font-normal ml-1">(leave empty = all columns)</span>
              )}
            </label>
            <ColumnPicker
              allColumns={allColumns}
              selected={"columns" in step ? step.columns : []}
              onChange={(cols) => onUpdate({ columns: cols } as Partial<Step>)}
            />
          </div>
        )}

        {/* Step-specific options */}
        {step.type === "impute" && (
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Strategy</label>
            <Select
              value={step.strategy}
              onChange={(v) => onUpdate({ strategy: v as ImputeStep["strategy"] } as Partial<Step>)}
              options={[
                { value: "mean",   label: "Mean" },
                { value: "median", label: "Median" },
                { value: "mode",   label: "Mode (most frequent)" },
                { value: "zero",   label: "Zero" },
              ]}
            />
          </div>
        )}

        {step.type === "clip_outliers" && (
          <div className="space-y-2">
            <label className="block text-xs font-medium text-slate-600 mb-1">Method</label>
            <div className="flex flex-wrap gap-2">
              {(["iqr", "zscore", "percentile"] as ClipStep["method"][]).map((m) => (
                <button
                  key={m}
                  onClick={() => onUpdate({ method: m } as Partial<Step>)}
                  className={`flex-1 min-w-[100px] py-1.5 rounded-lg text-xs border transition-colors ${
                    step.method === m
                      ? "bg-blue-600 border-blue-600 text-white"
                      : "border-slate-200 text-slate-600 hover:border-blue-400"
                  }`}
                >
                  {m === "iqr" ? "IQR (1.5×)" : m === "zscore" ? "Z-score (±3σ)" : "Percentile"}
                </button>
              ))}
            </div>
            {step.method === "percentile" && (
              <div className="flex gap-3 items-end">
                <div className="flex-1">
                  <label className="block text-[11px] text-slate-500 mb-0.5">Low quantile</label>
                  <input
                    type="number"
                    min={0}
                    max={0.5}
                    step={0.005}
                    value={step.p_low ?? 0.01}
                    onChange={(e) =>
                      onUpdate({ p_low: Math.min(0.49, Math.max(0, parseFloat(e.target.value) || 0)) } as Partial<Step>)
                    }
                    className="w-full text-sm border border-slate-200 rounded-lg px-2 py-1.5"
                  />
                </div>
                <div className="flex-1">
                  <label className="block text-[11px] text-slate-500 mb-0.5">High quantile</label>
                  <input
                    type="number"
                    min={0.5}
                    max={1}
                    step={0.005}
                    value={step.p_high ?? 0.99}
                    onChange={(e) =>
                      onUpdate({ p_high: Math.min(1, Math.max(0.51, parseFloat(e.target.value) || 1)) } as Partial<Step>)
                    }
                    className="w-full text-sm border border-slate-200 rounded-lg px-2 py-1.5"
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {step.type === "scale" && (
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Scaler</label>
            <div className="flex gap-2">
              {(["standard", "minmax", "robust"] as ScaleStep["method"][]).map((m) => (
                <button
                  key={m}
                  onClick={() => onUpdate({ method: m } as Partial<Step>)}
                  className={`flex-1 py-1.5 rounded-lg text-xs border transition-colors ${
                    step.method === m
                      ? "bg-blue-600 border-blue-600 text-white"
                      : "border-slate-200 text-slate-600 hover:border-blue-400"
                  }`}
                >
                  {m === "standard" ? "Standard (z)" : m === "minmax" ? "Min-Max" : "Robust"}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Rename columns UI */}
        {step.type === "rename_columns" && (
          <RenameEditor
            allColumns={allColumns}
            mapping={step.mapping}
            onChange={(mapping) => onUpdate({ mapping } as Partial<Step>)}
          />
        )}

        {/* ── New step options ── */}

        {step.type === "fix_skewness" && (
          <>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Method</label>
              <div className="flex flex-wrap gap-2">
                {(["auto", "log1p", "sqrt", "box_cox", "yeo_johnson"] as FixSkewStep["method"][]).map((m) => (
                  <button
                    key={m}
                    onClick={() => onUpdate({ method: m } as Partial<Step>)}
                    className={`px-3 py-1.5 rounded-lg text-xs border transition-colors ${
                      step.method === m
                        ? "bg-blue-600 border-blue-600 text-white"
                        : "border-slate-200 text-slate-600 hover:border-blue-400"
                    }`}
                  >
                    {m === "auto" ? "Auto ✨" : m === "box_cox" ? "Box-Cox" : m === "yeo_johnson" ? "Yeo-Johnson" : m}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <div className="flex items-baseline justify-between mb-1">
                <label className="text-xs font-medium text-slate-600">Skewness threshold</label>
                <input
                  type="number" min={0} max={3} step={0.1} value={step.threshold}
                  onChange={(e) => onUpdate({ threshold: parseFloat(e.target.value) || 0.5 } as Partial<Step>)}
                  className="w-16 text-right text-xs font-semibold border border-slate-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <p className="text-[11px] text-slate-400">Columns with |skewness| below this value are skipped. 0.5 is a common default.</p>
            </div>
            <p className="text-[11px] text-slate-400 bg-slate-50 rounded p-2">
              <strong>Auto</strong> picks log1p for positive-only columns, Yeo-Johnson otherwise.
              Box-Cox requires all values &gt; 0 (a small shift is applied automatically).
            </p>
          </>
        )}

        {step.type === "math_transform" && (
          <div className="space-y-2">
            <label className="block text-xs font-medium text-slate-600 mb-1">Method</label>
            <div className="flex flex-wrap gap-2">
              {(["log1p", "sqrt", "square", "reciprocal", "abs"] as MathStep["method"][]).map((m) => (
                <button
                  key={m}
                  onClick={() => onUpdate({ method: m } as Partial<Step>)}
                  className={`px-3 py-1.5 rounded-lg text-xs border transition-colors ${
                    step.method === m
                      ? "bg-blue-600 border-blue-600 text-white"
                      : "border-slate-200 text-slate-600 hover:border-blue-400"
                  }`}
                >
                  {m === "log1p" ? "log1p" : m === "sqrt" ? "√x" : m === "square" ? "x²" : m === "reciprocal" ? "1/x" : "|x|"}
                </button>
              ))}
            </div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Output</label>
            <div className="flex gap-2">
              {(["replace", "new_columns"] as NonNullable<MathStep["output_mode"]>[]).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => onUpdate({ output_mode: mode } as Partial<Step>)}
                  className={`flex-1 py-1.5 rounded-lg text-xs border transition-colors ${
                    (step.output_mode ?? "replace") === mode
                      ? "bg-slate-800 border-slate-800 text-white"
                      : "border-slate-200 text-slate-600 hover:border-blue-400"
                  }`}
                >
                  {mode === "replace" ? "Replace column" : "New columns"}
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-400">
              {(step.output_mode ?? "replace") === "new_columns"
                ? "Adds columns such as log1p_feat, sqrt_feat (prefix depends on method)."
                : "Overwrites selected numeric columns. log1p and sqrt clip negatives to 0."}
            </p>
          </div>
        )}

        {step.type === "bin_numeric" && (
          <>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Number of bins</label>
              <input
                type="number"
                min={2}
                max={50}
                value={step.n_bins}
                onChange={(e) => onUpdate({ n_bins: Math.max(2, parseInt(e.target.value) || 5) } as Partial<Step>)}
                className="w-24 border border-slate-200 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Strategy</label>
              <div className="flex gap-2">
                {(["equal_width", "quantile"] as BinStep["strategy"][]).map((s) => (
                  <button
                    key={s}
                    onClick={() => onUpdate({ strategy: s } as Partial<Step>)}
                    className={`flex-1 py-1.5 rounded-lg text-xs border transition-colors ${
                      step.strategy === s
                        ? "bg-blue-600 border-blue-600 text-white"
                        : "border-slate-200 text-slate-600 hover:border-blue-400"
                    }`}
                  >
                    {s === "equal_width" ? "Equal width (pd.cut)" : "Quantile (pd.qcut)"}
                  </button>
                ))}
              </div>
            </div>
            <p className="text-xs text-slate-400">Creates a new <code className="text-slate-500">{"{col}_bin"}</code> column with integer bin indices.</p>
          </>
        )}

        {step.type === "drop_duplicates" && (
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Keep</label>
            <div className="flex gap-2">
              {(["first", "last", "none"] as DropDupStep["keep"][]).map((k) => (
                <button
                  key={k}
                  onClick={() => onUpdate({ keep: k } as Partial<Step>)}
                  className={`flex-1 py-1.5 rounded-lg text-xs border transition-colors ${
                    step.keep === k
                      ? "bg-blue-600 border-blue-600 text-white"
                      : "border-slate-200 text-slate-600 hover:border-blue-400"
                  }`}
                >
                  {k === "first" ? "Keep first" : k === "last" ? "Keep last" : "Drop all"}
                </button>
              ))}
            </div>
          </div>
        )}

        {step.type === "drop_nulls" && (
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Drop row if</label>
            <div className="flex gap-2">
              {(["any", "all"] as DropNullStep["how"][]).map((h) => (
                <button
                  key={h}
                  onClick={() => onUpdate({ how: h } as Partial<Step>)}
                  className={`flex-1 py-1.5 rounded-lg text-sm border transition-colors ${
                    step.how === h
                      ? "bg-blue-600 border-blue-600 text-white"
                      : "border-slate-200 text-slate-600 hover:border-blue-400"
                  }`}
                >
                  {h === "any" ? "Any null" : "All null"}
                </button>
              ))}
            </div>
          </div>
        )}

        {step.type === "frequency_encode" && (
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Output</label>
            <div className="flex gap-2">
              <button
                onClick={() => onUpdate({ normalize: true } as Partial<Step>)}
                className={`flex-1 py-1.5 rounded-lg text-xs border transition-colors ${
                  step.normalize ? "bg-blue-600 border-blue-600 text-white" : "border-slate-200 text-slate-600 hover:border-blue-400"
                }`}
              >
                Proportion (0–1)
              </button>
              <button
                onClick={() => onUpdate({ normalize: false } as Partial<Step>)}
                className={`flex-1 py-1.5 rounded-lg text-xs border transition-colors ${
                  !step.normalize ? "bg-blue-600 border-blue-600 text-white" : "border-slate-200 text-slate-600 hover:border-blue-400"
                }`}
              >
                Count
              </button>
            </div>
            <p className="text-xs text-slate-400 mt-1.5">Creates a new <code className="text-slate-500">{"{col}_freq"}</code> column.</p>
          </div>
        )}

        {step.type === "cast_dtype" && (
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Target type</label>
            <div className="flex gap-2">
              {(["float", "int", "str"] as CastStep["dtype"][]).map((d) => (
                <button
                  key={d}
                  onClick={() => onUpdate({ dtype: d } as Partial<Step>)}
                  className={`flex-1 py-1.5 rounded-lg text-sm border transition-colors ${
                    step.dtype === d
                      ? "bg-blue-600 border-blue-600 text-white"
                      : "border-slate-200 text-slate-600 hover:border-blue-400"
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
            <p className="text-xs text-slate-400 mt-1.5">Unparseable values become NaN for float/int.</p>
          </div>
        )}

        {step.type === "polynomial_features" && (
          <>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Degree (2–4)</label>
              <input
                type="number"
                min={2}
                max={4}
                value={step.degree}
                onChange={(e) => onUpdate({ degree: Math.min(4, Math.max(2, parseInt(e.target.value, 10) || 2)) } as Partial<Step>)}
                className="w-20 border border-slate-200 rounded-lg px-2 py-1.5 text-sm"
              />
            </div>
            <label className="flex items-center gap-2 text-xs text-slate-600">
              <input
                type="checkbox"
                checked={step.interaction_only}
                onChange={(e) => onUpdate({ interaction_only: e.target.checked } as Partial<Step>)}
              />
              Interaction terms only (no pure powers)
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-600">
              <input
                type="checkbox"
                checked={step.include_bias}
                onChange={(e) => onUpdate({ include_bias: e.target.checked } as Partial<Step>)}
              />
              Include bias column
            </label>
            <p className="text-xs text-slate-400">Adds columns named <code className="text-slate-500">poly_*</code>. Can explode column count at higher degrees.</p>
          </>
        )}

        {step.type === "extract_datetime" && (
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-2">Parts to extract</label>
            <div className="flex flex-wrap gap-2">
              {DATETIME_PART_OPTIONS.map(({ value, label }) => {
                const parts = step.parts ?? [];
                const on = parts.includes(value);
                return (
                  <button
                    key={value}
                    type="button"
                    onClick={() => {
                      const next = on ? parts.filter((p) => p !== value) : [...parts, value];
                      onUpdate({ parts: next.length ? next : ["year"] } as Partial<Step>);
                    }}
                    className={`rounded-lg border px-2 py-1 text-xs transition-colors ${
                      on ? "border-blue-600 bg-blue-50 text-blue-800" : "border-slate-200 text-slate-600"
                    }`}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {step.type === "pca_projection" && (
          <>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Components</label>
              <input
                type="number"
                min={1}
                max={50}
                value={step.n_components}
                onChange={(e) => onUpdate({ n_components: Math.max(1, parseInt(e.target.value, 10) || 3) } as Partial<Step>)}
                className="w-24 border border-slate-200 rounded-lg px-2 py-1.5 text-sm"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Column prefix</label>
              <input
                type="text"
                value={step.prefix}
                onChange={(e) => onUpdate({ prefix: e.target.value || "PC_" } as Partial<Step>)}
                className="w-full border border-slate-200 rounded-lg px-2 py-1.5 text-sm"
              />
            </div>
            <label className="flex items-center gap-2 text-xs text-slate-600">
              <input
                type="checkbox"
                checked={step.drop_original}
                onChange={(e) => onUpdate({ drop_original: e.target.checked } as Partial<Step>)}
              />
              Drop original numeric columns used for PCA
            </label>
          </>
        )}

        {step.type === "tfidf_column" && (
          <>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Text column</label>
              <Select
                value={step.column || allColumns[0] || ""}
                onChange={(v) => onUpdate({ column: v } as Partial<Step>)}
                options={allColumns.map((c) => ({ value: c, label: c }))}
              />
            </div>
            <div className="flex gap-3 flex-wrap">
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Max features</label>
                <input
                  type="number"
                  min={5}
                  max={200}
                  value={step.max_features}
                  onChange={(e) => onUpdate({ max_features: Math.min(200, Math.max(5, parseInt(e.target.value, 10) || 50)) } as Partial<Step>)}
                  className="w-24 border border-slate-200 rounded-lg px-2 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">N-gram max</label>
                <input
                  type="number"
                  min={1}
                  max={3}
                  value={step.ngram_max}
                  onChange={(e) => onUpdate({ ngram_max: Math.min(3, Math.max(1, parseInt(e.target.value, 10) || 1)) } as Partial<Step>)}
                  className="w-20 border border-slate-200 rounded-lg px-2 py-1.5 text-sm"
                />
              </div>
            </div>
            <p className="text-xs text-slate-400">Adds dense columns <code className="text-slate-500">tfidf_&#123;col&#125;_0…</code></p>
          </>
        )}

        {step.type === "derive_numeric" && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Column A</label>
                <Select
                  value={step.column_a || allColumns[0] || ""}
                  onChange={(v) => onUpdate({ column_a: v } as Partial<Step>)}
                  options={allColumns.map((c) => ({ value: c, label: c }))}
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 mb-1">Column B</label>
                <Select
                  value={step.column_b || allColumns[0] || ""}
                  onChange={(v) => onUpdate({ column_b: v } as Partial<Step>)}
                  options={allColumns.map((c) => ({ value: c, label: c }))}
                />
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Operation</label>
              <div className="flex flex-wrap gap-2">
                {(["add", "subtract", "multiply", "divide"] as DeriveStep["op"][]).map((op) => (
                  <button
                    key={op}
                    type="button"
                    onClick={() => onUpdate({ op } as Partial<Step>)}
                    className={`rounded-lg border px-3 py-1 text-xs ${
                      step.op === op ? "border-blue-600 bg-blue-600 text-white" : "border-slate-200 text-slate-600"
                    }`}
                  >
                    {op === "add" ? "+" : op === "subtract" ? "−" : op === "multiply" ? "×" : "÷"}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Output column name</label>
              <input
                type="text"
                value={step.output_column}
                onChange={(e) => onUpdate({ output_column: e.target.value || "derived_feature" } as Partial<Step>)}
                className="w-full border border-slate-200 rounded-lg px-2 py-1.5 text-sm"
              />
            </div>
          </>
        )}

        {step.type === "target_encode_dataset" && (
          <>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Target column (must exist in data)</label>
              <Select
                value={step.target_column || allColumns[0] || ""}
                onChange={(v) => onUpdate({ target_column: v } as Partial<Step>)}
                options={allColumns.map((c) => ({ value: c, label: c }))}
              />
            </div>
            <p className="text-xs text-amber-700 bg-amber-50 border border-amber-100 rounded-lg p-2">
              Uses the full dataset to compute category means — can inflate metrics if you train on the same file. For modeling, prefer <strong>Train → Categorical encoding → Target encode (CV-safe)</strong>.
            </p>
          </>
        )}

        {step.type === "train_test_split" && (
          <>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Target column (for stratification; use the same column when training)</label>
              <Select
                value={step.target_column || allColumns[0] || ""}
                onChange={(v) => onUpdate({ target_column: v } as Partial<Step>)}
                options={allColumns.map((c) => ({ value: c, label: c }))}
              />
            </div>
            <div>
              <div className="flex items-baseline justify-between mb-1">
                <label className="text-xs font-medium text-slate-600">Test size</label>
                <span className="text-[10px] text-slate-400">
                  {Math.round((1 - step.test_size) * 100)}% train · {Math.round(step.test_size * 100)}% test
                </span>
              </div>
              <input
                type="range"
                min={0.05}
                max={0.5}
                step={0.05}
                value={step.test_size}
                onChange={(e) => onUpdate({ test_size: parseFloat(e.target.value) } as Partial<Step>)}
                className="w-full"
              />
            </div>
            <div className="flex items-baseline justify-between">
              <label className="text-xs font-medium text-slate-600">Random state</label>
              <input
                type="number"
                min={0}
                max={99999}
                step={1}
                value={step.random_state}
                onChange={(e) => onUpdate({ random_state: parseInt(e.target.value, 10) || 42 } as Partial<Step>)}
                className="w-20 text-right text-xs font-semibold border border-slate-200 rounded px-2 py-1"
              />
            </div>
            <div className="flex gap-3">
              <label className="flex items-center gap-2 text-xs text-slate-600">
                <input
                  type="checkbox"
                  checked={step.shuffle}
                  onChange={(e) => onUpdate({ shuffle: e.target.checked } as Partial<Step>)}
                />
                Shuffle
              </label>
              <label className="flex items-center gap-2 text-xs text-slate-600">
                <input
                  type="checkbox"
                  checked={step.stratify}
                  onChange={(e) => onUpdate({ stratify: e.target.checked } as Partial<Step>)}
                />
                Stratify (classification-friendly targets)
              </label>
            </div>
            <p className="text-xs text-amber-900 bg-amber-50 border border-amber-100 rounded-lg p-2">
              <strong>Legacy step</strong> — new workflows use the <strong>Split</strong> tab (before Transforms). This block remains so older saved pipelines still load. Prefer removing it after migrating.
            </p>
          </>
        )}

        {/* Helper text */}
        {step.type === "one_hot_encode" && (
          <p className="text-xs text-slate-400">Creates binary columns per category. Target column will be excluded automatically.</p>
        )}
        {step.type === "label_encode" && (
          <p className="text-xs text-slate-400">Maps each unique value to an integer. Use for ordinal variables or tree models.</p>
        )}
        {step.type === "drop_columns" && (
          <p className="text-xs text-slate-400">Columns will be permanently removed from the working dataset.</p>
        )}
      </CardContent>
    </Card>
  );
}

// ── PreviewTable ─────────────────────────────────────────────────────────────

interface PreviewTableProps {
  title: string;
  data: { columns: string[]; rows: unknown[][] };
}

function PreviewTable({ title, data }: PreviewTableProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto rounded-b-xl">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200">
                {data.columns.map((c) => (
                  <th key={c} className="text-left px-3 py-2 font-medium text-slate-600 whitespace-nowrap">{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row, i) => (
                <tr key={i} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                  {row.map((cell, j) => (
                    <td key={j} className="px-3 py-1.5 text-slate-700 whitespace-nowrap">
                      {typeof cell === "number"
                        ? Number.isInteger(cell) ? String(cell) : cell.toFixed(3)
                        : String(cell ?? "—")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
