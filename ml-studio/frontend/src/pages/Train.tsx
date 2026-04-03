import { ChevronDown, ChevronRight, Loader2, RefreshCw, Zap } from "lucide-react";
import { Select } from "../components/ui/select";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  previewDataset,
  trainModels,
  tuneModel,
  type PreviewResult,
  type SplitConfig,
  type TaskType,
  type TrainResult,
  type TuneResult,
} from "../api";
import {
  CLASSIFICATION_MODELS,
  HyperparamPanel,
  MODEL_META,
  REGRESSION_MODELS,
  type ParamValues,
} from "../components/HyperparamPanel";
import {
  DEFAULT_SPLIT_CONFIG,
  SplitConfigPanel,
  VAL_STRATEGIES,
} from "../components/SplitConfigPanel";
import { ML_PIPELINE_SPLIT_COLUMN } from "../transformTypes";
import { PageShell } from "../components/PageShell";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Skeleton } from "../components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";

interface Props {
  datasetId: string | null;
  /** Called after a successful train or Optuna tune so Experiments can refetch runs. */
  onTrainingComplete?: () => void;
}

type Hyperparams = Record<string, ParamValues>;

// ── Collapsible section ───────────────────────────────────────────────────────

function Section({ title, subtitle, defaultOpen = true, children }: {
  title: string; subtitle?: string; defaultOpen?: boolean; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-slate-200 rounded-xl">
      <button
        onClick={() => setOpen((o) => !o)}
        className={`w-full flex items-center gap-3 px-4 py-3 text-left bg-slate-50 hover:bg-slate-100 transition-colors ${open ? "rounded-t-xl" : "rounded-xl"}`}
      >
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-slate-800">{title}</p>
          {subtitle && <p className="text-xs text-slate-500 mt-0.5 truncate">{subtitle}</p>}
        </div>
        {open
          ? <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
          : <ChevronRight className="w-4 h-4 text-slate-400 shrink-0" />}
      </button>
      {open && <div className="p-4 bg-white rounded-b-xl">{children}</div>}
    </div>
  );
}

// ── Metric badge helper ───────────────────────────────────────────────────────

function MetricBadge({ label, value }: { label: string; value: number | undefined | null }) {
  if (value == null) return null;
  return (
    <div className="text-center">
      <p className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</p>
      <p className="text-lg font-bold tabular-nums text-slate-900">{value}</p>
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────

export default function Train({ datasetId, onTrainingComplete }: Props) {
  const [preview, setPreview]               = useState<PreviewResult | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [targetCol, setTargetCol]           = useState("");
  const [taskType, setTaskType]             = useState<TaskType | null>(null);
  const [hyperparams, setHyperparams]       = useState<Hyperparams>({});
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [splitConfig, setSplitConfig]       = useState<SplitConfig>(DEFAULT_SPLIT_CONFIG);
  const [training, setTraining]             = useState(false);
  const [trainResult, setTrainResult]       = useState<TrainResult | null>(null);
  const [runCount, setRunCount]             = useState(0);

  // Optuna state
  const [tuneModel_, setTuneModel_]         = useState("");
  const [nTrials, setNTrials]               = useState(30);
  const [tuning, setTuning]                 = useState(false);
  const [tuneResult, setTuneResult]         = useState<TuneResult | null>(null);

  const sharedModels = CLASSIFICATION_MODELS.filter((m) => REGRESSION_MODELS.includes(m));
  const allModels =
    taskType === "classification" ? CLASSIFICATION_MODELS
    : taskType === "regression"   ? REGRESSION_MODELS
    : Array.from(new Set([...REGRESSION_MODELS, ...CLASSIFICATION_MODELS]));

  const effectiveTuneModels = taskType === null ? sharedModels : allModels;

  useEffect(() => {
    setSelectedModels(new Set(allModels));
    // In auto mode we must only offer models that exist for both detected tasks,
    // otherwise Optuna would tune a model that the backend prunes.
    setTuneModel_((effectiveTuneModels[0] ?? allModels[0] ?? "") as string);
  }, [taskType]); // eslint-disable-line react-hooks/exhaustive-deps

  const loadPreview = (id: string) => {
    setLoadingPreview(true);
    previewDataset(id)
      .then((data) => {
        // Guard against error responses like { detail: "..." }
        if (Array.isArray(data.columns) && data.columns.length > 0) {
          setPreview(data);
        } else {
          setPreview(null);
          toast.error("Dataset not found on server. Please re-upload your file.");
        }
      })
      .catch(() => toast.error("Could not reach the backend. Is it running?"))
      .finally(() => setLoadingPreview(false));
  };

  useEffect(() => {
    if (!datasetId) return;
    loadPreview(datasetId);
  }, [datasetId]);

  useEffect(() => { setTrainResult(null); setTuneResult(null); }, [targetCol, taskType]);

  const setParam = (model: string, key: string, val: number | string) =>
    setHyperparams((p) => ({ ...p, [model]: { ...p[model], [key]: val } }));

  const toggleModel = (key: string) =>
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(key)) { if (next.size === 1) return prev; next.delete(key); }
      else next.add(key);
      return next;
    });

  const handleTrain = async () => {
    if (!targetCol || !datasetId) { toast.error("Select a target column first."); return; }
    if (taskType !== null && !selectedModels.size) { toast.error("Select at least one model."); return; }
    try {
      setTraining(true);
      // When taskType is "auto" (null), don't risk sending incompatible hyperparams or
      // filtering by a model list that assumes the wrong task. Backend will detect the task type.
      const modelsToSend = taskType === null ? [] : [...selectedModels];
      const hyperparamsToSend = taskType === null ? {} : hyperparams;
      const result = await trainModels(datasetId, targetCol, taskType, hyperparamsToSend, modelsToSend, splitConfig);
      setTrainResult(result);
      setRunCount((c) => c + 1);
      onTrainingComplete?.();
      toast.success(`Run #${runCount + 1} complete — ${result.results.length} model${result.results.length !== 1 ? "s" : ""} trained.`);
    } catch (e) {
      toast.error("Training failed: " + (e as Error).message);
    } finally {
      setTraining(false);
    }
  };

  const handleTune = async () => {
    if (!targetCol || !datasetId) { toast.error("Select a target column first."); return; }
    if (!tuneModel_) { toast.error("Select a model to tune."); return; }
    try {
      setTuning(true);
      setTuneResult(null);
      const result = await tuneModel(datasetId, targetCol, tuneModel_, nTrials, taskType, splitConfig);
      setTuneResult(result);
      onTrainingComplete?.();
      toast.success(`Optuna: best ${result.metric_name} = ${result.best_value} after ${result.n_trials} trials.`);
    } catch (e) {
      toast.error("Tuning failed: " + (e as Error).message);
    } finally {
      setTuning(false);
    }
  };

  if (!datasetId) {
    return (
      <PageShell title="Train">
        <p className="text-sm text-slate-500">Upload a dataset first, then come back here to train.</p>
      </PageShell>
    );
  }

  if (loadingPreview) {
    return (
      <PageShell title="Train">
        <div className="space-y-3">{[1,2,3].map((i) => <Skeleton key={i} className="h-12 w-full" />)}</div>
      </PageShell>
    );
  }

  if (!preview) {
    return (
      <PageShell title="Train">
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-5 space-y-3">
          <p className="text-sm font-medium text-amber-800">Could not load dataset columns.</p>
          <p className="text-xs text-amber-700">
            The dataset may have been lost after a backend restart. Try re-uploading your file, or click retry if the backend just started.
          </p>
          <Button variant="outline" size="sm" onClick={() => datasetId && loadPreview(datasetId)}>
            Retry
          </Button>
        </div>
      </PageShell>
    );
  }

  const valLabel = VAL_STRATEGIES.find((s) => s.value === splitConfig.val_strategy)?.label ?? splitConfig.val_strategy;
  const hasPipelineHoldout = (preview?.columns ?? []).includes(ML_PIPELINE_SPLIT_COLUMN);
  const splitSectionSubtitle = hasPipelineHoldout
    ? `Holdout from Transforms (${ML_PIPELINE_SPLIT_COLUMN}) · ${valLabel} · ${splitConfig.scaler}`
    : `${Math.round((1 - splitConfig.test_size) * 100)}% train · ${Math.round(splitConfig.test_size * 100)}% test · ${valLabel} · ${splitConfig.scaler}`;

  return (
    <PageShell
      title="Train"
      description="Configure target, splits, models, and hyperparameters, then run."
      action={runCount > 0 ? <Badge variant="secondary">{runCount} run{runCount > 1 ? "s" : ""}</Badge> : undefined}
    >
      <Tabs defaultValue="manual">
        <TabsList>
          <TabsTrigger value="manual">Manual Training</TabsTrigger>
          <TabsTrigger value="optuna">
            <Zap className="w-3.5 h-3.5 mr-1.5" />Optuna Tuning
          </TabsTrigger>
        </TabsList>

        {/* ── Manual Training tab ── */}
        <TabsContent value="manual">
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-5 mt-1">

            {/* Left column — config */}
            <div className="space-y-3">

              {/* Target & task type */}
              <Section title="Target column" subtitle={targetCol ? `Predicting: ${targetCol}` : "Select what to predict"}>
                <div className="space-y-4">
                  <div>
                    <label className="block text-xs font-medium text-slate-600 mb-1.5">
                      Column to predict
                    </label>
                    <Select
                      value={targetCol}
                      onChange={setTargetCol}
                      options={[
                        { value: "", label: "— select —" },
                        ...(preview?.columns ?? []).map((c) => ({
                          value: c,
                          label: preview?.numeric_columns?.includes(c) ? c : `${c} (text)`,
                        })),
                      ]}
                    />
                    {preview && (
                      <div className="flex gap-2 mt-2">
                        <Badge variant="secondary">{preview.shape[0].toLocaleString()} rows</Badge>
                        <Badge variant="secondary">{preview.shape[1]} columns</Badge>
                      </div>
                    )}
                  </div>

                  {targetCol && (
                    <div>
                      <label className="block text-xs font-medium text-slate-600 mb-1.5">
                        Task type <span className="text-slate-400">(auto-detected)</span>
                      </label>
                      <div className="flex gap-2">
                        {(["regression", "classification"] as TaskType[]).map((t) => (
                          <button
                            key={t}
                            onClick={() => setTaskType((prev) => prev === t ? null : t)}
                            className={`flex-1 py-1.5 rounded-lg text-sm border transition-colors ${
                              taskType === t
                                ? "bg-blue-600 border-blue-600 text-white"
                                : "border-slate-200 text-slate-600 hover:border-blue-400"
                            }`}
                          >
                            {t.charAt(0).toUpperCase() + t.slice(1)}
                          </button>
                        ))}
                      </div>
                      {taskType === null && (
                        <p className="mt-2 text-[11px] text-slate-500 leading-relaxed">
                          Auto mode: model selection + hyperparameters are applied safely by the backend after it detects the task type.
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </Section>

              {targetCol && (
                <Section
                  title="Split & Validation"
                  subtitle={splitSectionSubtitle}
                  defaultOpen={false}
                >
                  <SplitConfigPanel
                    config={splitConfig}
                    onChange={(patch) => setSplitConfig((p) => ({ ...p, ...patch }))}
                    taskType={taskType}
                    hideHoldoutSplit={hasPipelineHoldout}
                  />
                </Section>
              )}

              {targetCol && (
                <Section
                  title="Models & Hyperparameters"
                  subtitle={`${selectedModels.size} / ${allModels.length} selected`}
                  defaultOpen={false}
                >
                  <div className="space-y-2">
                    <div className="flex gap-3 pb-1 border-b border-slate-100">
                      <button onClick={() => setSelectedModels(new Set(allModels))} className="text-xs text-blue-600 hover:underline">All</button>
                      <button onClick={() => setSelectedModels(new Set([allModels[0]]))} className="text-xs text-slate-500 hover:underline">None</button>
                      <span className="ml-auto text-xs text-slate-400">{selectedModels.size} / {allModels.length}</span>
                    </div>
                    {allModels.map((modelKey) => (
                      <HyperparamPanel
                        key={modelKey}
                        modelKey={modelKey}
                        taskType={taskType}
                        selected={selectedModels.has(modelKey)}
                        onToggleSelect={() => toggleModel(modelKey)}
                        values={hyperparams[modelKey] ?? {}}
                        onChange={(key, val) => setParam(modelKey, key, val)}
                      />
                    ))}
                  </div>
                </Section>
              )}
            </div>

            {/* Right column — run + results */}
            <div className="space-y-4">
              {targetCol && (
                <Card>
                  <CardContent className="pt-4 space-y-3">
                    <div className="text-xs text-slate-500 leading-relaxed">
                      Training <strong className="text-slate-700">{selectedModels.size}</strong> model{selectedModels.size !== 1 ? "s" : ""}:{" "}
                      <span className="text-slate-600">{[...selectedModels].map((m) => MODEL_META[m]?.label ?? m).join(", ")}.</span>
                      {runCount > 0 && <span className="block mt-1 text-slate-400">Each run creates a new MLflow entry.</span>}
                    </div>
                    <Button
                      onClick={handleTrain}
                      disabled={training || !selectedModels.size}
                      size="lg"
                      className="w-full"
                    >
                      {training
                        ? <><Loader2 className="w-4 h-4 animate-spin" />Training…</>
                        : runCount > 0
                          ? <><RefreshCw className="w-4 h-4" />Retrain (Run #{runCount + 1})</>
                          : "Start Training"}
                    </Button>
                  </CardContent>
                </Card>
              )}

              {trainResult && (
                <Card>
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <CardTitle className="text-sm">Run #{runCount} Results</CardTitle>
                      <Badge variant="success">✓</Badge>
                    </div>
                    <div className="flex gap-1.5 flex-wrap mt-1">
                      <Badge variant="default">{trainResult.task_type}</Badge>
                      <Badge variant="secondary">{trainResult.n_features} features</Badge>
                      <Badge variant="secondary">{trainResult.n_train}↑ / {trainResult.n_test}↓</Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="overflow-x-auto rounded-lg border border-slate-200">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="bg-slate-50 border-b border-slate-200">
                            <th className="text-left px-3 py-2 font-semibold text-slate-600">Model</th>
                            {trainResult.task_type === "regression" ? (
                              <>
                                <th className="text-center px-2 py-2 font-semibold text-slate-600">RMSE</th>
                                <th className="text-center px-2 py-2 font-semibold text-slate-600">R²</th>
                                {trainResult.results[0]?.cv_rmse != null && <th className="text-center px-2 py-2 font-semibold text-slate-600">CV</th>}
                              </>
                            ) : (
                              <>
                                <th className="text-center px-2 py-2 font-semibold text-slate-600">Acc</th>
                                <th className="text-center px-2 py-2 font-semibold text-slate-600">F1</th>
                                {trainResult.results[0]?.cv_accuracy != null && <th className="text-center px-2 py-2 font-semibold text-slate-600">CV</th>}
                              </>
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {trainResult.results.map((r, i) => (
                            <tr key={r.model} className={`border-b border-slate-100 last:border-0 ${i === 0 ? "bg-emerald-50" : "hover:bg-slate-50"}`}>
                              <td className="px-3 py-2 font-medium text-slate-800">
                                {i === 0 && <span className="mr-1 text-emerald-600">★</span>}
                                {r.model.replace(/_/g, " ")}
                              </td>
                              {trainResult.task_type === "regression" ? (
                                <>
                                  <td className="px-2 py-2 text-center tabular-nums">{r.rmse}</td>
                                  <td className="px-2 py-2 text-center tabular-nums">{r.r2}</td>
                                  {r.cv_rmse != null && <td className="px-2 py-2 text-center tabular-nums text-slate-500">{r.cv_rmse}</td>}
                                </>
                              ) : (
                                <>
                                  <td className="px-2 py-2 text-center tabular-nums">{r.accuracy}</td>
                                  <td className="px-2 py-2 text-center tabular-nums">{r.f1_score}</td>
                                  {r.cv_accuracy != null && <td className="px-2 py-2 text-center tabular-nums text-slate-500">{r.cv_accuracy}</td>}
                                </>
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        {/* ── Optuna Tuning tab ── */}
        <TabsContent value="optuna">
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-5 mt-1">

            {/* Left — config */}
            <div className="space-y-3">
              <Section title="Target column" subtitle={targetCol ? `Predicting: ${targetCol}` : "Select what to predict"}>
                <div className="space-y-4">
                  <div>
                    <label className="block text-xs font-medium text-slate-600 mb-1.5">Column to predict</label>
                    <Select
                      value={targetCol}
                      onChange={setTargetCol}
                      options={[
                        { value: "", label: "— select —" },
                        ...(preview?.columns ?? []).map((c) => ({
                          value: c,
                          label: preview?.numeric_columns?.includes(c) ? c : `${c} (text)`,
                        })),
                      ]}
                    />
                  </div>
                  {targetCol && (
                    <div>
                      <label className="block text-xs font-medium text-slate-600 mb-1.5">Task type</label>
                      <div className="flex gap-2">
                        {(["regression", "classification"] as TaskType[]).map((t) => (
                          <button key={t} onClick={() => setTaskType((prev) => prev === t ? null : t)}
                            className={`flex-1 py-1.5 rounded-lg text-sm border transition-colors ${taskType === t ? "bg-blue-600 border-blue-600 text-white" : "border-slate-200 text-slate-600 hover:border-blue-400"}`}>
                            {t.charAt(0).toUpperCase() + t.slice(1)}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </Section>

              {targetCol && (
                <Section title="Tuning Configuration">
                  <div className="space-y-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-600 mb-1.5">Model to tune</label>
                      <Select
                        value={tuneModel_}
                        onChange={setTuneModel_}
                        options={effectiveTuneModels.map((m) => ({ value: m, label: MODEL_META[m]?.label ?? m }))}
                      />
                    </div>
                    <div>
                      <div className="flex items-baseline justify-between mb-1">
                        <label className="text-xs font-medium text-slate-600">Number of trials</label>
                        <input
                          type="number" value={nTrials} min={5} max={200} step={5}
                          onChange={(e) => setNTrials(Math.max(5, parseInt(e.target.value) || 30))}
                          className="w-20 text-right text-xs font-semibold border border-slate-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                        />
                      </div>
                      <input type="range" min={5} max={200} step={5} value={nTrials}
                        onChange={(e) => setNTrials(parseInt(e.target.value))}
                        className="w-full" />
                      <p className="text-[11px] text-slate-400 mt-1">More trials = better search, but longer wait. 30–50 is a good default.</p>
                    </div>
                    <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                      <p className="text-xs font-medium text-blue-800 mb-1">What Optuna does</p>
                      <p className="text-xs text-blue-700 leading-relaxed">
                        Uses Tree-structured Parzen Estimator (TPE) to intelligently explore the hyperparameter search space, finding better configs than grid search in far fewer trials.
                      </p>
                    </div>
                  </div>
                </Section>
              )}

              {targetCol && (
                <Section
                  title="Split & Validation"
                  subtitle={hasPipelineHoldout ? `Holdout from Transforms (${ML_PIPELINE_SPLIT_COLUMN})` : `${Math.round((1 - splitConfig.test_size) * 100)}% train · ${Math.round(splitConfig.test_size * 100)}% test`}
                  defaultOpen={false}
                >
                  <SplitConfigPanel
                    config={splitConfig}
                    onChange={(patch) => setSplitConfig((p) => ({ ...p, ...patch }))}
                    taskType={taskType}
                    hideHoldoutSplit={hasPipelineHoldout}
                  />
                </Section>
              )}
            </div>

            {/* Right — run + results */}
            <div className="space-y-4">
              {targetCol && (
                <Card>
                  <CardContent className="pt-4 space-y-3">
                    <p className="text-xs text-slate-500">
                      Optuna will run <strong className="text-slate-700">{nTrials}</strong> trials on{" "}
                      <strong className="text-slate-700">{MODEL_META[tuneModel_]?.label ?? tuneModel_}</strong> and log the best run to MLflow.
                    </p>
                    <Button onClick={handleTune} disabled={tuning || !tuneModel_} size="lg" className="w-full">
                      {tuning
                        ? <><Loader2 className="w-4 h-4 animate-spin" />Tuning ({nTrials} trials)…</>
                        : <><Zap className="w-4 h-4" />Run Optuna Tuning</>}
                    </Button>
                  </CardContent>
                </Card>
              )}

              {tuneResult && (
                <Card>
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <CardTitle className="text-sm">Best Result</CardTitle>
                      <Badge variant="success">✓</Badge>
                    </div>
                    <div className="flex gap-1.5 flex-wrap mt-1">
                      <Badge variant="default">{tuneResult.model.replace(/_/g, " ")}</Badge>
                      <Badge variant="secondary">{tuneResult.n_trials} trials</Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-0 space-y-4">
                    {/* Best metrics */}
                    <div className="grid grid-cols-3 gap-2">
                      {Object.entries(tuneResult.metrics).map(([k, v]) => (
                        <div key={k} className="text-center p-2 bg-emerald-50 rounded-lg border border-emerald-200">
                          <p className="text-[10px] text-slate-500 uppercase tracking-wide">{k.replace(/_/g, " ")}</p>
                          <p className="text-base font-bold tabular-nums text-emerald-800">{v}</p>
                        </div>
                      ))}
                    </div>

                    {/* Best params */}
                    <div>
                      <p className="text-xs font-semibold text-slate-600 mb-2">Best hyperparameters</p>
                      <div className="space-y-1">
                        {Object.entries(tuneResult.best_params).map(([k, v]) => (
                          <div key={k} className="flex justify-between items-center text-xs py-1 border-b border-slate-100 last:border-0">
                            <span className="text-slate-600 font-mono">{k}</span>
                            <span className="font-semibold text-slate-900 tabular-nums">{typeof v === "number" ? v.toFixed(4).replace(/\.?0+$/, "") : v}</span>
                          </div>
                        ))}
                        {Object.keys(tuneResult.best_params).length === 0 && (
                          <p className="text-xs text-slate-400">No tunable parameters for this model.</p>
                        )}
                      </div>
                    </div>

                    {/* Trial convergence chart */}
                    {tuneResult.trials.length > 1 && (() => {
                      const vals = tuneResult.trials.map((t) => t.value);
                      const isMin = tuneResult.metric_name === "rmse";
                      const best: number[] = [];
                      let running = isMin ? Infinity : -Infinity;
                      vals.forEach((v) => {
                        if (isMin ? v < running : v > running) running = v;
                        best.push(running);
                      });
                      const allVals = [...vals, ...best];
                      const minV = Math.min(...allVals), maxV = Math.max(...allVals);
                      const range = maxV - minV || 1;
                      const W = 280, H = 100, PAD = 8;
                      const sx = (i: number) => PAD + (i / (vals.length - 1)) * (W - PAD * 2);
                      const sy = (v: number) => H - PAD - ((v - minV) / range) * (H - PAD * 2);
                      const trialPts = vals.map((v, i) => `${sx(i)},${sy(v)}`).join(" ");
                      const bestPts  = best.map((v, i) => `${sx(i)},${sy(v)}`).join(" ");
                      return (
                        <div>
                          <p className="text-xs font-semibold text-slate-600 mb-1.5">Convergence — {tuneResult.metric_name}</p>
                          <svg width={W} height={H} className="w-full border border-slate-100 rounded-lg bg-slate-50">
                            <polyline points={trialPts} fill="none" stroke="#94a3b8" strokeWidth={1} />
                            <polyline points={bestPts}  fill="none" stroke="#10b981" strokeWidth={2} />
                          </svg>
                          <div className="flex gap-3 mt-1 text-[10px]">
                            <span className="flex items-center gap-1"><span className="w-3 h-0.5 inline-block bg-slate-400" />Each trial</span>
                            <span className="flex items-center gap-1"><span className="w-3 h-0.5 inline-block bg-emerald-500" />Best so far</span>
                          </div>
                        </div>
                      );
                    })()}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </PageShell>
  );
}
