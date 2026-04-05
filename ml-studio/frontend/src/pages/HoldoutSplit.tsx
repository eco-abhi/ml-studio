import { Loader2, SplitSquareHorizontal } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { getHoldoutSplitStatus, getTransformHistory, previewDataset, saveHoldoutSplit, type HoldoutSplitStatus } from "../api";
import { LoadingState } from "../components/LoadingState";
import { PageShell } from "../components/PageShell";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Select } from "../components/ui/select";
import { ML_PIPELINE_SPLIT_COLUMN } from "../transformTypes";

interface Props {
  datasetId: string | null;
  onHoldoutSaved?: () => void;
}

export default function HoldoutSplit({ datasetId, onHoldoutSaved }: Props) {
  const [columns, setColumns] = useState<string[]>([]);
  const [status, setStatus] = useState<HoldoutSplitStatus | null>(null);
  const [loading, setLoading] = useState(Boolean(datasetId));
  const [saving, setSaving] = useState(false);

  const [targetColumn, setTargetColumn] = useState("");
  const [testSize, setTestSize] = useState(0.2);
  const [randomState, setRandomState] = useState(42);
  const [shuffle, setShuffle] = useState(true);
  const [stratify, setStratify] = useState(true);

  const refresh = useCallback(() => {
    if (!datasetId) return;
    setLoading(true);
    Promise.all([previewDataset(datasetId, 1), getHoldoutSplitStatus(datasetId)])
      .then(([prev, st]) => {
        setColumns(prev.columns ?? []);
        setStatus(st);
        const cfg = st.config;
        if (cfg) {
          setTargetColumn((t) => (t && prev.columns.includes(t) ? t : cfg.target_column));
          setTestSize(cfg.test_size);
          setRandomState(cfg.random_state);
          setShuffle(cfg.shuffle);
          setStratify(cfg.stratify);
        } else {
          setTargetColumn((t) => (t && prev.columns.includes(t) ? t : prev.columns[prev.columns.length - 1] ?? ""));
        }
      })
      .catch(() => toast.error("Could not load dataset."))
      .finally(() => setLoading(false));
  }, [datasetId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleSave = async () => {
    if (!datasetId) return;
    if (!targetColumn) {
      toast.error("Select a target column.");
      return;
    }
    const hist = await getTransformHistory(datasetId).catch(() => null);
    if (hist?.active && hist.steps?.length) {
      const ok = window.confirm(
        "Saving the hold-out split rebuilds the working CSV from your original upload and clears all transform steps. Continue?",
      );
      if (!ok) return;
    }
    setSaving(true);
    try {
      const res = await saveHoldoutSplit(datasetId, {
        target_column: targetColumn,
        test_size: testSize,
        random_state: randomState,
        shuffle,
        stratify,
      });
      toast.success(
        `Hold-out saved — ${res.counts.train.toLocaleString()} train · ${res.counts.test.toLocaleString()} test rows.`,
      );
      onHoldoutSaved?.();
      refresh();
    } catch (e) {
      toast.error((e as Error).message || "Save failed");
    } finally {
      setSaving(false);
    }
  };

  if (!datasetId) {
    return (
      <PageShell title="Hold-out split">
        <p className="text-sm text-slate-500">Upload a dataset first.</p>
      </PageShell>
    );
  }

  if (loading && !status) {
    return (
      <PageShell title="Hold-out split" description="Assign train vs test on the raw upload before preprocessing.">
        <LoadingState variant="page" message="Loading…" />
      </PageShell>
    );
  }

  return (
    <PageShell
      title="Hold-out split"
      description="Label each row as train or test on a copy of your original CSV. This runs before Transforms so preprocessing steps see a fixed evaluation set. Clears the transform pipeline when you save."
    >
      <div className="mb-6 flex items-start gap-3 rounded-xl border border-blue-200 bg-blue-50/80 px-4 py-3 text-sm text-blue-950">
        <SplitSquareHorizontal className="mt-0.5 h-5 w-5 shrink-0 text-blue-600" />
        <div>
          <p className="font-semibold text-blue-950">Required before Transforms</p>
          <p className="mt-1 text-xs leading-relaxed text-blue-900/90">
            The working dataset gets a <code className="rounded bg-white/90 px-1 text-[11px]">{ML_PIPELINE_SPLIT_COLUMN}</code> column.
            The Train tab uses these labels and skips its own random hold-out when the column is still present.
          </p>
        </div>
      </div>

      {status?.column_present && status.counts && (
        <Card className="mb-6 border-emerald-200 bg-emerald-50/50">
          <CardContent className="pt-4 text-sm text-emerald-900">
            <p>
              Current working data includes the hold-out:{" "}
              <strong className="tabular-nums">{status.counts.train.toLocaleString()}</strong> train ·{" "}
              <strong className="tabular-nums">{status.counts.test.toLocaleString()}</strong> test.
            </p>
            <p className="mt-1 text-xs text-emerald-800/90">Change options below and save again to reassign (transform steps will be cleared).</p>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Split options</CardTitle>
          <p className="text-xs text-slate-500">Uses sklearn train_test_split on row indices with a non-null target.</p>
        </CardHeader>
        <CardContent className="space-y-4 max-w-lg">
          <div>
            <label className="mb-1 block text-xs font-medium text-slate-600">Target column</label>
            <Select
              value={targetColumn}
              onChange={setTargetColumn}
              options={columns.map((c) => ({ value: c, label: c }))}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-600">Test fraction</label>
              <input
                type="number"
                min={0.05}
                max={0.95}
                step={0.05}
                value={testSize}
                onChange={(e) => setTestSize(Math.min(0.95, Math.max(0.05, parseFloat(e.target.value) || 0.2)))}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-600">Random state</label>
              <input
                type="number"
                value={randomState}
                onChange={(e) => setRandomState(parseInt(e.target.value, 10) || 0)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm"
              />
            </div>
          </div>
          <label className="flex items-center gap-2 text-sm text-slate-700">
            <input type="checkbox" checked={shuffle} onChange={(e) => setShuffle(e.target.checked)} />
            Shuffle before split
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-700">
            <input type="checkbox" checked={stratify} onChange={(e) => setStratify(e.target.checked)} />
            Stratify (classification-friendly; auto-disabled for regression-like targets)
          </label>
          <Button type="button" onClick={() => void handleSave()} disabled={saving || !targetColumn} className="gap-2">
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            {saving ? "Saving…" : "Save hold-out split"}
          </Button>
        </CardContent>
      </Card>
    </PageShell>
  );
}
