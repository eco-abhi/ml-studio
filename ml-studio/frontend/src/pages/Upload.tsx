import { ArrowRight, CloudUpload, Database, Link2, Loader2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  previewDataset,
  uploadFile,
  uploadURL,
  type PreviewResult,
} from "../api";
import { LoadingState } from "../components/LoadingState";
import { PageShell } from "../components/PageShell";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Skeleton } from "../components/ui/skeleton";

interface UploadProps {
  /** Active dataset from app shell (URL / localStorage). */
  currentDatasetId: string | null;
  onClearDataset: () => void;
  onDatasetCreated: (id: string) => void;
  onNavigateToTrain: () => void;
}

export default function Upload({
  currentDatasetId,
  onClearDataset,
  onDatasetCreated,
  onNavigateToTrain,
}: UploadProps) {
  const [file, setFile]           = useState<File | null>(null);
  const [url, setUrl]             = useState("");
  const [preview, setPreview]     = useState<PreviewResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [loadingCurrent, setLoadingCurrent] = useState(false);
  /** Skip effect fetch when we already loaded preview for this id (e.g. just uploaded). */
  const previewSyncedForId = useRef<string | null>(null);

  useEffect(() => {
    if (!currentDatasetId) {
      setPreview(null);
      previewSyncedForId.current = null;
      setLoadingCurrent(false);
      return;
    }
    if (previewSyncedForId.current === currentDatasetId) return;

    let cancelled = false;
    setLoadingCurrent(true);
    previewDataset(currentDatasetId)
      .then((p) => {
        if (cancelled) return;
        if (Array.isArray(p.columns) && p.columns.length > 0) {
          setPreview(p);
          previewSyncedForId.current = currentDatasetId;
        } else {
          setPreview(null);
          previewSyncedForId.current = null;
        }
      })
      .catch(() => {
        if (!cancelled) {
          setPreview(null);
          previewSyncedForId.current = null;
          toast.error("Could not load the current dataset — it may have been removed. Try uploading again.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingCurrent(false);
      });
    return () => {
      cancelled = true;
    };
  }, [currentDatasetId]);

  const handleUpload = async () => {
    if (!file && !url) { toast.error("Select a file or enter a URL"); return; }
    try {
      setUploading(true);
      const result = file ? await uploadFile(file) : await uploadURL(url);
      onDatasetCreated(result.dataset_id);
      const p = await previewDataset(result.dataset_id);
      setPreview(p);
      previewSyncedForId.current = result.dataset_id;
      toast.success(`Uploaded "${result.filename}" — ${result.shape[0].toLocaleString()} rows`);
    } catch (e) {
      toast.error("Upload failed: " + (e as Error).message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <PageShell
      title="Upload Dataset"
      description="Upload a CSV file or link to a public URL to get started."
    >
      <div className="space-y-5">
        {currentDatasetId && (
          <Card className="border-blue-200 bg-gradient-to-br from-blue-50/90 to-white shadow-sm">
            <CardHeader className="pb-2">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="flex items-start gap-3 min-w-0">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-white shadow-md shadow-blue-600/20">
                    <Database className="h-5 w-5" />
                  </div>
                  <div className="min-w-0">
                    <CardTitle className="text-base text-blue-950">Current dataset</CardTitle>
                    <p className="mt-1 font-mono text-xs text-blue-900/80 break-all">{currentDatasetId}</p>
                    {preview && !loadingCurrent && (
                      <p className="mt-2 text-xs text-blue-700">
                        {preview.shape[0].toLocaleString()} rows · {preview.shape[1]} columns ·{" "}
                        {preview.numeric_columns.length} numeric
                      </p>
                    )}
                    {loadingCurrent && !preview && (
                      <div className="mt-3 rounded-lg border border-blue-100 bg-white/70 px-2 py-4">
                        <LoadingState message="Loading dataset preview…" className="!py-4" />
                      </div>
                    )}
                  </div>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={onClearDataset}
                  className="shrink-0 border-blue-200 text-blue-800 hover:bg-blue-100"
                >
                  <X className="h-3.5 w-3.5 mr-1" />
                  Clear selection
                </Button>
              </div>
            </CardHeader>
          </Card>
        )}

        {/* ── Step 1: Upload ── */}
        <Card>
          <CardHeader><CardTitle>1 — Upload or link</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <label className="flex flex-col items-center justify-center gap-2 border-2 border-dashed border-slate-200 rounded-xl p-8 cursor-pointer hover:border-blue-400 hover:bg-blue-50/40 transition-colors group">
              <CloudUpload className="w-8 h-8 text-slate-400 group-hover:text-blue-500 transition-colors" />
              <span className="text-sm text-slate-500 group-hover:text-blue-600">
                {file ? file.name : "Click to select a CSV file"}
              </span>
              <input type="file" accept=".csv" className="hidden" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </label>

            <div className="flex items-center gap-3">
              <hr className="flex-1 border-slate-200" />
              <span className="text-xs text-slate-400 font-medium">OR</span>
              <hr className="flex-1 border-slate-200" />
            </div>

            <div className="relative">
              <Link2 className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="https://example.com/data.csv"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                className="w-full pl-9 pr-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <Button onClick={handleUpload} disabled={uploading} className="w-full">
              {uploading ? <><Loader2 className="w-4 h-4 animate-spin" />Uploading…</> : "Upload"}
            </Button>
          </CardContent>
        </Card>

        {uploading && !preview && (
          <Card>
            <CardContent className="pt-6 space-y-3">
              <Skeleton className="h-4 w-2/3" />
              <Skeleton className="h-4 w-1/2" />
              <Skeleton className="h-24 w-full" />
            </CardContent>
          </Card>
        )}

        {/* ── Step 2: Preview ── */}
        {preview && (
          <Card>
            <CardHeader>
              <CardTitle>2 — Preview</CardTitle>
              <div className="flex gap-2 mt-1">
                <Badge variant="secondary">{preview.shape[0].toLocaleString()} rows</Badge>
                <Badge variant="secondary">{preview.shape[1]} columns</Badge>
                <Badge variant="secondary">{preview.numeric_columns.length} numeric</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="overflow-x-auto rounded-lg border border-slate-200">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200">
                      {preview.columns.map((c) => (
                        <th key={c} className="text-left px-3 py-2 font-medium text-slate-600 whitespace-nowrap">
                          {c}
                          {preview.missing[c] > 0 && <span className="ml-1 text-amber-500">({preview.missing[c]} null)</span>}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.preview.map((row, i) => (
                      <tr key={i} className="border-b border-slate-100 last:border-0 hover:bg-slate-50">
                        {row.map((cell, j) => (
                          <td key={j} className="px-3 py-1.5 text-slate-700 whitespace-nowrap">
                            {typeof cell === "number" ? cell.toFixed(3) : String(cell ?? "—")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* CTA to Train */}
              <div className="flex items-center gap-3 p-4 bg-blue-50 border border-blue-200 rounded-xl">
                <div className="flex-1">
                  <p className="text-sm font-semibold text-blue-900">Dataset ready to train</p>
                  <p className="text-xs text-blue-600 mt-0.5">
                    Go to <strong>Train</strong> to select a target column, configure hyperparameters, and run training.
                  </p>
                </div>
                <Button onClick={onNavigateToTrain} size="sm" className="shrink-0">
                  Train <ArrowRight className="w-3.5 h-3.5" />
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </PageShell>
  );
}
