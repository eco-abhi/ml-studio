import { Check, Copy, Download, Loader2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  downloadTrainedPipeline,
  getEDA,
  getExperiments,
  listDatasets,
  type ExperimentRun,
} from "../api";
import { formatRunLabel } from "../lib/runLabel";
import { LoadingState } from "../components/LoadingState";
import { PageShell } from "../components/PageShell";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Select } from "../components/ui/select";

interface Props {
  datasetId: string | null;
  experimentsSyncKey?: number;
}

function buildLoadSnippet(
  modelFileBase: string,
  featureCols: string[],
  sampleRow: Record<string, number | string>,
  truncated: boolean,
) {
  const dictLines = featureCols
    .map((c) => {
      const v = sampleRow[c];
      if (typeof v === "number" && Number.isFinite(v)) return `    "${c}": ${v},`;
      if (typeof v === "string") return `    "${c}": "${v.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}",`;
      return `    "${c}": 0,`;
    })
    .join("\n");
  const tail = truncated
    ? "\n# … include all training feature columns; names match the saved Pipeline.\n"
    : "";
  return `import joblib
import pandas as pd

# File from ML Studio (gzip-compressed joblib)
pipe = joblib.load("${modelFileBase}_pipeline.joblib")

# One row: same inputs as training (no target column; no __ml_split__)
X = pd.DataFrame([{
${dictLines}
}])${tail}
y_hat = pipe.predict(X)
print(y_hat)

# Classifiers with predict_proba:
# proba = pipe.predict_proba(X)
`;
}

export default function PipelineExport({ datasetId, experimentsSyncKey = 0 }: Props) {
  const [runs, setRuns] = useState<ExperimentRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [featureCols, setFeatureCols] = useState<string[]>([]);
  const [snippetColsTruncated, setSnippetColsTruncated] = useState(false);
  const [sampleRow, setSampleRow] = useState<Record<string, number | string>>({});
  const [copied, setCopied] = useState(false);

  const SPLIT_COL = "__ml_split__";
  const MAX_SNIPPET_FEATURES = 14;

  useEffect(() => {
    if (!datasetId) return;
    setLoadingRuns(true);
    Promise.all([getExperiments(datasetId), getEDA(datasetId), listDatasets()])
      .then(([exp, eda, dsList]) => {
        setRuns(exp.runs);
        setSelectedRunId((id) => {
          if (id && exp.runs.some((r) => r.run_id === id)) return id;
          return exp.runs[0]?.run_id ?? "";
        });

        const meta = (dsList.datasets as { id?: string; target_col?: string; features?: string[] }[]).find(
          (d) => d.id === datasetId,
        );
        const target = meta?.target_col;
        const trainedFeatures = Array.isArray(meta?.features) && meta.features.length > 0 ? meta.features : null;

        let cols: string[];
        if (trainedFeatures) {
          cols = trainedFeatures;
        } else {
          cols = eda.columns.filter((c) => c !== target && c !== SPLIT_COL && eda.stats[c]);
        }

        const truncated = cols.length > MAX_SNIPPET_FEATURES;
        const snippetList = truncated ? cols.slice(0, MAX_SNIPPET_FEATURES) : cols;
        setFeatureCols(snippetList);
        setSnippetColsTruncated(truncated);

        const row: Record<string, number | string> = {};
        snippetList.forEach((c) => {
          const s = eda.stats[c];
          if (s) row[c] = s.mean;
          else row[c] = 0;
        });
        setSampleRow(row);
      })
      .catch(() => toast.error("Could not load dataset or experiments."))
      .finally(() => setLoadingRuns(false));
  }, [datasetId, experimentsSyncKey]);

  const selectedRun = runs.find((r) => r.run_id === selectedRunId);
  const model = selectedRun?.model ?? "";

  const safeBase = model.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_|_$/g, "") || "pipeline";

  const loadSnippet = useMemo(
    () => buildLoadSnippet(safeBase, featureCols, sampleRow, snippetColsTruncated),
    [safeBase, featureCols, sampleRow, snippetColsTruncated],
  );

  const handleDownload = async () => {
    if (!datasetId || !model) return;
    try {
      setDownloading(true);
      await downloadTrainedPipeline(datasetId, model, selectedRunId);
      toast.success("Pipeline downloaded (.joblib)");
    } catch (e) {
      toast.error((e as Error).message || "Download failed");
    } finally {
      setDownloading(false);
    }
  };

  const copySnippet = () => {
    void navigator.clipboard.writeText(loadSnippet).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    });
  };

  if (!datasetId) {
    return (
      <PageShell title="Pipeline export">
        <p className="text-sm text-slate-500">Upload a dataset and train a model first.</p>
      </PageShell>
    );
  }

  if (loadingRuns) {
    return (
      <PageShell
        title="Pipeline export"
        description="Download the trained sklearn Pipeline as joblib and run it locally."
      >
        <LoadingState variant="page" message="Loading…" />
      </PageShell>
    );
  }

  if (!runs.length) {
    return (
      <PageShell
        title="Pipeline export"
        description="Download the trained sklearn Pipeline as joblib and run it locally."
      >
        <p className="text-sm text-slate-500">No trained models yet — use the Train tab first.</p>
      </PageShell>
    );
  }

  return (
    <PageShell
      title="Pipeline export"
      description="Same artifact MLflow stores: gzip-compressed joblib. Use a Python env with the same sklearn and dependencies as training."
    >
      <div className="mx-auto max-w-3xl space-y-6">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Download</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="mb-1.5 block text-xs font-medium text-slate-600">Run</label>
              <Select
                value={selectedRunId}
                onChange={setSelectedRunId}
                options={runs.map((r) => ({
                  value: r.run_id,
                  label: formatRunLabel(r),
                }))}
              />
            </div>
            <Button
              type="button"
              size="lg"
              className="w-full sm:w-auto gap-2"
              disabled={!model || !selectedRunId || downloading}
              onClick={() => void handleDownload()}
            >
              {downloading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Preparing download…
                </>
              ) : (
                <>
                  <Download className="h-4 w-4" />
                  Download {safeBase}_pipeline.joblib
                </>
              )}
            </Button>
            <p className="text-xs text-slate-500 leading-relaxed">
              Browser saves the file locally. Filename matches the button label. If download fails, confirm this dataset
              has a target and the model was trained in ML Studio.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
            <div>
              <CardTitle className="text-base">Load in Python</CardTitle>
              <p className="mt-1 text-xs text-slate-500 font-normal">
                joblib auto-decompresses gzip. Match package versions to training when possible.
              </p>
            </div>
            <Button type="button" variant="outline" size="sm" className="shrink-0 gap-1.5" onClick={copySnippet}>
              {copied ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
              {copied ? "Copied" : "Copy"}
            </Button>
          </CardHeader>
          <CardContent>
            <pre className="max-h-[min(420px,55vh)] overflow-auto rounded-lg border border-slate-200 bg-slate-950 p-4 text-[11px] leading-relaxed text-slate-100 font-mono whitespace-pre">
              {loadSnippet}
            </pre>
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-slate-50/80">
          <CardContent className="pt-4 text-xs text-slate-600 leading-relaxed space-y-2">
            <p>
              <strong className="text-slate-800">Features:</strong> The example row uses EDA means for numeric columns.
              Your production table must use the same column names and types the pipeline saw after preprocessing (categorical
              encoding, etc. is inside the saved Pipeline).
            </p>
            <p>
              <strong className="text-slate-800">Diagnostics:</strong> You can also download from the Diagnostics tab
              while inspecting plots.
            </p>
          </CardContent>
        </Card>
      </div>
    </PageShell>
  );
}
