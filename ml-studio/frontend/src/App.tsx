import {
  Activity,
  BarChart2,
  Cpu,
  Database,
  FileDown,
  FlaskConical,
  HelpCircle,
  Layers,
  SlidersHorizontal,
  Sparkles,
  SplitSquareHorizontal,
  Upload,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { InfoDrawer } from "./components/InfoDrawer";
import { SidebarWorkflowNav } from "./components/SidebarWorkflowNav";
import { WorkflowNextBar } from "./components/WorkflowNextBar";
import EDA from "./pages/EDA";
import Experiments from "./pages/Experiments";
import Diagnostics from "./pages/Diagnostics";
import Importances from "./pages/Importances";
import PipelineExport from "./pages/PipelineExport";
import Predict from "./pages/Predict";
import Train from "./pages/Train";
import HoldoutSplit from "./pages/HoldoutSplit";
import Transforms from "./pages/Transforms";
import UploadPage from "./pages/Upload";

type Page =
  | "upload"
  | "split"
  | "eda"
  | "transforms"
  | "train"
  | "experiments"
  | "importances"
  | "diagnostics"
  | "export"
  | "predict";

const NAV: { key: Page; path: string; label: string; icon: React.ElementType; group: string }[] = [
  { key: "upload",      path: "/",             label: "Upload",      icon: Upload,            group: "Data"   },
  { key: "split",       path: "/split",        label: "Split",       icon: SplitSquareHorizontal, group: "Data" },
  { key: "eda",         path: "/eda",           label: "EDA",         icon: BarChart2,         group: "Data"   },
  { key: "transforms",  path: "/transforms",   label: "Transforms",  icon: SlidersHorizontal, group: "Data"   },
  { key: "train",       path: "/train",         label: "Train",       icon: Cpu,               group: "Models" },
  { key: "experiments", path: "/experiments",   label: "Experiments", icon: FlaskConical,      group: "Models" },
  { key: "importances", path: "/importances",   label: "Importances", icon: Layers,            group: "Models" },
  { key: "diagnostics", path: "/diagnostics",   label: "Diagnostics", icon: Activity,          group: "Models" },
  { key: "export",      path: "/export",        label: "Export",      icon: FileDown,          group: "Models" },
  { key: "predict",     path: "/predict",       label: "Predict",     icon: Sparkles,          group: "Models" },
];

function pathToPage(pathname: string): Page {
  const match = NAV.find((n) => n.path === pathname);
  return match ? match.key : "upload";
}

/** Build a search string carrying dataset `d` and any extras (e.g. `tab`). */
function buildSearch(d: string | null, extra?: Record<string, string>): string {
  const p = new URLSearchParams();
  if (d) p.set("d", d);
  if (extra) Object.entries(extra).forEach(([k, v]) => p.set(k, v));
  const s = p.toString();
  return s ? `?${s}` : "";
}

export default function App() {
  const navigate        = useNavigate();
  const location        = useLocation();
  const [searchParams]  = useSearchParams();
  const page            = pathToPage(location.pathname);

  // Initialise from URL param first, then localStorage fallback
  const [datasetId, setDatasetId] = useState<string | null>(() => {
    const fromUrl = searchParams.get("d");
    if (fromUrl) return fromUrl;
    return localStorage.getItem("activeDatasetId");
  });

  const [drawerOpen, setDrawerOpen] = useState(false);
  /** Bumps when transforms are applied/reset from EDA or Transforms so both tabs refetch pipeline state. */
  const [transformSyncKey, setTransformSyncKey] = useState(0);
  const onTransformsMutated = useCallback(() => setTransformSyncKey((k) => k + 1), []);

  /** Bumps after a successful train or tune so Experiments (and related tabs) refetch MLflow runs. */
  const [experimentsSyncKey, setExperimentsSyncKey] = useState(0);
  const onTrainingComplete = useCallback(() => setExperimentsSyncKey((k) => k + 1), []);

  // Persist to localStorage
  useEffect(() => {
    if (datasetId) localStorage.setItem("activeDatasetId", datasetId);
    else localStorage.removeItem("activeDatasetId");
  }, [datasetId]);

  // Sync `?d=` param into the URL whenever datasetId changes (replace so back-button isn't polluted)
  const prevDatasetId = useRef<string | null>(null);
  useEffect(() => {
    if (datasetId === prevDatasetId.current) return;
    prevDatasetId.current = datasetId;
    const currentD = searchParams.get("d");
    if (datasetId === currentD) return; // already in sync
    // Preserve any other params (e.g. `tab`)
    const extras: Record<string, string> = {};
    searchParams.forEach((v, k) => { if (k !== "d") extras[k] = v; });
    navigate(
      { pathname: location.pathname, search: buildSearch(datasetId, Object.keys(extras).length ? extras : undefined) },
      { replace: true }
    );
  }, [datasetId]); // eslint-disable-line react-hooks/exhaustive-deps

  const goTo = (path: string) => navigate({ pathname: path, search: buildSearch(datasetId) });

  const navIndex = NAV.findIndex((n) => n.key === page);
  const nextNav = navIndex >= 0 && navIndex < NAV.length - 1 ? NAV[navIndex + 1] : null;

  return (
    <div className="flex h-screen min-h-0 overflow-hidden bg-slate-50">
      {/* ── Sidebar ──────────────────────────────────────────────────────── */}
      <aside className="w-[15.5rem] bg-slate-900 flex flex-col fixed inset-y-0 left-0 z-40 border-r border-slate-800/80">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <Database className="w-5 h-5 text-blue-400" />
            <span className="text-white font-semibold text-base tracking-tight">ML Studio</span>
            <button
              onClick={() => setDrawerOpen(true)}
              className="ml-auto p-1 rounded text-slate-500 hover:text-blue-400 transition-colors"
              title="How it works"
            >
              <HelpCircle className="w-4 h-4" />
            </button>
          </div>
        </div>

        <SidebarWorkflowNav
          items={NAV}
          activeIndex={navIndex}
          currentKey={page}
          onSelect={goTo}
        />

        {/* Active dataset */}
        <div className="px-4 py-4 border-t border-slate-800 space-y-3">
          {datasetId ? (
            <div>
              <p className="text-xs text-slate-500 font-medium mb-1">Active Dataset</p>
              <p className="text-xs text-slate-300 font-mono break-all">{datasetId}</p>
              <button
                onClick={() => { setDatasetId(null); navigate("/"); }}
                className="mt-2 w-full text-xs py-1.5 px-2 rounded bg-slate-800 text-slate-400 hover:bg-rose-900/50 hover:text-rose-300 transition-colors"
              >
                Clear dataset
              </button>
            </div>
          ) : (
            <p className="text-xs text-slate-600">No dataset loaded</p>
          )}

          <a
            href={import.meta.env.VITE_MLFLOW_UI_URL ?? "http://localhost:5000"}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-blue-400 transition-colors"
          >
            <FlaskConical className="w-3 h-3" />
            MLflow UI
          </a>
        </div>
      </aside>

      <InfoDrawer open={drawerOpen} page={page} onClose={() => setDrawerOpen(false)} />

      {/* ── Main content — all pages stay mounted ────────────────────────── */}
      <main className="ml-[15.5rem] flex min-h-0 flex-1 flex-col overflow-hidden">
        <div
          className={
            nextNav
              ? "min-h-0 flex-1 overflow-y-auto pb-[5.25rem]"
              : "min-h-0 flex-1 overflow-y-auto"
          }
        >
          <div className={page === "upload"      ? undefined : "hidden"}>
            <UploadPage
              currentDatasetId={datasetId}
              onClearDataset={() => setDatasetId(null)}
              onDatasetCreated={setDatasetId}
              onNavigateToTrain={() => goTo("/train")}
              onNavigateToSplit={() => goTo("/split")}
            />
          </div>
          <div className={page === "split"       ? undefined : "hidden"}>
            <HoldoutSplit datasetId={datasetId} onHoldoutSaved={onTransformsMutated} />
          </div>
          <div className={page === "eda"         ? undefined : "hidden"}>
            <EDA datasetId={datasetId} transformSyncKey={transformSyncKey} onTransformsMutated={onTransformsMutated} />
          </div>
          <div className={page === "transforms"  ? undefined : "hidden"}>
            <Transforms datasetId={datasetId} transformSyncKey={transformSyncKey} onTransformsMutated={onTransformsMutated} />
          </div>
          <div className={page === "train"       ? undefined : "hidden"}>
            <Train datasetId={datasetId} onTrainingComplete={onTrainingComplete} />
          </div>
          <div className={page === "experiments" ? undefined : "hidden"}>
            <Experiments datasetId={datasetId} experimentsSyncKey={experimentsSyncKey} />
          </div>
          <div className={page === "importances" ? undefined : "hidden"}>
            <Importances datasetId={datasetId} experimentsSyncKey={experimentsSyncKey} />
          </div>
          <div className={page === "diagnostics" ? undefined : "hidden"}>
            <Diagnostics datasetId={datasetId} experimentsSyncKey={experimentsSyncKey} />
          </div>
          <div className={page === "export"      ? undefined : "hidden"}>
            <PipelineExport datasetId={datasetId} experimentsSyncKey={experimentsSyncKey} />
          </div>
          <div className={page === "predict"     ? undefined : "hidden"}>
            <Predict datasetId={datasetId} experimentsSyncKey={experimentsSyncKey} />
          </div>
        </div>
        {nextNav && (
          <WorkflowNextBar nextLabel={nextNav.label} onNext={() => goTo(nextNav.path)} />
        )}
      </main>
    </div>
  );
}
