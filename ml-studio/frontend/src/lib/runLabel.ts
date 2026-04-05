import type { ExperimentRun } from "../api";

/** Last 8 chars of MLflow run id (hex), for compact display. */
export function shortRunId(runId: string): string {
  if (!runId) return "";
  return runId.length <= 8 ? runId : runId.slice(-8);
}

/** One line: "random forest · a1b2c3d4" */
export function compactRunLabel(run: Pick<ExperimentRun, "model" | "run_id">): string {
  const name = run.model.replace(/_/g, " ");
  return `${name} · ${shortRunId(run.run_id)}`;
}

/** Includes local start time when MLflow provides it (ms epoch). */
export function formatRunLabel(run: Pick<ExperimentRun, "model" | "run_id" | "started_at">): string {
  const base = compactRunLabel(run);
  if (run.started_at == null) return base;
  const t = new Date(run.started_at).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
  return `${base} · ${t}`;
}
