import { useState } from "react";
import { toast } from "sonner";
import { exportElementToPdf, exportElementToPng, sanitizeFilenameBase } from "../lib/chartExport";
import { Button } from "./ui/button";

interface Props {
  targetRef: React.RefObject<HTMLElement | null>;
  filenameBase: string;
  disabled?: boolean;
  className?: string;
}

export function ChartExportButtons({ targetRef, filenameBase, disabled, className }: Props) {
  const [busy, setBusy] = useState<"png" | "pdf" | null>(null);

  const run = async (kind: "png" | "pdf") => {
    const el = targetRef.current;
    if (!el) {
      toast.error("Nothing to export yet.");
      return;
    }
    const base = sanitizeFilenameBase(filenameBase);
    setBusy(kind);
    try {
      if (kind === "png") await exportElementToPng(el, base);
      else await exportElementToPdf(el, base);
    } catch {
      toast.error("Export failed. If the chart uses web fonts, try again after it finishes loading.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className={`flex flex-wrap items-center gap-1 ${className ?? ""}`}>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="h-7 px-2 text-[10px] font-semibold"
        disabled={disabled || busy !== null}
        onClick={() => void run("png")}
      >
        {busy === "png" ? "…" : "PNG"}
      </Button>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="h-7 px-2 text-[10px] font-semibold"
        disabled={disabled || busy !== null}
        onClick={() => void run("pdf")}
      >
        {busy === "pdf" ? "…" : "PDF"}
      </Button>
    </div>
  );
}
