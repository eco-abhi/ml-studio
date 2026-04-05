import html2canvas from "html2canvas";
import { jsPDF } from "jspdf";

const CANVAS_OPTS: Partial<Parameters<typeof html2canvas>[1]> = {
  scale: 2,
  backgroundColor: "#ffffff",
  logging: false,
  useCORS: true,
};

/**
 * Rasterize a DOM subtree (chart, table, etc.) and download as PNG.
 * Uses html2canvas so Tailwind / SVG render consistently in the snapshot.
 */
export async function exportElementToPng(el: HTMLElement, filenameBase: string): Promise<void> {
  const canvas = await html2canvas(el, CANVAS_OPTS);
  const a = document.createElement("a");
  a.download = `${filenameBase}.png`;
  a.href = canvas.toDataURL("image/png");
  a.click();
}

/** Same capture as PNG, embedded in a single-page PDF sized to the image. */
export async function exportElementToPdf(el: HTMLElement, filenameBase: string): Promise<void> {
  const canvas = await html2canvas(el, CANVAS_OPTS);
  const imgData = canvas.toDataURL("image/png");
  const pdfW = canvas.width;
  const pdfH = canvas.height;
  const pdf = new jsPDF({
    orientation: pdfW >= pdfH ? "landscape" : "portrait",
    unit: "px",
    format: [pdfW, pdfH],
  });
  pdf.addImage(imgData, "PNG", 0, 0, pdfW, pdfH);
  pdf.save(`${filenameBase}.pdf`);
}

export function sanitizeFilenameBase(s: string): string {
  return s.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_|_$/g, "") || "figure";
}
