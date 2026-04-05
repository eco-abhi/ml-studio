import { Loader2 } from "lucide-react";
import { cn } from "../lib/utils";

interface LoadingStateProps {
  message?: string;
  /** `page` — tall block for full sections; `overlay` — absolute fill (parent must be `relative`) */
  variant?: "page" | "inline" | "overlay";
  className?: string;
}

export function LoadingState({
  message = "Loading…",
  variant = "inline",
  className,
}: LoadingStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 text-slate-600",
        variant === "page" && "min-h-[min(360px,55vh)]",
        variant === "inline" && "py-10",
        variant === "overlay" &&
          "absolute inset-0 z-20 flex rounded-[inherit] bg-white/90 backdrop-blur-[2px]",
        className,
      )}
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <Loader2 className="h-8 w-8 shrink-0 animate-spin text-blue-600" aria-hidden />
      <p className="max-w-sm text-center text-sm font-medium text-slate-600">{message}</p>
    </div>
  );
}
