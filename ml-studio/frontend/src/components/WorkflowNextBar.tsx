import { ArrowRight } from "lucide-react";
import { Button } from "./ui/button";
import { cn } from "../lib/utils";

interface Props {
  nextLabel: string;
  onNext: () => void;
}

/**
 * Fixed to the bottom of the main content column (right of sidebar) so it stays visible
 * while the page above scrolls. Matches sidebar width `15.5rem` in App layout.
 */
export function WorkflowNextBar({ nextLabel, onNext }: Props) {
  return (
    <div
      className={cn(
        "fixed bottom-0 left-[15.5rem] right-0 z-30",
        "border-t border-slate-200 bg-white/95 px-6 py-3",
        "pb-[max(0.75rem,env(safe-area-inset-bottom,0px))]",
        "shadow-[0_-8px_24px_-4px_rgba(15,23,42,0.08)] backdrop-blur-sm supports-[backdrop-filter]:bg-white/85"
      )}
    >
      <div className="mx-auto flex max-w-6xl justify-end">
        <Button
          type="button"
          size="lg"
          className="min-w-[10rem] gap-2 shadow-md"
          onClick={onNext}
          aria-label={`Go to next step: ${nextLabel}`}
        >
          Next
          <span className="font-normal text-white/90">· {nextLabel}</span>
          <ArrowRight className="h-4 w-4 shrink-0 opacity-90" aria-hidden />
        </Button>
      </div>
    </div>
  );
}
