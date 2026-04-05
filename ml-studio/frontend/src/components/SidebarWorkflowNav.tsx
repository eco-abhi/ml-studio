import { Check } from "lucide-react";
import type { ElementType } from "react";
import { cn } from "../lib/utils";

export interface WorkflowNavItem {
  key: string;
  path: string;
  label: string;
  icon: ElementType;
  group: string;
}

interface Props {
  items: WorkflowNavItem[];
  activeIndex: number;
  currentKey: string;
  onSelect: (path: string) => void;
}

export function SidebarWorkflowNav({ items, activeIndex, currentKey, onSelect }: Props) {
  const safeIndex = activeIndex >= 0 ? activeIndex : 0;

  return (
    <nav className="flex-1 overflow-y-auto px-2 py-3" aria-label="Workflow steps">
      <p className="mb-3 px-2 text-[10px] font-bold uppercase tracking-[0.12em] text-slate-500">
        Workflow
      </p>
      <ol className="space-y-0 list-none m-0 p-0">
        {items.map((item, i) => {
          const isDone = i < safeIndex;
          const isCurrent = item.key === currentKey;
          const Icon = item.icon;
          const phaseStart = i === 0 || item.group !== items[i - 1].group;

          return (
            <li key={item.key}>
              {phaseStart && i > 0 && (
                <div
                  className="mb-2 mt-4 border-t border-slate-800/90 pt-3 px-2 text-[10px] font-bold uppercase tracking-[0.12em] text-slate-600"
                  aria-hidden
                >
                  {item.group}
                </div>
              )}
              {phaseStart && i === 0 && (
                <div className="mb-1 px-2 text-[10px] font-bold uppercase tracking-[0.12em] text-slate-600">
                  {item.group}
                </div>
              )}
              <div className="flex gap-2">
                <div className="flex w-8 shrink-0 flex-col items-center">
                  {i > 0 && (
                    <div
                      className={cn(
                        "w-0.5 shrink-0 rounded-full",
                        safeIndex >= i ? "bg-blue-500/55" : "bg-slate-800",
                        "h-2.5"
                      )}
                      aria-hidden
                    />
                  )}
                  <button
                    type="button"
                    onClick={() => onSelect(item.path)}
                    title={`Step ${i + 1}: ${item.label}`}
                    className={cn(
                      "relative z-10 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[11px] font-bold tabular-nums transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900",
                      isDone &&
                        "bg-blue-600 text-white shadow-md shadow-blue-950/40 hover:bg-blue-500",
                      isCurrent &&
                        "bg-blue-500 text-white ring-2 ring-blue-300/90 ring-offset-2 ring-offset-slate-900 shadow-lg shadow-blue-950/50 hover:bg-blue-400",
                      !isDone &&
                        !isCurrent &&
                        "bg-slate-800 text-slate-500 ring-1 ring-slate-700 hover:bg-slate-700 hover:text-slate-300"
                    )}
                  >
                    {isDone ? <Check className="h-3.5 w-3.5" strokeWidth={2.5} aria-hidden /> : i + 1}
                  </button>
                  {i < items.length - 1 && (
                    <div
                      className={cn(
                        "w-0.5 shrink-0 rounded-full",
                        safeIndex > i ? "bg-blue-500/55" : "bg-slate-800",
                        "h-2.5 mt-0.5"
                      )}
                      aria-hidden
                    />
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => onSelect(item.path)}
                  className={cn(
                    "min-w-0 flex-1 rounded-lg px-2 py-1.5 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900",
                    isCurrent && "bg-slate-800 text-white font-medium",
                    isDone && !isCurrent && "text-slate-300 hover:bg-slate-800/90",
                    !isDone && !isCurrent && "text-slate-500 hover:bg-slate-800/70 hover:text-slate-200"
                  )}
                >
                  <span className="flex items-center gap-2">
                    <Icon
                      className={cn(
                        "h-3.5 w-3.5 shrink-0",
                        isCurrent ? "text-blue-300" : isDone ? "text-blue-400/90" : "text-slate-600"
                      )}
                      aria-hidden
                    />
                    <span className="truncate">{item.label}</span>
                  </span>
                </button>
              </div>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
