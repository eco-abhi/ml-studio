import { ChevronDown, Search } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { cn } from "../lib/utils";
import { STEP_META, TRANSFORM_OPTION_GROUPS, type StepType } from "../transformTypes";

interface TransformTypePickerProps {
  value: StepType;
  onChange: (t: StepType) => void;
  className?: string;
  size?: "sm" | "md";
  disabled?: boolean;
}

function matchesQuery(t: StepType, q: string): boolean {
  if (!q) return true;
  const label = STEP_META[t].label.toLowerCase();
  const desc = STEP_META[t].description.toLowerCase();
  const key = t.replace(/_/g, " ").toLowerCase();
  return label.includes(q) || desc.includes(q) || key.includes(q);
}

export function TransformTypePicker({ value, onChange, className = "", size = "md", disabled = false }: TransformTypePickerProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const anchorRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [menuBox, setMenuBox] = useState({ top: 0, left: 0, width: 0, maxH: 320 });

  const updateMenuPosition = () => {
    const el = anchorRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const gap = 4;
    const viewportPad = 12;
    const preferredMax = 320;
    const belowTop = r.bottom + gap;
    const spaceBelow = window.innerHeight - belowTop - viewportPad;
    const spaceAbove = r.top - viewportPad;
    const openDown = spaceBelow >= 180 || spaceBelow >= spaceAbove;
    const maxH = Math.min(
      preferredMax,
      Math.max(120, openDown ? spaceBelow : spaceAbove - gap),
    );
    const top = openDown ? belowTop : Math.max(viewportPad, r.top - gap - maxH);
    const width = Math.max(r.width, 280);
    const left = Math.min(Math.max(viewportPad, r.left), window.innerWidth - width - viewportPad);
    setMenuBox({ top, left, width, maxH });
  };

  useLayoutEffect(() => {
    if (!open) return;
    updateMenuPosition();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => updateMenuPosition()) : null;
    if (anchorRef.current && ro) ro.observe(anchorRef.current);
    window.addEventListener("scroll", updateMenuPosition, true);
    window.addEventListener("resize", updateMenuPosition);
    return () => {
      ro?.disconnect();
      window.removeEventListener("scroll", updateMenuPosition, true);
      window.removeEventListener("resize", updateMenuPosition);
    };
  }, [open]);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      const t = e.target as Node;
      if (anchorRef.current?.contains(t) || menuRef.current?.contains(t)) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const qNorm = query.trim().toLowerCase();

  const filteredGroups = useMemo(() => {
    return TRANSFORM_OPTION_GROUPS.map((g) => {
      const catMatch = qNorm.length > 0 && g.category.toLowerCase().includes(qNorm);
      const types = g.types.filter((t) => (catMatch ? true : matchesQuery(t, qNorm)));
      return { category: g.category, types };
    }).filter((g) => g.types.length > 0);
  }, [qNorm]);

  const pad = size === "sm" ? "px-2 py-1 text-xs" : "px-3 py-2 text-sm";

  const menu = open && !disabled && (
    <div
      ref={menuRef}
      role="listbox"
      className="fixed z-[200] flex flex-col overflow-hidden rounded-lg border border-slate-200 bg-white shadow-lg"
      style={{
        top: menuBox.top,
        left: menuBox.left,
        width: menuBox.width,
        maxHeight: menuBox.maxH,
      }}
    >
      <div className="shrink-0 border-b border-slate-100 bg-slate-50/80 p-2">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search transforms…"
            className="w-full rounded-md border border-slate-200 bg-white py-1.5 pl-8 pr-2 text-xs focus:outline-none focus:ring-2 focus:ring-blue-500"
            autoFocus
          />
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto py-1">
        {filteredGroups.length === 0 ? (
          <p className="px-3 py-4 text-center text-xs text-slate-400">No matches</p>
        ) : (
          filteredGroups.map((g) => (
            <div key={g.category}>
              <p className="sticky top-0 bg-slate-50/95 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-400 backdrop-blur-sm">
                {g.category}
              </p>
              {g.types.map((t) => (
                <button
                  key={t}
                  type="button"
                  role="option"
                  aria-selected={t === value}
                  onClick={() => {
                    onChange(t);
                    setOpen(false);
                    setQuery("");
                  }}
                  className={cn(
                    "w-full border-b border-slate-50 px-3 py-2 text-left text-sm transition-colors last:border-0",
                    t === value ? "bg-blue-50 font-medium text-blue-800" : "text-slate-700 hover:bg-slate-50",
                  )}
                >
                  <span className="block">{STEP_META[t].label}</span>
                  <span className="mt-0.5 block text-[10px] font-normal leading-snug text-slate-400">
                    {STEP_META[t].description}
                  </span>
                </button>
              ))}
            </div>
          ))
        )}
      </div>
    </div>
  );

  return (
    <>
      <div className={cn("relative", className)} ref={anchorRef}>
        <button
          type="button"
          aria-expanded={open}
          aria-haspopup="listbox"
          disabled={disabled}
          onClick={() => !disabled && setOpen((o) => !o)}
          className={cn(
            "flex w-full items-center justify-between gap-2 rounded-lg border border-slate-200 bg-white transition-colors hover:border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500",
            pad,
            disabled && "cursor-not-allowed opacity-50 hover:border-slate-200",
          )}
        >
          <span className="truncate text-left text-slate-800">{STEP_META[value].label}</span>
          <ChevronDown className={cn("h-3.5 w-3.5 shrink-0 text-slate-400 transition-transform", open && "rotate-180")} />
        </button>
      </div>
      {typeof document !== "undefined" && menu ? createPortal(menu, document.body) : null}
    </>
  );
}
