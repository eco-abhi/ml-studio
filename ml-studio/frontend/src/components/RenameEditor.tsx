import { Plus, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Select } from "./ui/select";

export interface RenameEditorProps {
  allColumns: string[];
  mapping: Record<string, string>;
  onChange: (mapping: Record<string, string>) => void;
}

export function RenameEditor({ allColumns, mapping, onChange }: RenameEditorProps) {
  const rows = Object.entries(mapping);

  const addRow = () => {
    const unused = allColumns.find((c) => !(c in mapping)) ?? "";
    if (!unused) {
      toast.error("All columns are already mapped.");
      return;
    }
    onChange({ ...mapping, [unused]: "" });
  };

  const updateRow = (oldKey: string, newOldKey: string, newVal: string) => {
    const next: Record<string, string> = {};
    for (const [k, v] of Object.entries(mapping)) {
      if (k === oldKey) {
        if (newOldKey) next[newOldKey] = newVal;
      } else {
        next[k] = v;
      }
    }
    onChange(next);
  };

  const removeRow = (key: string) => {
    const { [key]: _, ...rest } = mapping;
    onChange(rest);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <label className="text-xs font-medium text-slate-600">Column renames</label>
        <button
          type="button"
          onClick={addRow}
          className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium text-blue-600 hover:bg-blue-50 transition-colors"
        >
          <Plus className="w-3.5 h-3.5" />
          Add pair
        </button>
      </div>

      {rows.length === 0 && (
        <p className="text-xs text-slate-400 rounded-lg border border-dashed border-slate-200 bg-slate-50/80 px-3 py-2.5">
          Add at least one <strong className="font-medium text-slate-600">old → new</strong> column name pair.
        </p>
      )}

      {rows.map(([oldName, newName]) => (
        <div key={oldName} className="flex flex-wrap sm:flex-nowrap items-center gap-2">
          <Select
            value={oldName}
            onChange={(v) => updateRow(oldName, v, newName)}
            options={allColumns
              .filter((c) => c === oldName || !(c in mapping))
              .map((c) => ({ value: c, label: c }))}
            size="sm"
            className="flex-1 min-w-[120px]"
          />
          <span className="text-xs text-slate-400 shrink-0 hidden sm:inline">→</span>
          <input
            type="text"
            value={newName}
            placeholder="New column name"
            onChange={(e) => updateRow(oldName, oldName, e.target.value)}
            className="flex-1 min-w-[120px] border border-slate-200 rounded-lg px-2.5 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="button"
            onClick={() => removeRow(oldName)}
            className="p-1.5 rounded-lg hover:bg-rose-50 text-slate-400 hover:text-rose-600 transition-colors shrink-0"
            aria-label="Remove rename row"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
}
