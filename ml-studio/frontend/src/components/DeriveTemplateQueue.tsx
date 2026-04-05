import { useMemo, useState } from "react";
import { toast } from "sonner";
import { DERIVE_TEMPLATES, validateTemplateFields } from "../deriveTemplates";
import type { Step } from "../transformTypes";
import { Select } from "./ui/select";
import { Button } from "./ui/button";

interface Props {
  allColumns: string[];
  onQueueSteps: (steps: Step[]) => void;
  disabled?: boolean;
}

export function DeriveTemplateQueue({ allColumns, onQueueSteps, disabled }: Props) {
  const [templateId, setTemplateId] = useState(DERIVE_TEMPLATES[0]?.id ?? "");
  const [values, setValues] = useState<Record<string, string>>({});

  const template = useMemo(
    () => DERIVE_TEMPLATES.find((t) => t.id === templateId) ?? DERIVE_TEMPLATES[0],
    [templateId],
  );

  const colOpts = useMemo(
    () => allColumns.map((c) => ({ value: c, label: c })),
    [allColumns],
  );

  const columnSet = useMemo(() => new Set(allColumns), [allColumns]);

  const queue = () => {
    if (!template) return;
    const err = validateTemplateFields(template, values, columnSet);
    if (err) {
      toast.error(err);
      return;
    }
    const steps = template.build(values);
    onQueueSteps(steps);
    toast.success(`Added ${steps.length} derive step(s) to the queue.`);
  };

  if (!template) return null;

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm ring-1 ring-slate-100/80">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-slate-800">Derived columns</h3>
        <p className="mt-0.5 text-xs leading-relaxed text-slate-500">
          Shortcut for ratio-style features (e.g. rooms ÷ households). Pick columns, then queue the step.
        </p>
      </div>
      <div className="space-y-3">
        <div>
          <label className="mb-1.5 block text-[11px] font-medium text-slate-600">Template</label>
          <Select
            value={templateId}
            onChange={(v) => {
              setTemplateId(v);
              setValues({});
            }}
            options={DERIVE_TEMPLATES.map((t) => ({ value: t.id, label: t.label }))}
            className="w-full"
          />
        </div>
        <p className="text-xs leading-relaxed text-slate-600">{template.description}</p>
        <div className="grid gap-3 sm:grid-cols-2">
          {template.fields.map((f) => (
            <div key={f.key}>
              <label className="mb-1 block text-[11px] font-medium text-slate-600">
                {f.label}
                {f.optional ? <span className="font-normal text-slate-400"> (optional)</span> : null}
              </label>
              {f.mustBeColumn === false ? (
                <input
                  type="text"
                  value={values[f.key] ?? ""}
                  placeholder={f.optional ? "Auto if empty" : ""}
                  onChange={(e) => setValues((prev) => ({ ...prev, [f.key]: e.target.value }))}
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm"
                />
              ) : (
                <Select
                  value={values[f.key] ?? ""}
                  onChange={(v) => setValues((prev) => ({ ...prev, [f.key]: v }))}
                  options={[{ value: "", label: "Choose column…" }, ...colOpts]}
                  className="w-full"
                />
              )}
            </div>
          ))}
        </div>
        <Button
          type="button"
          className="w-full"
          disabled={disabled}
          onClick={queue}
        >
          Queue derived step
        </Button>
      </div>
    </section>
  );
}
