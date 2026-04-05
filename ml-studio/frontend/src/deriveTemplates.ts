import type { DeriveStep, Step } from "./transformTypes";
import { makeStep } from "./transformTypes";

export interface TemplateField {
  key: string;
  label: string;
  optional?: boolean;
  /** When false, value is a new column name (not validated against the dataset). Default true. */
  mustBeColumn?: boolean;
}

export interface DeriveTemplateDef {
  id: string;
  label: string;
  description: string;
  fields: TemplateField[];
  /** Build one or more derive_numeric steps from chosen columns. */
  build: (values: Record<string, string>) => Step[];
}

function derive(
  column_a: string,
  column_b: string,
  op: DeriveStep["op"],
  output_column: string,
): Step {
  const s = makeStep("derive_numeric") as DeriveStep;
  s.column_a = column_a;
  s.column_b = column_b;
  s.op = op;
  s.output_column = output_column;
  return s;
}

/** Curated presets for tabular regression / geospatial-style feature engineering. */
export const DERIVE_TEMPLATES: DeriveTemplateDef[] = [
  {
    id: "ratio",
    label: "Ratio (A ÷ B)",
    description: "Density or per-capita style features, e.g. rooms ÷ households.",
    fields: [
      { key: "numerator", label: "Numerator (A)" },
      { key: "denominator", label: "Denominator (B)" },
      { key: "output", label: "Output column name", optional: true, mustBeColumn: false },
    ],
    build: (v) => {
      const out =
        v.output?.trim() ||
        `${v.numerator}_per_${v.denominator}`.replace(/[^a-zA-Z0-9_]+/g, "_");
      return [derive(v.numerator, v.denominator, "divide", out)];
    },
  },
  {
    id: "product",
    label: "Product (A × B)",
    description: "Simple interaction term between two numeric columns.",
    fields: [
      { key: "a", label: "Column A" },
      { key: "b", label: "Column B" },
      { key: "output", label: "Output column name", optional: true, mustBeColumn: false },
    ],
    build: (v) => {
      const out = v.output?.trim() || `${v.a}_x_${v.b}`.replace(/[^a-zA-Z0-9_]+/g, "_");
      return [derive(v.a, v.b, "multiply", out)];
    },
  },
  {
    id: "sum_diff",
    label: "Sum and difference",
    description: "Adds A+B and A−B as two new columns (e.g. total and spread).",
    fields: [
      { key: "a", label: "Column A" },
      { key: "b", label: "Column B" },
    ],
    build: (v) => [
      derive(v.a, v.b, "add", `${v.a}_plus_${v.b}`.replace(/[^a-zA-Z0-9_]+/g, "_")),
      derive(v.a, v.b, "subtract", `${v.a}_minus_${v.b}`.replace(/[^a-zA-Z0-9_]+/g, "_")),
    ],
  },
  {
    id: "housing_rates",
    label: "Housing-style rates",
    description: "Classic California-housing style: rooms per household and/or bedrooms per room (map your column names).",
    fields: [
      { key: "total_rooms", label: "Total rooms", optional: true },
      { key: "households", label: "Households", optional: true },
      { key: "bedrooms", label: "Bedrooms", optional: true },
    ],
    build: (v) => {
      const steps: Step[] = [];
      if (v.total_rooms && v.households) {
        steps.push(derive(v.total_rooms, v.households, "divide", "rooms_per_household"));
      }
      if (v.bedrooms && v.total_rooms) {
        steps.push(derive(v.bedrooms, v.total_rooms, "divide", "bedrooms_per_room"));
      }
      return steps;
    },
  },
  {
    id: "lat_lon_interaction",
    label: "Lat × lon interaction",
    description: "Single multiplicative geography proxy (nonlinear location signal).",
    fields: [
      { key: "lat", label: "Latitude column" },
      { key: "lon", label: "Longitude column" },
    ],
    build: (v) => [derive(v.lat, v.lon, "multiply", "lat_times_lon")],
  },
];

export function validateTemplateFields(
  t: DeriveTemplateDef,
  values: Record<string, string>,
  columnSet: Set<string>,
): string | null {
  for (const f of t.fields) {
    const raw = (values[f.key] ?? "").trim();
    if (!f.optional && !raw) return `Choose ${f.label}.`;
    if (raw && f.mustBeColumn !== false && !columnSet.has(raw)) return `Column not found: ${raw}`;
  }
  if (t.id === "housing_rates") {
    const tr = values.total_rooms?.trim();
    const hh = values.households?.trim();
    const br = values.bedrooms?.trim();
    const ok1 = Boolean(tr && hh);
    const ok2 = Boolean(br && tr);
    if (!ok1 && !ok2) {
      return "Provide total rooms + households and/or bedrooms + total rooms.";
    }
  }
  const built = t.build(values);
  if (!built.length) return "Nothing to add — fill the required column mappings.";
  return null;
}
