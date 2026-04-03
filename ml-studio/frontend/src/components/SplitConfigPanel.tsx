import type { SplitConfig } from "../api";
import { Select } from "./ui/select";

// ── Defaults ──────────────────────────────────────────────────────────────────

export const DEFAULT_SPLIT_CONFIG: SplitConfig = {
  test_size:    0.2,
  random_state: 42,
  shuffle:      true,
  stratify:     true,
  val_strategy: "kfold",
  cv_folds:     5,
  cv_repeats:   3,
  val_size:     0.1,
  scaler:       "standard",
  imbalance_method:         "none",
  train_pca_components:       0,
  categorical_encoding:       "numeric_only",
  max_category_cardinality: 50,
};

// ── Option lists ──────────────────────────────────────────────────────────────

export const VAL_STRATEGIES: { value: string; label: string; description: string }[] = [
  {
    value: "none",
    label: "None (hold-out only)",
    description: "No cross-validation. Evaluates only on the single hold-out test split.",
  },
  {
    value: "kfold",
    label: "K-Fold CV",
    description: "Splits training data into k equal folds. Each fold acts as validation once. Most common choice.",
  },
  {
    value: "stratified_kfold",
    label: "Stratified K-Fold CV",
    description: "Like K-Fold but preserves class proportions in each fold. Best for imbalanced classification.",
  },
  {
    value: "repeated_kfold",
    label: "Repeated K-Fold CV",
    description: "Repeats K-Fold multiple times with different random splits. More robust variance estimate.",
  },
  {
    value: "repeated_stratified_kfold",
    label: "Repeated Stratified K-Fold",
    description: "Combines stratification with repeats. Most thorough, but slowest.",
  },
  {
    value: "shuffle_split",
    label: "Shuffle Split",
    description: "Random train/val splits repeated k times (Monte Carlo CV). Each split is independent.",
  },
  {
    value: "loo",
    label: "Leave-One-Out (LOO)",
    description: "Each sample is the validation set once. Nearly unbiased but very slow on large datasets.",
  },
];

const SCALERS: { value: string; label: string; description: string }[] = [
  { value: "standard", label: "Standard Scaler",  description: "Normalises to zero mean, unit variance: (x − μ) / σ. Default and works well for most models." },
  { value: "minmax",   label: "Min-Max Scaler",   description: "Scales to [0, 1]: (x − min) / (max − min). Preserves zero entries; sensitive to outliers." },
  { value: "robust",   label: "Robust Scaler",    description: "Uses median and IQR: (x − median) / IQR. Resistant to outliers." },
  { value: "none",     label: "No Scaling",        description: "Passes raw features. Fine for tree-based models; can hurt linear/SVM models." },
];

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  config: SplitConfig;
  onChange: (patch: Partial<SplitConfig>) => void;
  taskType: "regression" | "classification" | null;
  /** When the dataset has __ml_split__ from Transforms, hide train/test holdout controls. */
  hideHoldoutSplit?: boolean;
}

export function SplitConfigPanel({ config: c, onChange, taskType, hideHoldoutSplit = false }: Props) {
  const isClassification = taskType === "classification";
  const needsFolds  = c.val_strategy !== "none" && c.val_strategy !== "loo";
  const needsRepeats = c.val_strategy === "repeated_kfold" || c.val_strategy === "repeated_stratified_kfold";
  const needsValSize = c.val_strategy === "shuffle_split";

  const currentStrategy = VAL_STRATEGIES.find((s) => s.value === c.val_strategy);
  const currentScaler   = SCALERS.find((s) => s.value === c.scaler);

  return (
    <div className="space-y-5">

      {/* ── Train / Test split ── */}
      <section className="space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">Train / Test Split</h3>

        {hideHoldoutSplit && (
          <p className="rounded-lg border border-blue-200 bg-blue-50 p-3 text-xs text-blue-900 leading-relaxed">
            Hold-out rows are defined by the <code className="rounded bg-blue-100 px-1 text-[11px]">__ml_split__</code> column from your{" "}
            <strong>Transforms</strong> pipeline. CV, scaling, and pipeline extras below still apply only to training data (no leakage into the test set).
          </p>
        )}

        {!hideHoldoutSplit && (
        <>
        {/* Test size */}
        <div>
          <div className="flex items-baseline justify-between mb-1">
            <label className="text-sm font-medium text-slate-700">Test size</label>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">{Math.round((1 - c.test_size) * 100)}% train · {Math.round(c.test_size * 100)}% test</span>
              <input
                type="number" value={c.test_size} step={0.05} min={0.05} max={0.5}
                onChange={(e) => onChange({ test_size: Math.min(0.5, Math.max(0.05, parseFloat(e.target.value) || 0.2)) })}
                className="w-16 text-right text-xs font-semibold border border-slate-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>
          </div>
          <input
            type="range" min={0.05} max={0.5} step={0.05} value={c.test_size}
            onChange={(e) => onChange({ test_size: parseFloat(e.target.value) })}
            className="w-full"
          />
          <p className="text-[11px] text-slate-400 mt-1">
            Fraction of data held out as the final test set. 0.2 (20%) is the standard default.
          </p>
        </div>

        {/* Random state */}
        <div>
          <div className="flex items-baseline justify-between mb-1">
            <label className="text-sm font-medium text-slate-700">Random state</label>
            <input
              type="number" value={c.random_state} min={0} max={99999} step={1}
              onChange={(e) => onChange({ random_state: parseInt(e.target.value) || 42 })}
              className="w-20 text-right text-xs font-semibold border border-slate-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
          <p className="text-[11px] text-slate-400">Seed for reproducibility. Same seed = same split every run.</p>
        </div>

        {/* Shuffle + Stratify toggles */}
        <div className="flex gap-3">
          <Toggle
            label="Shuffle"
            checked={c.shuffle}
            onChange={(v) => onChange({ shuffle: v })}
            description="Randomise row order before splitting. Always recommended unless data has temporal ordering."
          />
          <Toggle
            label="Stratify"
            checked={c.stratify}
            onChange={(v) => onChange({ stratify: v })}
            disabled={!isClassification}
            description={isClassification
              ? "Preserve class ratios in each split. Strongly recommended for imbalanced classes."
              : "Only applies to classification tasks."}
          />
        </div>
        </>
        )}
      </section>

      {/* ── Scaler ── */}
      <section className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">Feature Scaling</h3>
        <div className="grid grid-cols-2 gap-2">
          {SCALERS.map((s) => (
            <button
              key={s.value}
              onClick={() => onChange({ scaler: s.value })}
              className={`text-left px-3 py-2 rounded-lg border text-sm transition-colors ${
                c.scaler === s.value
                  ? "border-blue-500 bg-blue-50 text-blue-800"
                  : "border-slate-200 text-slate-600 hover:border-slate-300"
              }`}
            >
              <p className="font-medium text-xs">{s.label}</p>
            </button>
          ))}
        </div>
        {currentScaler && (
          <p className="text-[11px] text-slate-400 leading-relaxed">{currentScaler.description}</p>
        )}
      </section>

      {/* ── Extra preprocessing (Train / Tune pipeline) ── */}
      <section className="space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">Model pipeline extras</h3>
        <p className="text-[11px] text-slate-400">
          Applied inside sklearn / imblearn pipelines after the train/test split (no leakage into the held-out test set).
        </p>

        {isClassification && (
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Imbalance (training data only)</label>
            <Select
              value={c.imbalance_method}
              onChange={(v) => onChange({ imbalance_method: v })}
              options={[
                { value: "none", label: "None" },
                { value: "smote", label: "SMOTE oversampling" },
                { value: "random_under", label: "Random undersampling" },
              ]}
              size="sm"
            />
            <p className="text-[11px] text-slate-400 mt-1">Requires imbalanced-learn. SMOTE needs numeric features after preprocessing.</p>
          </div>
        )}

        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">PCA inside pipeline</label>
          <div className="flex items-center gap-2">
            <input
              type="number"
              min={0}
              max={100}
              value={c.train_pca_components}
              onChange={(e) =>
                onChange({ train_pca_components: Math.max(0, parseInt(e.target.value, 10) || 0) })
              }
              className="w-20 text-right text-xs font-semibold border border-slate-200 rounded px-2 py-1"
            />
            <span className="text-[11px] text-slate-400">components (0 = disabled)</span>
          </div>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Non-numeric columns</label>
          <Select
            value={c.categorical_encoding}
            onChange={(v) => onChange({ categorical_encoding: v })}
            options={[
              { value: "numeric_only", label: "Use numeric columns only (default)" },
              { value: "target_encode", label: "Target encoding (CV-safe in sklearn)" },
              { value: "one_hot", label: "One-hot encode (≤ max categories)" },
            ]}
            size="sm"
          />
          <div className="flex items-center gap-2 mt-2">
            <label className="text-[11px] text-slate-500">Max categories / column</label>
            <input
              type="number"
              min={2}
              max={200}
              value={c.max_category_cardinality}
              onChange={(e) =>
                onChange({ max_category_cardinality: Math.max(2, parseInt(e.target.value, 10) || 50) })
              }
              className="w-16 text-right text-xs border border-slate-200 rounded px-2 py-1"
            />
          </div>
          <p className="text-[11px] text-slate-400 mt-1">
            Higher-cardinality text/object columns are dropped before training when not using numeric-only mode.
          </p>
        </div>
      </section>

      {/* ── Validation strategy ── */}
      <section className="space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">Cross-Validation Strategy</h3>
        <Select
          value={c.val_strategy}
          onChange={(v) => onChange({ val_strategy: v })}
          options={VAL_STRATEGIES.map((s) => ({ value: s.value, label: s.label }))}
        />
        {currentStrategy && (
          <p className="text-[11px] text-slate-400 leading-relaxed">{currentStrategy.description}</p>
        )}

        {/* Fold / repeat / val_size params */}
        {needsFolds && (
          <div className="grid grid-cols-2 gap-3 pt-1">
            <div>
              <div className="flex items-baseline justify-between mb-1">
                <label className="text-xs font-medium text-slate-700">Folds (k)</label>
                <input
                  type="number" value={c.cv_folds} min={2} max={20} step={1}
                  onChange={(e) => onChange({ cv_folds: Math.max(2, parseInt(e.target.value) || 5) })}
                  className="w-14 text-right text-xs font-semibold border border-slate-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>
              <input
                type="range" min={2} max={20} step={1} value={c.cv_folds}
                onChange={(e) => onChange({ cv_folds: parseInt(e.target.value) })}
                className="w-full"
              />
              <p className="text-[11px] text-slate-400 mt-0.5">5 or 10 is standard.</p>
            </div>

            {needsRepeats && (
              <div>
                <div className="flex items-baseline justify-between mb-1">
                  <label className="text-xs font-medium text-slate-700">Repeats</label>
                  <input
                    type="number" value={c.cv_repeats} min={1} max={20} step={1}
                    onChange={(e) => onChange({ cv_repeats: Math.max(1, parseInt(e.target.value) || 3) })}
                    className="w-14 text-right text-xs font-semibold border border-slate-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                </div>
                <input
                  type="range" min={1} max={20} step={1} value={c.cv_repeats}
                  onChange={(e) => onChange({ cv_repeats: parseInt(e.target.value) })}
                  className="w-full"
                />
                <p className="text-[11px] text-slate-400 mt-0.5">3–5 repeats gives a stable estimate.</p>
              </div>
            )}

            {needsValSize && (
              <div>
                <div className="flex items-baseline justify-between mb-1">
                  <label className="text-xs font-medium text-slate-700">Val size</label>
                  <input
                    type="number" value={c.val_size} min={0.05} max={0.5} step={0.05}
                    onChange={(e) => onChange({ val_size: parseFloat(e.target.value) || 0.1 })}
                    className="w-14 text-right text-xs font-semibold border border-slate-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                </div>
                <input
                  type="range" min={0.05} max={0.5} step={0.05} value={c.val_size}
                  onChange={(e) => onChange({ val_size: parseFloat(e.target.value) })}
                  className="w-full"
                />
                <p className="text-[11px] text-slate-400 mt-0.5">Fraction used for each validation split.</p>
              </div>
            )}
          </div>
        )}
      </section>

      {/* ── Summary code snippet ── */}
      <section>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">Equivalent Python</h3>
        <div className="rounded-xl overflow-hidden border border-slate-800">
          <div className="flex items-center gap-1.5 px-4 py-2 bg-slate-900 border-b border-slate-700">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
            <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
            <span className="ml-2 text-[10px] text-slate-500 font-mono">Python · sklearn</span>
          </div>
          <pre className="bg-slate-950 text-slate-100 text-[11px] leading-relaxed px-5 py-4 overflow-x-auto font-mono whitespace-pre">
{hideHoldoutSplit
  ? `# Hold-out split is already in df["__ml_split__"] (from Transforms).
train_mask = df["__ml_split__"] == "train"
test_mask  = df["__ml_split__"] == "test"
X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_test,  y_test  = X.loc[test_mask],  y.loc[test_mask]
${c.scaler !== "none" ? `
from sklearn.preprocessing import ${scalerImport(c.scaler)}
scaler = ${scalerClass(c.scaler)}()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)` : ""}${c.val_strategy !== "none" ? `
from sklearn.model_selection import ${valImport(c)}
${cvCode(c, isClassification)}` : ""}`
  : `from sklearn.model_selection import train_test_split${c.val_strategy !== "none" ? `, ${valImport(c)}` : ""}
${c.scaler !== "none" ? `from sklearn.preprocessing import ${scalerImport(c.scaler)}` : "# no scaler"}

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=${c.test_size},
    random_state=${c.shuffle ? c.random_state : "None"},
    shuffle=${c.shuffle ? "True" : "False"},${isClassification && c.stratify ? "\n    stratify=y," : ""}
)
${c.scaler !== "none" ? `
scaler = ${scalerClass(c.scaler)}()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)` : ""}${c.val_strategy !== "none" ? `
${cvCode(c, isClassification)}` : ""}`}
          </pre>
        </div>
      </section>
    </div>
  );
}

// ── Toggle ────────────────────────────────────────────────────────────────────

function Toggle({ label, checked, onChange, disabled, description }: {
  label: string; checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean; description: string;
}) {
  return (
    <div className={`flex-1 p-3 rounded-lg border transition-colors ${
      disabled ? "opacity-40 cursor-not-allowed border-slate-200 bg-slate-50"
               : checked ? "border-blue-300 bg-blue-50" : "border-slate-200 bg-slate-50"
    }`}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-semibold text-slate-700">{label}</span>
        <button
          type="button"
          disabled={disabled}
          onClick={() => !disabled && onChange(!checked)}
          className={`w-10 h-6 rounded-full transition-colors relative overflow-hidden shrink-0 ${
            checked && !disabled ? "bg-blue-600" : "bg-slate-300"
          }`}
        >
          <span className={`absolute top-1 w-4 h-4 rounded-full bg-white shadow-sm transition-transform duration-200 ${
            checked ? "translate-x-5" : "translate-x-1"
          }`} />
        </button>
      </div>
      <p className="text-[10px] text-slate-400 leading-snug">{description}</p>
    </div>
  );
}

// ── Code generation helpers ───────────────────────────────────────────────────

function valImport(c: SplitConfig): string {
  const map: Record<string, string> = {
    kfold:                      "KFold",
    stratified_kfold:           "StratifiedKFold",
    repeated_kfold:             "RepeatedKFold",
    repeated_stratified_kfold:  "RepeatedStratifiedKFold",
    shuffle_split:              "ShuffleSplit",
    loo:                        "LeaveOneOut",
  };
  return map[c.val_strategy] ?? "KFold";
}

function scalerImport(s: string): string {
  return { standard: "StandardScaler", minmax: "MinMaxScaler", robust: "RobustScaler" }[s] ?? "StandardScaler";
}

function scalerClass(s: string): string {
  return { standard: "StandardScaler", minmax: "MinMaxScaler", robust: "RobustScaler" }[s] ?? "StandardScaler";
}

function cvCode(c: SplitConfig, clf: boolean): string {
  const cls = valImport(c);
  let init = "";
  if (c.val_strategy === "kfold")
    init = `cv = KFold(n_splits=${c.cv_folds}, shuffle=${c.shuffle ? "True" : "False"}, random_state=${c.random_state})`;
  else if (c.val_strategy === "stratified_kfold")
    init = `cv = StratifiedKFold(n_splits=${c.cv_folds}, shuffle=True, random_state=${c.random_state})`;
  else if (c.val_strategy === "repeated_kfold")
    init = `cv = RepeatedKFold(n_splits=${c.cv_folds}, n_repeats=${c.cv_repeats}, random_state=${c.random_state})`;
  else if (c.val_strategy === "repeated_stratified_kfold")
    init = `cv = RepeatedStratifiedKFold(n_splits=${c.cv_folds}, n_repeats=${c.cv_repeats}, random_state=${c.random_state})`;
  else if (c.val_strategy === "shuffle_split")
    init = `cv = ShuffleSplit(n_splits=${c.cv_folds}, test_size=${c.val_size}, random_state=${c.random_state})`;
  else if (c.val_strategy === "loo")
    init = `cv = LeaveOneOut()`;

  const scoring = clf ? '"accuracy"' : '"neg_root_mean_squared_error"';
  return `${init}
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=${scoring})
print(f"CV mean: {scores.mean():.4f} ± {scores.std():.4f}")`;
}
