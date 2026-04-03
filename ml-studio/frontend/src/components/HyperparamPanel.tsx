import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import type { TaskType } from "../api";
import { Select } from "./ui/select";

// ── Types ─────────────────────────────────────────────────────────────────────

interface HParamBase {
  label: string;
  key: string;
  description: string;
}

interface HParamNumeric extends HParamBase {
  type: "float" | "int";
  min: number;
  max: number;
  step: number;
  default: number;
}

interface HParamSelect extends HParamBase {
  type: "select";
  options: { value: string; label: string }[];
  default: string;
}

export type HParam = HParamNumeric | HParamSelect;

export interface ModelMeta {
  label: string;
  color: string;
  regression: HParam[];
  classification: HParam[];
}

// ── Shared param definitions ──────────────────────────────────────────────────

const MIN_SAMPLES_LEAF: HParam = {
  label: "Min Samples Leaf", key: "min_samples_leaf", type: "int",
  min: 1, max: 50, step: 1, default: 1,
  description: "Minimum samples required at a leaf node. Higher values smooth the model and prevent tiny leaves.",
};

const CCP_ALPHA: HParam = {
  label: "CCP Alpha (pruning)", key: "ccp_alpha", type: "float",
  min: 0, max: 0.05, step: 0.001, default: 0.0,
  description: "Complexity cost for pruning. Higher values prune more aggressively; 0 = no pruning.",
};

const CRITERION_REGRESSOR: HParam = {
  label: "Criterion", key: "criterion", type: "select",
  options: [
    { value: "squared_error", label: "Squared Error (MSE)" },
    { value: "friedman_mse",  label: "Friedman MSE" },
    { value: "absolute_error",label: "Absolute Error (MAE)" },
    { value: "poisson",       label: "Poisson Deviance" },
  ],
  default: "squared_error",
  description: "Split quality function. Squared Error is fastest; Friedman MSE can improve on it for trees.",
};

const CRITERION_CLASSIFIER: HParam = {
  label: "Criterion", key: "criterion", type: "select",
  options: [
    { value: "gini",      label: "Gini Impurity" },
    { value: "entropy",   label: "Entropy (Information Gain)" },
    { value: "log_loss",  label: "Log Loss" },
  ],
  default: "gini",
  description: "Split quality function. Gini and Entropy usually give similar results.",
};

const MAX_FEATURES_TREE: HParam = {
  label: "Max Features", key: "max_features", type: "select",
  options: [
    { value: "sqrt",  label: "sqrt(n_features)" },
    { value: "log2",  label: "log2(n_features)" },
    { value: "none",  label: "All features" },
  ],
  default: "sqrt",
  description: "Number of features to consider per split. sqrt/log2 add randomness and speed; 'All' uses every feature.",
};

const TREE_COMMON_REG: HParam[] = [
  { label: "Max Depth",         key: "max_depth",         type: "int",   min: 0,  max: 30, step: 1, default: 5,
    description: "Maximum tree depth. 0 = unlimited (can overfit). Start at 3–6." },
  { label: "Min Samples Split", key: "min_samples_split", type: "int",   min: 2,  max: 50, step: 1, default: 2,
    description: "Minimum samples required to attempt a split." },
  MIN_SAMPLES_LEAF,
  CRITERION_REGRESSOR,
  MAX_FEATURES_TREE,
  CCP_ALPHA,
];

const TREE_COMMON_CLF: HParam[] = [
  { label: "Max Depth",         key: "max_depth",         type: "int",   min: 0,  max: 30, step: 1, default: 5,
    description: "Maximum tree depth. 0 = unlimited (can overfit). Start at 3–6." },
  { label: "Min Samples Split", key: "min_samples_split", type: "int",   min: 2,  max: 50, step: 1, default: 2,
    description: "Minimum samples required to attempt a split." },
  MIN_SAMPLES_LEAF,
  CRITERION_CLASSIFIER,
  MAX_FEATURES_TREE,
  CCP_ALPHA,
];

const RF_EXTRA: HParam[] = [
  { label: "N Estimators",  key: "n_estimators",  type: "int",   min: 10,  max: 500, step: 10,   default: 100,
    description: "Number of trees. More = better & slower. Diminishing returns after ~300." },
  { label: "Max Samples",   key: "max_samples",   type: "float", min: 0.1, max: 1.0, step: 0.05, default: 1.0,
    description: "Fraction of rows sampled per tree (when bootstrap=True). <1.0 = bagging." },
  {
    label: "Bootstrap", key: "bootstrap", type: "select",
    options: [{ value: "true", label: "True (bagging)" }, { value: "false", label: "False (no replacement)" }],
    default: "true",
    description: "Whether to use bootstrap samples. True enables bagging; False uses the full training set per tree.",
  },
];

const GB_COMMON: HParam[] = [
  { label: "N Estimators",        key: "n_estimators",        type: "int",   min: 10,   max: 500,  step: 10,   default: 100,
    description: "Number of boosting stages (trees). More = better but slower and risks overfitting without shrinkage." },
  { label: "Learning Rate",       key: "learning_rate",       type: "float", min: 0.01, max: 1.0,  step: 0.01, default: 0.1,
    description: "Shrinks each tree's contribution. Lower (0.01–0.05) + more estimators often beats a high rate." },
  { label: "Max Depth",           key: "max_depth",           type: "int",   min: 1,    max: 10,   step: 1,    default: 3,
    description: "Max depth per tree. GBM trees are typically shallow (3–5)." },
  { label: "Subsample",           key: "subsample",           type: "float", min: 0.1,  max: 1.0,  step: 0.05, default: 1.0,
    description: "Row fraction sampled per stage. <1.0 = stochastic boosting, adds regularisation." },
  { label: "Min Samples Split",   key: "min_samples_split",   type: "int",   min: 2,    max: 50,   step: 1,    default: 2,
    description: "Min samples required to split an internal node." },
  MIN_SAMPLES_LEAF,
  {
    label: "Max Features", key: "max_features", type: "select",
    options: [
      { value: "none",  label: "All features" },
      { value: "sqrt",  label: "sqrt(n_features)" },
      { value: "log2",  label: "log2(n_features)" },
    ],
    default: "none",
    description: "Features considered per split. Subset adds randomness and can prevent overfitting.",
  },
  { label: "Validation Fraction", key: "validation_fraction", type: "float", min: 0.01, max: 0.5,  step: 0.01, default: 0.1,
    description: "Fraction of training data held out for early stopping (only when n_iter_no_change > 0)." },
  { label: "N Iter No Change",    key: "n_iter_no_change",    type: "int",   min: 0,    max: 50,   step: 1,    default: 0,
    description: "Early stopping: stop after this many rounds with no val improvement. 0 = disabled." },
  CCP_ALPHA,
];

// ── Model meta ────────────────────────────────────────────────────────────────

export const MODEL_META: Record<string, ModelMeta> = {
  linear_regression: {
    label: "Linear Regression", color: "#3b82f6",
    regression: [
      {
        label: "Fit Intercept", key: "fit_intercept", type: "select",
        options: [{ value: "true", label: "True" }, { value: "false", label: "False" }],
        default: "true",
        description: "Whether to calculate the intercept. Set to False if data is already centred.",
      },
    ],
    classification: [],
  },

  ridge: {
    label: "Ridge", color: "#10b981",
    regression: [
      { label: "Alpha (λ)", key: "alpha", type: "float", min: 0.001, max: 100, step: 0.1, default: 1.0,
        description: "L2 regularisation strength. Higher shrinks coefficients more, reducing overfitting." },
      { label: "Max Iter",  key: "max_iter", type: "int", min: 100, max: 10000, step: 100, default: 1000,
        description: "Max iterations for iterative solvers (sag, saga, lsqr). Increase if not converging." },
      {
        label: "Solver", key: "solver", type: "select",
        options: [
          { value: "auto",      label: "Auto (auto-select)" },
          { value: "svd",       label: "SVD" },
          { value: "cholesky",  label: "Cholesky" },
          { value: "lsqr",      label: "LSQR" },
          { value: "sparse_cg", label: "Sparse CG" },
          { value: "sag",       label: "SAG (large datasets)" },
          { value: "saga",      label: "SAGA" },
          { value: "lbfgs",     label: "L-BFGS" },
        ],
        default: "auto",
        description: "Solver for the weight matrix. 'auto' chooses based on data size. SAG/SAGA suit large datasets.",
      },
      {
        label: "Fit Intercept", key: "fit_intercept", type: "select",
        options: [{ value: "true", label: "True" }, { value: "false", label: "False" }],
        default: "true",
        description: "Whether to fit an intercept term. False if data is mean-centred.",
      },
    ],
    classification: [],
  },

  logistic_regression: {
    label: "Logistic Regression", color: "#3b82f6",
    regression: [],
    classification: [
      { label: "C (inverse λ)", key: "C", type: "float", min: 0.001, max: 100, step: 0.1, default: 1.0,
        description: "Inverse regularisation strength. Smaller = stronger penalty. Like 1/alpha for Ridge." },
      { label: "Max Iterations", key: "max_iter", type: "int", min: 100, max: 5000, step: 100, default: 1000,
        description: "Max iterations for the solver. Increase if you see convergence warnings." },
      {
        label: "Penalty", key: "penalty", type: "select",
        options: [
          { value: "l2",         label: "L2 (Ridge)" },
          { value: "l1",         label: "L1 (Lasso) — needs liblinear/saga" },
          { value: "elasticnet", label: "Elastic Net — needs saga" },
          { value: "none",       label: "None (no regularisation)" },
        ],
        default: "l2",
        description: "Regularisation type. L2 is the standard; L1 produces sparse weights; Elastic Net combines both.",
      },
      {
        label: "Solver", key: "solver", type: "select",
        options: [
          { value: "lbfgs",           label: "L-BFGS (default, L2/no reg)" },
          { value: "liblinear",       label: "Liblinear (small datasets, L1/L2)" },
          { value: "newton-cg",       label: "Newton-CG (L2/no reg)" },
          { value: "newton-cholesky", label: "Newton-Cholesky (L2)" },
          { value: "sag",             label: "SAG (large, L2)" },
          { value: "saga",            label: "SAGA (large, L1/L2/EN)" },
        ],
        default: "lbfgs",
        description: "Optimisation algorithm. Must be compatible with the chosen penalty. lbfgs is a safe default.",
      },
      { label: "Tol",       key: "tol",       type: "float", min: 1e-6, max: 0.01, step: 1e-5, default: 1e-4,
        description: "Convergence tolerance. Smaller = more precise but slower." },
      { label: "L1 Ratio",  key: "l1_ratio",  type: "float", min: 0.0, max: 1.0,  step: 0.05, default: 0.5,
        description: "Mix of L1 vs L2 for Elastic Net penalty. 0 = pure L2, 1 = pure L1." },
    ],
  },

  ridge_classifier: {
    label: "Ridge Classifier", color: "#10b981",
    regression: [],
    classification: [
      { label: "Alpha (λ)",  key: "alpha",    type: "float", min: 0.001, max: 100,   step: 0.1,  default: 1.0,
        description: "Regularisation strength. Higher shrinks class weights toward zero." },
      { label: "Max Iter",   key: "max_iter", type: "int",   min: 100,   max: 10000, step: 100,  default: 1000,
        description: "Max iterations for iterative solvers. Increase if not converging." },
      {
        label: "Solver", key: "solver", type: "select",
        options: [
          { value: "auto",      label: "Auto" },
          { value: "svd",       label: "SVD" },
          { value: "cholesky",  label: "Cholesky" },
          { value: "lsqr",      label: "LSQR" },
          { value: "sparse_cg", label: "Sparse CG" },
          { value: "sag",       label: "SAG" },
          { value: "saga",      label: "SAGA" },
          { value: "lbfgs",     label: "L-BFGS" },
        ],
        default: "auto",
        description: "Algorithm used to solve the linear system. 'auto' is fine for most cases.",
      },
      {
        label: "Fit Intercept", key: "fit_intercept", type: "select",
        options: [{ value: "true", label: "True" }, { value: "false", label: "False" }],
        default: "true",
        description: "Whether to fit a bias/intercept term.",
      },
    ],
  },

  decision_tree: {
    label: "Decision Tree", color: "#f59e0b",
    regression:     TREE_COMMON_REG,
    classification: TREE_COMMON_CLF,
  },

  random_forest: {
    label: "Random Forest", color: "#8b5cf6",
    regression: [
      ...RF_EXTRA,
      { label: "Min Samples Split", key: "min_samples_split", type: "int", min: 2, max: 50, step: 1, default: 2,
        description: "Min samples to split an internal node." },
      MIN_SAMPLES_LEAF,
      { ...CRITERION_REGRESSOR },
      MAX_FEATURES_TREE,
      { label: "Max Depth", key: "max_depth", type: "int", min: 0, max: 30, step: 1, default: 0,
        description: "Max depth per tree. 0 = unlimited. Limiting depth prevents overfitting." },
      CCP_ALPHA,
    ],
    classification: [
      ...RF_EXTRA,
      { label: "Min Samples Split", key: "min_samples_split", type: "int", min: 2, max: 50, step: 1, default: 2,
        description: "Min samples to split an internal node." },
      MIN_SAMPLES_LEAF,
      { ...CRITERION_CLASSIFIER },
      MAX_FEATURES_TREE,
      { label: "Max Depth", key: "max_depth", type: "int", min: 0, max: 30, step: 1, default: 0,
        description: "Max depth per tree. 0 = unlimited." },
      CCP_ALPHA,
    ],
  },

  gradient_boosting: {
    label: "Gradient Boosting", color: "#ef4444",
    regression:     GB_COMMON,
    classification: GB_COMMON,
  },
};

export const REGRESSION_MODELS     = ["linear_regression","ridge","decision_tree","random_forest","gradient_boosting"];
export const CLASSIFICATION_MODELS = ["logistic_regression","ridge_classifier","decision_tree","random_forest","gradient_boosting"];

// ── Component ─────────────────────────────────────────────────────────────────

// Values can be numbers (sliders) or strings (selects)
export type ParamValues = Record<string, number | string>;

interface HyperparamPanelProps {
  modelKey: string;
  taskType: TaskType | null;
  selected: boolean;
  onToggleSelect: () => void;
  values: ParamValues;
  onChange: (key: string, val: number | string) => void;
}

export function HyperparamPanel({
  modelKey, taskType, selected, onToggleSelect, values, onChange,
}: HyperparamPanelProps) {
  const [open, setOpen] = useState(false);
  const meta = MODEL_META[modelKey];
  if (!meta) return null;

  const params = taskType === "classification" ? meta.classification : meta.regression;

  return (
    <div className={`rounded-lg border overflow-hidden transition-colors ${
      selected ? "border-blue-300" : "border-slate-200 opacity-60"
    }`}>
      {/* Header row */}
      <div className={`flex items-center gap-2 px-3 py-2.5 ${selected ? "bg-blue-50" : "bg-slate-50"}`}>
        {/* Checkbox */}
        <button
          onClick={onToggleSelect}
          className={`w-4 h-4 rounded border-2 shrink-0 flex items-center justify-center transition-colors ${
            selected ? "bg-blue-600 border-blue-600" : "border-slate-300 bg-white"
          }`}
          title={selected ? "Deselect model" : "Select model"}
        >
          {selected && (
            <svg className="w-2.5 h-2.5 text-white" viewBox="0 0 10 10" fill="none">
              <path d="M2 5l2.5 2.5L8 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          )}
        </button>

        <span className="w-2 h-2 rounded-full shrink-0" style={{ background: meta.color }} />
        <span className="text-sm font-medium text-slate-700 flex-1">{meta.label}</span>

        {/* Expand / collapse (only when selected and has params) */}
        {selected && params.length > 0 && (
          <button
            onClick={() => setOpen((o) => !o)}
            className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700 transition-colors"
          >
            <span>{params.length} param{params.length > 1 ? "s" : ""}</span>
            {open ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          </button>
        )}
        {selected && !params.length && (
          <span className="text-xs text-slate-400">No hyperparameters</span>
        )}
        {!selected && (
          <span className="text-xs text-slate-400 italic">skipped</span>
        )}
      </div>

      {/* Params body */}
      {selected && open && params.length > 0 && (
        <div className="px-4 py-3 space-y-4 bg-white border-t border-slate-100">
          {params.map((p) => (
            <ParamRow key={p.key} param={p} values={values} onChange={onChange} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── ParamRow ──────────────────────────────────────────────────────────────────

function ParamRow({
  param, values, onChange,
}: {
  param: HParam;
  values: ParamValues;
  onChange: (key: string, val: number | string) => void;
}) {
  if (param.type === "select") {
    const val = (values[param.key] as string) ?? param.default;
    return (
      <div>
        <label className="block text-xs font-semibold text-slate-700 mb-1">{param.label}</label>
        <Select
          value={val}
          onChange={(v) => onChange(param.key, v)}
          options={param.options}
          size="sm"
        />
        <p className="text-[11px] text-slate-400 mt-1">{param.description}</p>
      </div>
    );
  }

  const val = (values[param.key] as number) ?? param.default;
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <label className="text-xs font-semibold text-slate-700">{param.label}</label>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-400">[{param.min} – {param.max}]</span>
          <input
            type="number"
            value={val}
            step={param.step}
            min={param.min}
            max={param.max}
            onChange={(e) => onChange(param.key, parseFloat(e.target.value) || param.default)}
            className="w-20 text-right text-xs font-semibold border border-slate-200 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
      </div>
      <input
        type="range"
        min={param.min} max={param.max} step={param.step} value={val}
        onChange={(e) => onChange(param.key, parseFloat(e.target.value))}
        className="w-full"
      />
      <p className="text-[11px] text-slate-400 mt-1">{param.description}</p>
    </div>
  );
}
