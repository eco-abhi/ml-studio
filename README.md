# ML Studio

**End-to-end ML workspace in the browser** — upload CSVs, explore and transform data, train and tune models with Optuna, compare runs in-app and in **MLflow**, inspect feature importances, and run predictions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)

---

## What you get

| Area | Capabilities |
|------|----------------|
| **Data** | CSV upload & URL import, preview, column stats |
| **EDA** | Health checks, distributions, correlations, skewness, pairplots, missing values, outliers |
| **Transforms** | Visual pipeline (impute, scale, encode, splits, …) with preview and revert |
| **Train** | Multi-model training (regression & classification), hyperparameters, train/test split options |
| **Tune** | Optuna hyperparameter search |
| **Experiments** | Compare metrics and charts without leaving the app |
| **Importances** | Model-specific and SHAP-style explanations where supported |
| **Predict** | Try trained models on custom feature inputs |
| **Tracking** | MLflow runs + optional local or remote UI |

---

## Repository layout

This repo is a small monorepo: the runnable app lives under **`ml-studio/`**.

```
.
├── README.md                 ← You are here (overview & quick start)
├── .gitignore
└── ml-studio/
    ├── README.md             ← Full guide: env vars, API, troubleshooting
    ├── backend/              # FastAPI, scikit-learn, MLflow, SHAP, XGBoost, Optuna
    ├── frontend/             # React 18, Vite, Tailwind, Radix-style UI
    ├── data/                 # Sample CSV for demos
    └── setup.sh              # Optional dependency bootstrap
```

---

## Quick start

**Prerequisites:** Python **3.10+**, **Node.js 18+**, and (optionally) an **MLflow** tracking UI if you want the external dashboard in addition to the in-app Experiments tab.

```bash
git clone https://github.com/eco-abhi/ml-studio.git
cd ml-studio
cd ml-studio                    # application package (nested folder)

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # edit if needed (see full README)
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd ../frontend
npm install
cp .env.example .env          # optional: API URL
npm run dev
```

- **App UI:** [http://localhost:5173](http://localhost:5173)  
- **API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)  
- **MLflow UI (optional):** e.g. `mlflow server --host 127.0.0.1 --port 5000` — align `--backend-store-uri` with your backend `.env` if you use a file or DB store.

---

## Documentation

For **environment variables**, **authentication**, **MLflow URI setup**, **API reference**, and **detailed usage**, see **[`ml-studio/README.md`](./ml-studio/README.md)** and **[`ml-studio/QUICKSTART.md`](./ml-studio/QUICKSTART.md)**.

---

## License

Add a `LICENSE` file at the repo root when you publish under a specific license.
