# 🔬 ML Studio - Universal ML Comparison Platform

A full-stack web application for uploading datasets, training multiple ML models, comparing performance, analyzing features, and making predictions — all with integrated MLflow tracking.

## Features

- **📤 Upload Datasets**: Upload CSV files or link to URLs
- **🔍 Data Preview**: View dataset structure, statistics, and data types
- **🧪 Auto-train Models**: Train 5 different models automatically (Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting)
- **📊 EDA**: Explore distributions, statistics, and correlations
- **🧪 Compare Experiments**: Side-by-side model comparison with metrics
- **🎯 Feature Importances**: Understand which features drive predictions (SHAP + built-in importances)
- **🔮 Make Predictions**: Test trained models on new data
- **📈 MLflow Integration**: Full experiment tracking with model registry

## Tech Stack

**Backend:**
- FastAPI (Python web framework)
- scikit-learn (ML models)
- MLflow (experiment tracking)
- SHAP (feature explanations)
- XGBoost (gradient boosting)

**Frontend:**
- React 18
- Vite (dev server)
- Vanilla CSS (no dependencies)

## Quick Start

### 1. Clone & Setup

```bash
cd ml-studio
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Start MLflow Tracking Server

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Open in browser: http://localhost:5000

### 5. Start Backend API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 6. Start Frontend

```bash
cd frontend
npm run dev
```

Open in browser: http://localhost:5173

---

## Usage

### Step 1: Upload a Dataset

1. Go to **Upload** tab
2. Choose **CSV file upload** or **URL input**
3. Click **Upload**
4. Preview the data

### Step 2: Configure & Train

1. Select **target column** (what to predict)
2. System auto-detects task type (regression/classification), but you can override
3. Click **Start Training**
4. Wait for all 5 models to train

### Step 3: Analyze Results

**📊 EDA Tab:**
- View statistics for each feature (mean, std, min, max, median)
- See data distribution histograms
- Explore correlation with other features

**🧪 Experiments Tab:**
- Compare all models side-by-side
- View metrics (RMSE/Accuracy, MAE/F1, R²)
- Best model ranking

**🎯 Importances Tab:**
- Select any trained model
- View which features matter most
- Tree models: built-in feature importance
- Linear models: SHAP-based importance

**🔮 Predict Tab:**
- Choose a model
- Adjust feature sliders
- Get instant predictions with confidence scores

---

## API Endpoints

### Dataset Management
- `POST /upload` - Upload CSV file or URL
- `GET /preview/{dataset_id}` - Preview data
- `GET /datasets` - List all datasets

### Training & MLflow
- `POST /train/{dataset_id}` - Train all models
- `GET /experiments/{dataset_id}` - Get all runs

### Analysis
- `GET /eda/{dataset_id}` - Get statistics & correlations
- `GET /importances/{dataset_id}/{model_name}` - Feature importances

### Inference
- `POST /predict/{dataset_id}/{model_name}` - Make predictions

---

## Example Datasets

### Regression
- **Wine Quality**: Red/white wine features → quality score (0-10)
- **House Prices**: Property features → price
- **Stock Returns**: Technical indicators → returns

### Classification
- **Iris**: Flower measurements → species
- **Breast Cancer**: Medical features → malignant/benign
- **Titanic**: Passenger info → survived/died

---

## Project Structure

```
ml-studio/
├── backend/
│   ├── main.py              # FastAPI app + all routes
│   ├── train.py             # Training pipeline (regression + classification)
│   ├── eda.py               # EDA statistics computation
│   ├── predict.py           # Inference & feature importance
│   ├── requirements.txt
│   └── mlruns/              # Auto-created MLflow runs
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main router
│   │   ├── api.js           # API calls
│   │   └── pages/
│   │       ├── Upload.jsx   # Upload & train
│   │       ├── EDA.jsx      # Data analysis
│   │       ├── Experiments.jsx # Model comparison
│   │       ├── Importances.jsx # Feature analysis
│   │       └── Predict.jsx  # Inference
│   ├── index.html
│   ├── package.json
│   └── vite.config.js (if using Vite)
└── data/
    └── uploads/             # User-uploaded datasets
```

---

## Advanced Features

### Auto-detect Task Type
- Regression: Continuous targets with many unique values
- Classification: Discrete targets with few unique values
- Override manually if needed

### MLflow Integration
- Every model run is logged with:
  - Parameters (model type, dataset name)
  - Metrics (RMSE, MAE, R², accuracy, F1, AUC)
  - Serialized model (for loading later)
  - Tags (dataset, task type, model family)

- View interactive charts at http://localhost:5000
- Promote best models to "Production" registry

### Feature Importance Methods
- **Tree models**: Gini/gain-based importances (built-in)
- **Linear models**: SHAP LocalExplainer (model-agnostic)
- Normalized to 0-1 scale for comparison

---

## Troubleshooting

**"Connection refused" error?**
- Make sure MLflow server is running on port 5000
- Check backend is running on port 8000
- Check frontend is running on port 5173

**Models not appearing in Experiments?**
- Wait for training to complete (check terminal for logs)
- Refresh the page
- Check MLflow UI for runs

**CSV upload fails?**
- Ensure file is UTF-8 encoded
- Check for special characters in column names
- Ensure at least one numeric column

**Prediction shows "Model not found"?**
- Make sure you trained at least one model
- Wait for training to complete

---

## Next Steps

- Add hyperparameter tuning UI
- Support categorical features (one-hot encoding)
- Add ensemble methods
- Deploy to cloud (Heroku, AWS, Google Cloud)
- Add real-time monitoring dashboards

---

**Made with ❤️ for ML practitioners who value clarity and reproducibility.**
