# 🚀 Quick Start Guide - ML Studio

Get up and running in **5 minutes**.

## Prerequisites

- **Python 3.8+** (check: `python3 --version`)
- **Node.js 16+** (check: `node --version`)
- **pip** and **npm** package managers

## Installation

### 1️⃣ Install Dependencies

```bash
# Backend
cd backend
pip install -r requirements.txt
cd ..

# Frontend
cd frontend
npm install
cd ..
```

**On macOS with ARM (M1/M2/M3):** You might need to install some packages with conda instead:
```bash
conda install scikit-learn pandas numpy mlflow shap xgboost
```

## Running the App

You'll need **3 terminal windows**:

### Terminal 1: MLflow Tracking Server

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Then open: **http://localhost:5000**

You should see the MLflow UI with an empty experiments list.

---

### Terminal 2: Backend API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
Uvicorn running on http://0.0.0.0:8000
```

Test it: **http://localhost:8000/health** → should return `{"status":"ok"}`

---

### Terminal 3: Frontend

```bash
cd frontend
npm run dev
```

You should see:
```
VITE v4.3.0  ready in 123 ms

➜  Local:   http://localhost:5173/
```

Open: **http://localhost:5173**

---

## 🎯 First Run: Upload & Train

### Step 1: Upload Data

1. Go to **Upload** page (should be default)
2. Choose one of these:
   - **CSV File**: Use `data/sample_wine.csv` (already included)
   - **URL**: Paste a Kaggle/UCI dataset URL
3. Click **Upload**

### Step 2: Configure Target

1. Select **Target Column** (what to predict)
   - For wine data: select `quality`
2. System auto-detects task type (regression/classification)
   - For wine: **regression** (0-10 score)
3. Click **Start Training**

### Step 3: Wait & Watch

Training takes ~30 seconds. You'll see:
- Terminal 2: Model training logs
- Backend creating 5 trained models
- All runs logged to MLflow

### Step 4: Explore Results

Once training is done:

#### 📊 **EDA Tab** (Explore Data)
- Select any feature
- See: mean, std, min, max, median
- View distribution histogram
- Check correlations

#### 🧪 **Experiments Tab** (Compare Models)
- All 5 models ranked by performance
- RMSE, MAE, R² metrics
- Best model highlighted

#### 🎯 **Importances Tab** (Feature Analysis)
- Select any model
- See which features matter most
- Green bars show importance score

#### 🔮 **Predict Tab** (Make Predictions)
- Select model & adjust feature sliders
- Click **Predict**
- Get instant prediction + confidence

---

## 📈 View in MLflow

For interactive charts and experiment management:

1. Open http://localhost:5000
2. Find "dataset-{id}" experiment
3. Click to see:
   - Metrics comparison charts
   - Parameter values
   - Model artifacts
   - Model registry (promote to production)

---

## 📂 Test with Different Datasets

### Built-in Sample
- Already in: `data/sample_wine.csv`
- 20 rows of wine quality data
- Perfect for testing

### Recommended Public Datasets

**Regression:**
- [Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- [Bike Sharing](https://www.kaggle.com/c/bike-sharing-demand/data)
- [Energy Efficiency](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)

**Classification:**
- [Iris Flowers](https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/iris.csv)
- [Titanic](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv)
- [Wine Quality (Binary)](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)

**How to use URLs:**
1. Upload → paste URL → click Upload
2. System downloads and processes the CSV

---

## 🛠️ Troubleshooting

### "Connection refused" errors?

**Problem:** Backend/MLflow not running

**Solution:**
- Check all 3 terminals are running
- MLflow: port 5000
- Backend: port 8000
- Frontend: port 5173

```bash
# Verify ports in use:
lsof -i :5000  # MLflow
lsof -i :8000  # Backend
lsof -i :5173  # Frontend
```

---

### "ModuleNotFoundError: No module named 'fastapi'"?

**Problem:** Dependencies not installed

**Solution:**
```bash
cd backend
pip install -r requirements.txt
# Or on ARM Mac:
pip install fastapi uvicorn mlflow scikit-learn pandas numpy shap xgboost requests python-multipart
```

---

### "No models found" in Experiments?

**Problem:** Training didn't complete

**Solution:**
- Wait 30 more seconds
- Check Terminal 2 for errors
- Refresh browser
- Check MLflow UI (port 5000) for experiment

---

### CSV upload fails?

**Problem:** File format or encoding issue

**Solution:**
- Ensure CSV is UTF-8 encoded
- No special characters in column names
- At least one numeric column
- Comma or semicolon separated (auto-detected)

---

## 🎓 What's Happening Under the Hood?

### Training Pipeline

1. **Upload** → File stored in `data/uploads/{id}/`
2. **Preview** → Dataset info sent to frontend
3. **Configure** → You pick target column
4. **Auto-detect** → Task type determined from target
5. **Train** → All 5 models fitted in backend:
   - Linear Regression
   - Ridge Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
6. **Log** → Each run logged to MLflow with:
   - Hyperparameters
   - Metrics (RMSE, MAE, R², CV scores)
   - Serialized model
   - Tags (dataset, task type, model)

### Inference Pipeline

1. **Load** → Best model loaded from MLflow registry
2. **Preprocess** → Features scaled (StandardScaler)
3. **Predict** → Model makes prediction
4. **Explain** → Feature importances computed:
   - Tree models: built-in Gini/gain importances
   - Linear models: SHAP-based explanations

---

## 🚀 Next Steps After First Run

### Try These Features:

1. **Train on Different Data**
   - Upload a classification dataset (Iris, Titanic)
   - System auto-detects task type
   - Compare accuracy/F1 scores instead of RMSE

2. **Use MLflow UI**
   - Go to http://localhost:5000
   - Compare metrics across runs
   - Create charts
   - Promote best model to "Production"

3. **Analyze Feature Importance**
   - Understand which inputs drive predictions
   - Compare importance across models
   - Identify key features for business

4. **Make Predictions**
   - Use best model to predict on new data
   - Adjust feature sliders
   - Get confidence scores

5. **Experiment with Hyperparameters**
   - Edit `train.py` (e.g., increase `n_estimators`)
   - Run training again
   - New runs appear in MLflow

---

## 📊 Example Workflow: Wine Quality Prediction

```
1. Upload data/sample_wine.csv
   ↓
2. Select target: "quality"
   ↓
3. System detects: regression
   ↓
4. Train 5 models (30 seconds)
   ↓
5. View results:
   - EDA: alcohol correlates with quality (+0.48)
   - Experiments: Random Forest wins (RMSE 0.62)
   - Importances: Alcohol is feature #1
   - Predict: Set alcohol=11 → quality≈6.2
```

---

## 💡 Pro Tips

1. **Use URL uploads** for reproducibility
2. **Check MLflow UI** for advanced charts
3. **Sort Experiments** by best metric
4. **Export predictions** for downstream use
5. **Track iterations** by uploading same data multiple times

---

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] MLflow server running (port 5000)
- [ ] Backend API running (port 8000)
- [ ] Frontend running (port 5173)
- [ ] Uploaded first dataset
- [ ] Selected target column
- [ ] Training completed
- [ ] Viewed EDA/Experiments/Importances
- [ ] Made first prediction

---

**Ready to explore?** Open http://localhost:5173 and upload your first dataset! 🎉
