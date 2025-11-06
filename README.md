# NL Rent Prediction — FastAPI + XGBoost

**Live demo:** https://rent-prediction-app-production.up.railway.app
**OpenAPI docs:** https://rent-prediction-app-production.up.railway.app/docs  
**Metrics:** https://rent-prediction-app-production.up.railway.app/metrics  
**Version:** https://rent-prediction-app-production.up.railway.app/version

[![CI](https://github.com/ParissaPeimaniyfard/rent-prediction-app/actions/workflows/ci.yml/badge.svg)](https://github.com/ParissaPeimaniyfard/rent-prediction-app/actions/workflows/ci.yml)

---

## ✨ What’s inside

- Trained **XGBoost** model with reproducible preprocessing and location priors  
- **FastAPI web app** with minimal HTML interface  
- **Monitoring endpoints:** `/metrics`, `/version`  
- **Structured logging** with `request_id`  
- **Feedback form** (user ratings + actual rent)  
- **CI tests** via GitHub Actions  
- Deployed publicly on **Railway**

---

## 🧠 Problem & Approach

**Goal:** Predict monthly rent (EUR) for Dutch rental listings.

**Dataset:** Kaggle NL Rentals (2019 – 2020).  
Created a target `rent_adj` = historical rent × **1.50 uplift** (to account for market growth).

**Features used**
- Numeric: `areaSqm`, `latitude`, `longitude`
- Categorical: `propertyType`, `furnish`, `internet`, `kitchen`, `shower`, `toilet`, `living`, `smokingInside`, `pets`
- Location priors: smoothed city mean and PC4 mean (with back-off to global mean)

**Model:** XGBoost Regressor in a scikit-learn Pipeline with ColumnTransformer.  
**Performance:** R² ≈ 0.845, MAE ≈ €142 (on held-out test split).

---

## 🏗️ Project Structure

.
├─ app.py # FastAPI app (main entry)
├─ monitor.py # Metrics endpoints (/metrics, /version)
├─ logging_setup.py # Structured logging (request_id, events)
├─ static/
│ └─ index.html # Web form + feedback card
├─ artifacts/ # Saved model + metadata
│ ├─ rent_pipeline_xgb.pkl
│ ├─ priors.pkl
│ ├─ features.json
│ └─ model_meta.json # {"uplift_factor": 1.50}
├─ tests/
│ └─ test_smoke.py # CI smoke tests (/version, /predict)
├─ .github/workflows/ci.yml # GitHub Actions CI setup
├─ requirements.txt
└─ README.md

---

## 🚀 Run Locally

```bash
# 1️⃣ (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Start the API locally
uvicorn app:app --reload

# 4️⃣ Open these in your browser:
# UI:        http://127.0.0.1:8000/
# Docs:      http://127.0.0.1:8000/docs
# Metrics:   http://127.0.0.1:8000/metrics
# Version:   http://127.0.0.1:8000/version

---

## 🌐 Deploy on Railway

1️⃣ Push your project to GitHub.  
2️⃣ Go to [https://railway.app](https://railway.app) and create a new project.  
3️⃣ Link it to your GitHub repo.  
4️⃣ Add the following in your **Settings ▸ Variables**:

PORT = 8000

5️⃣ In **Settings ▸ Deployments**, set this **Start Command**:

uvicorn app:app --host 0.0.0.0 --port $PORT

6️⃣ Add a file `runtime.txt` with `3.10` to lock the Python version.  
7️⃣ Deploy — Railway will give you a public URL like:
https://rent-prediction-app-production.up.railway.app/
8️⃣ Visit `/`, `/docs`, `/metrics`, `/version` to test everything.

---

## 📈 Observability (Metrics & Logs)

- **/metrics** (Prometheus format):  
  - `pred_requests_total` — count of predictions  
  - `pred_errors_total` — failed predictions  
  - `pred_latency_seconds_*` — latency histogram  
  - `model_version_info{version="v1"} 1` — current model version
- **/version** — quick human-readable version JSON
- **Structured logs** (visible in Railway ▸ Logs): JSON lines with:
  - `event` (`predict_request`, `predict_success`, `predict_error`, `feedback_submitted`)
  - `request_id` (to correlate logs per request)
  - `model_version`

---
 
## 🧪 Tests & Continuous Integration (CI)

This project includes lightweight smoke tests to ensure the API starts and responds correctly.

Run locally:
```bash
pytest -q

Tests cover:

- /version returns 200 and includes "model_version"

- /predict returns 200 and outputs "predicted_rent"

CI is configured with GitHub Actions (.github/workflows/ci.yml) and runs automatically on each push.
Status is shown by the badge at the top of this README.
