from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib, json, pandas as pd, numpy as np
from pathlib import Path



# --- load artifacts ---
ART = Path("artifacts")
pipe      = joblib.load(ART / "rent_pipeline_xgb.pkl")
priors    = joblib.load(ART / "priors.pkl")
feat      = json.load(open(ART / "features.json"))
UPLIFT    = json.load(open(ART / "model_meta.json"))["uplift_factor"]

gmean = float(priors["gmean"])
city_prior_map = priors["city_prior"]
pc4_prior_map  = priors["pc4_prior"]
num_cols = feat["num_cols"]
cat_cols = feat["cat_cols"]

# --- FastAPI init ---
app = FastAPI(title="Rent Prediction API")

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Input schema ---
class RentInput(BaseModel):
    areaSqm: float
    latitude: float
    longitude: float
    city: str
    pc4: str
    propertyType: str
    furnish: str
    internet: str
    kitchen: str
    shower: str
    toilet: str
    living: str
    smokingInside: str
    pets: str

# --- Prediction route ---
@app.post("/predict")
def predict_rent(data: RentInput):
    row = pd.DataFrame([data.dict()])
    # basic normalization
    for c in ["city","propertyType","furnish","internet","kitchen","shower","toilet","living","smokingInside","pets"]:
        row[c] = row[c].astype(str).str.strip().str.lower()
    row["pc4"] = row["pc4"].astype(str).str.extract(r"(\d{4})")

    # priors
    row["city_prior"] = row["city"].map(city_prior_map).fillna(gmean)
    row["pc4_prior"]  = row["pc4"].map(pc4_prior_map).fillna(row["city_prior"]).fillna(gmean)

    X_new = row[num_cols + cat_cols].copy()
    X_new[cat_cols] = X_new[cat_cols].astype("object")

    pred = float(pipe.predict(X_new)[0])
    return {"predicted_rent": round(pred, 2)}


@app.get("/", response_class=HTMLResponse)
def home():
    return open("static/index.html", "r", encoding="utf-8").read()
