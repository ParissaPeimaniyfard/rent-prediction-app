from fastapi import FastAPI
import uuid
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib, json, pandas as pd, numpy as np
from logging_setup import logger
from pathlib import Path
from monitor import PRED_REQUESTS, PRED_ERRORS, predict_timer, MODEL_VERSION



# --- load artifacts ---
ART = Path("artifacts")
pipe      = joblib.load(ART / "rent_pipeline_xgb.pkl")
priors    = joblib.load(ART / "priors.pkl")
feat      = json.load(open(ART / "features.json"))
UPLIFT    = json.load(open(ART / "model_meta.json"))["uplift_factor"]
MODEL_VER = "v1"
MODEL_VERSION.labels(version=MODEL_VER).set(1)



gmean = float(priors["gmean"])
city_prior_map = priors["city_prior"]
pc4_prior_map  = priors["pc4_prior"]
num_cols = feat["num_cols"]
cat_cols = feat["cat_cols"]

# --- FastAPI init ---
app = FastAPI(title="Rent Prediction API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include monitoring router
from monitor import router as monitor_router
app.include_router(monitor_router)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())[:8]  # short id like 'a3f7b9c2'
    response = await call_next(request)
    return response


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
def predict_rent(data: RentInput, request: Request):
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

    
    # ---- metrics + logging ----
    PRED_REQUESTS.inc()  # count every prediction request

    # Log request (basic fields only)
    logger.info(
        "predict_request",
        extra={
            "event": "predict_request",
            "model_version": MODEL_VER,
            "request_id": request.state.request_id,
            "area": float(row["areaSqm"].iloc[0]),
            "city": str(row["city"].iloc[0]),
            "pc4": str(row["pc4"].iloc[0]),
        },
    )

    try:
        with predict_timer():  # measure prediction time
            pred = float(pipe.predict(X_new)[0])

        # Log success
        logger.info(
            "predict_success",
            extra={
                "event": "predict_success",
                "model_version": MODEL_VER,
                "request_id": request.state.request_id,
                "predicted_rent": round(pred, 2),
            },
        )

        return {"predicted_rent": round(pred, 2)}

    except Exception:
        PRED_ERRORS.inc()  # count any failed predictions
        logger.error(
            "predict_error",
            extra={
                "event": "predict_error",
                "model_version": MODEL_VER,
                "request_id": request.state.request_id,
            },
        )
        raise




@app.get("/", response_class=HTMLResponse)
def home():
    return open("static/index.html", "r", encoding="utf-8").read()


# --- Version endpoint 
@app.get("/version")
def version():
    return {"model_version": "v1"}