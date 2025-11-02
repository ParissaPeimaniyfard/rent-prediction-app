# monitor.py
import time
from contextlib import contextmanager
from fastapi import APIRouter, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

router = APIRouter()

# --- Metrics we will use from /predict ---
PRED_REQUESTS = Counter(
    "pred_requests_total",
    "Number of prediction requests received."
)

PRED_ERRORS = Counter(
    "pred_errors_total",
    "Number of prediction requests that raised an error."
)

PRED_LATENCY = Histogram(
    "pred_latency_seconds",
    "Prediction latency in seconds."
)

MODEL_VERSION = Gauge(
    "model_version_info",
    "Deployed model version",
    ["version"]
)

# after PRED_LATENCY
FEEDBACK_SUBMITTED = Counter(
    "feedback_submitted_total",
    "Number of feedback submissions received."
)


@contextmanager
def predict_timer():
    """Use as: with predict_timer():  model.predict(...)"""
    t0 = time.time()
    try:
        yield
    finally:
        PRED_LATENCY.observe(time.time() - t0)

# --- Metrics endpoint (no change) ---
@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)




