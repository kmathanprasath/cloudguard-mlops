"""
CloudGuard FastAPI Inference Service
Serves the Isolation Forest anomaly detection model with MITRE ATT&CK mapping.
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cloudguard")

app = FastAPI(
    title="CloudGuard Threat Detection API",
    description="AI-powered cloud threat detection using Isolation Forest",
    version="1.0.0",
)

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "/models/iforest.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "/models/scaler.pkl")
THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.5"))

model = None
scaler = None

FEATURE_COLS = [
    "hour", "day_of_week", "is_weekend", "is_offhours",
    "is_failed", "is_rare_ip", "user_enc", "action_enc",
    "source_type_enc", "action_count_1h",
]

MITRE_RULES = [
    {
        "ttp": "T1110",
        "name": "Brute Force",
        "tactic": "Credential Access",
        "condition": lambda r: r.is_failed and r.action_count_1h > 20,
    },
    {
        "ttp": "T1485",
        "name": "Data Destruction",
        "tactic": "Impact",
        "condition": lambda r: "delete" in r.action.lower() and r.is_offhours,
    },
    {
        "ttp": "T1530",
        "name": "Data from Cloud Storage",
        "tactic": "Collection",
        "condition": lambda r: "list" in r.action.lower() and r.is_rare_ip,
    },
    {
        "ttp": "T1078",
        "name": "Valid Accounts",
        "tactic": "Defense Evasion",
        "condition": lambda r: r.is_offhours and r.is_rare_ip and not r.is_failed,
    },
]


@app.on_event("startup")
def load_model():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        logger.warning(f"Model not found at {MODEL_PATH} — running in demo mode")

    try:
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Scaler loaded from {SCALER_PATH}")
    except FileNotFoundError:
        logger.warning(f"Scaler not found at {SCALER_PATH} — raw features will be used")


# ── Schemas ────────────────────────────────────────────────────────────────────
class LogEvent(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon)")
    is_weekend: int = Field(..., ge=0, le=1)
    is_offhours: int = Field(..., ge=0, le=1)
    is_failed: int = Field(..., ge=0, le=1, description="Whether the action failed")
    is_rare_ip: int = Field(..., ge=0, le=1, description="Source IP seen < 3 times")
    user_enc: int = Field(..., ge=0, description="Label-encoded user identity")
    action_enc: int = Field(..., ge=0, description="Label-encoded action/event type")
    source_type_enc: int = Field(..., ge=0, description="Label-encoded source (0=cloudtrail,1=k8s,2=auth)")
    action_count_1h: int = Field(..., ge=0, description="Action count in last 1 hour")
    # Raw fields for MITRE mapping (not fed to model)
    action: Optional[str] = Field(default="", description="Raw action string for TTP mapping")


class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    ttp_detections: list
    message: str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/")
def root():
    return {"service": "CloudGuard Threat Detection API", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(event: LogEvent):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array([[
        event.hour, event.day_of_week, event.is_weekend, event.is_offhours,
        event.is_failed, event.is_rare_ip, event.user_enc, event.action_enc,
        event.source_type_enc, event.action_count_1h,
    ]], dtype=float)

    if scaler is not None:
        features = scaler.transform(features)

    # Isolation Forest: score_samples returns negative anomaly scores
    raw_score = model.score_samples(features)[0]
    # Normalise to 0-1 (higher = more anomalous)
    anomaly_score = float(1 - (raw_score - (-0.5)) / (0.5 - (-0.5)))
    anomaly_score = max(0.0, min(1.0, anomaly_score))
    is_anomaly = anomaly_score >= THRESHOLD

    # MITRE ATT&CK TTP mapping
    ttps = [
        {"ttp": r["ttp"], "name": r["name"], "tactic": r["tactic"]}
        for r in MITRE_RULES
        if r["condition"](event)
    ]

    return PredictionResponse(
        is_anomaly=is_anomaly,
        anomaly_score=round(anomaly_score, 4),
        threshold=THRESHOLD,
        ttp_detections=ttps,
        message="THREAT DETECTED" if is_anomaly else "Normal activity",
    )


@app.post("/predict/batch")
def predict_batch(events: list[LogEvent]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return [predict(e) for e in events]
