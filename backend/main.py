import logging
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ConfigDict
import joblib
import xgboost as xgb

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("predictive_maintenance_backend")

# ---------------------------------------------------------------------------
# Cross-Platform XGBoost Deserialization Patch
# Fixes libstdc++ (Linux) to MSVC (Windows) std::mt19937 RNG state mismatch
# ---------------------------------------------------------------------------
def apply_xgboost_cross_platform_patch():
    """
    Patches xgboost.core.Booster.__setstate__ to handle cross-platform
    differences in std::mt19937 (RNG state) between Linux (decimal stream)
    and Windows MSVC (hexadecimal stream).
    """
    try:
        dummy = xgb.XGBClassifier(n_estimators=1)
        dummy.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
        win_buf = dummy.get_booster().__getstate__()["handle"]
        idx_rng_win = win_buf.find(b"rng_state")
        idx_seed_win = win_buf.find(b"L\x00\x00\x00\x00\x00\x00\x00\x04seed")
        if idx_rng_win == -1 or idx_seed_win == -1:
            logger.warning("Could not isolate platform RNG slice for patch.")
            return
        platform_rng_slice = win_buf[idx_rng_win:idx_seed_win]

        orig_setstate = xgb.core.Booster.__setstate__

        def patched_setstate(self, state):
            handle = state.get("handle")
            if handle is not None:
                try:
                    orig_setstate(self, state)
                    return
                except Exception:
                    buf = bytearray(handle)
                    idx_rng = buf.find(b"rng_state")
                    idx_seed = buf.find(b"L\x00\x00\x00\x00\x00\x00\x00\x04seed")
                    if idx_rng != -1 and idx_seed != -1:
                        state["handle"] = buf[:idx_rng] + platform_rng_slice + buf[idx_seed:]
                        orig_setstate(self, state)
                        return
                    raise
            orig_setstate(self, state)

        xgb.core.Booster.__setstate__ = patched_setstate
        logger.info("XGBoost cross-platform compatibility patch applied successfully.")
    except Exception as e:
        logger.warning(f"Could not apply XGBoost cross-platform patch: {e}")

apply_xgboost_cross_platform_patch()

# ---------------------------------------------------------------------------
# Path Resolution for Models & Metadata
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

CANDIDATE_MODEL_DIRS = [
    ROOT_DIR / "models",
    BASE_DIR / "models",
    Path("../models").resolve(),
    Path("models").resolve(),
]

MODELS_DIR = None
for candidate in CANDIDATE_MODEL_DIRS:
    if (candidate / "schema.json").is_file():
        MODELS_DIR = candidate
        break

if not MODELS_DIR:
    # Default to ROOT_DIR / "models" even if not found immediately
    MODELS_DIR = ROOT_DIR / "models"

logger.info(f"Resolved models directory: {MODELS_DIR}")

# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------
schema_data: Dict[str, Any] = {}
metrics_data: Dict[str, Any] = {}
feature_columns: List[str] = []
failure_threshold: float = 0.5
failure_model = None
rul_model = None

def load_assets():
    global schema_data, metrics_data, feature_columns, failure_threshold, failure_model, rul_model

    # 1. Load schema.json
    schema_path = MODELS_DIR / "schema.json"
    if schema_path.is_file():
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
            feature_columns = schema_data.get("features", [])
            logger.info(f"Loaded schema with {len(feature_columns)} features from {schema_path}")
        except Exception as e:
            logger.error(f"Error loading schema.json: {e}")
            raise RuntimeError(f"Failed to load schema.json: {e}")
    else:
        logger.error(f"schema.json not found at {schema_path}")
        raise FileNotFoundError(f"schema.json not found at {schema_path}")

    # 2. Load metrics.json
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.is_file():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            raw_threshold = metrics_data.get("threshold", 0.5)
            # Normalize threshold to 0.0 - 1.0 range
            failure_threshold = float(raw_threshold) if raw_threshold <= 1.0 else float(raw_threshold) / 100.0
            logger.info(f"Loaded metrics from {metrics_path}. Decision threshold: {failure_threshold}")
        except Exception as e:
            logger.warning(f"Error loading metrics.json, defaulting threshold to 0.5: {e}")
            failure_threshold = 0.5
            metrics_data = {"threshold": 0.5}
    else:
        logger.warning(f"metrics.json not found at {metrics_path}, defaulting threshold to 0.5")
        failure_threshold = 0.5
        metrics_data = {"threshold": 0.5}

    # 3. Load Failure Model
    failure_model_path = MODELS_DIR / "predictive_maintenance_failure_model.joblib"
    if failure_model_path.is_file():
        try:
            failure_model = joblib.load(failure_model_path)
            logger.info(f"Successfully loaded failure model from {failure_model_path}")
        except Exception as e:
            logger.error(f"Failed to load failure model: {e}")
            failure_model = None
    else:
        logger.error(f"Failure model not found at {failure_model_path}")
        failure_model = None

    # 4. Load RUL Model (Optional - skip gracefully if missing or fails)
    rul_model_path = MODELS_DIR / "predictive_maintenance_rul_model.joblib"
    if rul_model_path.is_file():
        try:
            rul_model = joblib.load(rul_model_path)
            logger.info(f"Successfully loaded RUL model from {rul_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load RUL model, skipping RUL predictions gracefully: {e}")
            rul_model = None
    else:
        logger.info(f"RUL model not found at {rul_model_path}, skipping RUL predictions.")
        rul_model = None

# ---------------------------------------------------------------------------
# FastAPI Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models and schema on startup
    load_assets()
    yield

# ---------------------------------------------------------------------------
# FastAPI App Initialization & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    description="FastAPI backend for predictive maintenance failure classification and remaining useful life (RUL) estimation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "body": exc.body,
        },
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"},
    )

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class MachineInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    Machine_Type: Optional[str] = Field(None, description="Machine category/type")
    Installation_Year: Optional[float] = Field(None, description="Year installed (e.g. 2020)")
    Operational_Hours: Optional[float] = Field(None, description="Total operational hours")
    Temperature_C: Optional[float] = Field(None, description="Operating temperature in Celsius")
    Vibration_mms: Optional[float] = Field(None, description="Vibration in mm/s")
    Sound_dB: Optional[float] = Field(None, description="Noise level in dB")
    Oil_Level_pct: Optional[float] = Field(None, description="Oil level percentage (0-100)")
    Coolant_Level_pct: Optional[float] = Field(None, description="Coolant level percentage (0-100)")
    Power_Consumption_kW: Optional[float] = Field(None, description="Power consumption in kW")
    Last_Maintenance_Days_Ago: Optional[float] = Field(None, description="Days since last maintenance")
    Maintenance_History_Count: Optional[float] = Field(None, description="Total count of prior maintenance events")
    Failure_History_Count: Optional[float] = Field(None, description="Total count of prior failures")
    AI_Supervision: Optional[float] = Field(None, description="Whether AI supervision is enabled (0 or 1)")
    Error_Codes_Last_30_Days: Optional[float] = Field(None, description="Number of error codes logged in last 30 days")
    Machine_Age: Optional[float] = Field(None, description="Machine age in years (computed server-side if not supplied)")

class PredictionResponse(BaseModel):
    failure_probability: float = Field(..., description="Predicted failure probability (0 to 100)")
    failure_prediction: int = Field(..., description="Binary failure flag (0 or 1) based on metrics threshold")
    risk_category: str = Field(..., description="Display risk category: LOW (<30), MEDIUM (30-60), HIGH (60-80), CRITICAL (>=80)")
    remaining_useful_life_days: Optional[float] = Field(None, description="Estimated remaining useful life in days, or null")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def compute_risk_category(failure_probability: float) -> str:
    """
    Display bands:
      < 30: LOW
      30 - 60: MEDIUM
      60 - 80: HIGH
      >= 80: CRITICAL
    """
    if failure_probability < 30.0:
        return "LOW"
    elif failure_probability < 60.0:
        return "MEDIUM"
    elif failure_probability < 80.0:
        return "HIGH"
    else:
        return "CRITICAL"

def prepare_dataframe(machines: List[MachineInput]) -> pd.DataFrame:
    """
    Converts a list of MachineInput objects to a pandas DataFrame containing
    exactly the schema.json feature columns in order.
    Computes Machine_Age = 2026 - Installation_Year if not supplied.
    """
    rows = []
    for machine in machines:
        item = machine.model_dump()

        # Exclude Machine_ID, Remaining_Useful_Life_days, Failure_Within_7_Days
        for excluded in ("Machine_ID", "Remaining_Useful_Life_days", "Failure_Within_7_Days"):
            item.pop(excluded, None)

        machine_age = item.get("Machine_Age")
        installation_year = item.get("Installation_Year")

        # Compute Machine_Age = 2026 - Installation_Year if not supplied
        if machine_age is None or (isinstance(machine_age, float) and np.isnan(machine_age)):
            if installation_year is not None and not (isinstance(installation_year, float) and np.isnan(installation_year)):
                item["Machine_Age"] = float(2026 - installation_year)
            else:
                item["Machine_Age"] = np.nan

        # If installation year is missing but machine age is present
        if installation_year is None or (isinstance(installation_year, float) and np.isnan(installation_year)):
            if item.get("Machine_Age") is not None and not (isinstance(item["Machine_Age"], float) and np.isnan(item["Machine_Age"])):
                item["Installation_Year"] = float(2026 - item["Machine_Age"])
            else:
                item["Installation_Year"] = np.nan

        # Select only expected schema features in order
        row = {col: item.get(col, np.nan) for col in feature_columns}
        rows.append(row)

    return pd.DataFrame(rows, columns=feature_columns)

def run_predictions(df: pd.DataFrame) -> List[PredictionResponse]:
    """
    Runs inference on the DataFrame and returns a list of PredictionResponse.
    """
    if failure_model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failure prediction model is not loaded."
        )

    try:
        proba_matrix = failure_model.predict_proba(df)
        failure_probs = proba_matrix[:, 1]
    except Exception as e:
        logger.error(f"Inference error with failure model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during failure prediction: {str(e)}"
        )

    rul_preds = None
    if rul_model is not None:
        try:
            rul_preds = rul_model.predict(df)
        except Exception as e:
            logger.warning(f"RUL model prediction failed, setting RUL to null: {e}")
            rul_preds = None

    results: List[PredictionResponse] = []
    th = failure_threshold if failure_threshold <= 1.0 else failure_threshold / 100.0

    for i in range(len(df)):
        raw_prob = float(failure_probs[i])
        prob_pct = round(raw_prob * 100.0, 2)
        prediction_flag = 1 if raw_prob >= th else 0
        risk_cat = compute_risk_category(prob_pct)

        rul_val = None
        if rul_preds is not None:
            raw_rul = float(rul_preds[i])
            rul_val = round(max(0.0, raw_rul), 1)

        results.append(
            PredictionResponse(
                failure_probability=prob_pct,
                failure_prediction=prediction_flag,
                risk_category=risk_cat,
                remaining_useful_life_days=rul_val,
            )
        )

    return results

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Health Check")
def health_check():
    """
    Returns server status and whether models and schemas are loaded.
    """
    return {
        "status": "ok",
        "failure_model_loaded": failure_model is not None,
        "rul_model_loaded": rul_model is not None,
        "schema_loaded": bool(schema_data),
        "metrics_loaded": bool(metrics_data),
        "decision_threshold": failure_threshold,
    }

@app.get("/schema", summary="Get Schema")
def get_schema():
    """
    Returns schema.json contents including features, machine types, and numeric ranges.
    """
    if not schema_data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Schema metadata is not loaded."
        )
    return schema_data

@app.get("/metrics", summary="Get Model Metrics")
def get_metrics():
    """
    Returns metrics.json contents including evaluation metrics and decision threshold.
    """
    if not metrics_data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model metrics metadata is not loaded."
        )
    return metrics_data

@app.post("/predict", response_model=PredictionResponse, summary="Predict Machine Failure")
def predict_single(machine: MachineInput):
    """
    Accepts telemetry and metadata for a single machine and returns:
    - failure_probability (0-100)
    - failure_prediction (0 or 1 based on threshold from metrics.json)
    - risk_category (LOW, MEDIUM, HIGH, CRITICAL)
    - remaining_useful_life_days (or null)
    """
    df = prepare_dataframe([machine])
    results = run_predictions(df)
    return results[0]

@app.post("/predict-batch", response_model=List[PredictionResponse], summary="Batch Predict Machine Failure")
def predict_batch(machines: List[MachineInput]):
    """
    Accepts a list of machine inputs and returns a list of predictions.
    """
    if not machines:
        return []

    df = prepare_dataframe(machines)
    return run_predictions(df)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
