"""FastAPI inference endpoint."""

import datetime
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MLOps Autopilot")

model = None
feature_names = None


class PredictRequest(BaseModel):
    features: dict[str, float]


class PredictBatchRequest(BaseModel):
    instances: list[dict[str, float]]


@app.on_event("startup")
def load_model():
    global model, feature_names

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name = os.getenv("MODEL_NAME", "mlops_autopilot_model")

    mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}/Production"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        feature_names = model.feature_names_in_.tolist()
        print(f"Loaded model: {model_uri}")
        print(f"Features: {feature_names}")
    except Exception as e:
        print(f"WARNING: Could not load model: {e}")
        print("Run 'train' first to register a model.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    df = pd.DataFrame([request.features])

    missing = set(feature_names) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}. Required: {feature_names}",
        )

    df = df[feature_names]
    prediction = model.predict(df)

    _log_live_data(df)

    return {"prediction": prediction.tolist()}


@app.post("/predict/batch")
def predict_batch(request: PredictBatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    df = pd.DataFrame(request.instances)

    missing = set(feature_names) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}. Required: {feature_names}",
        )

    df = df[feature_names]
    predictions = model.predict(df)

    _log_live_data(df)

    return {"predictions": predictions.tolist()}


def _log_live_data(df: pd.DataFrame):
    """Append incoming features to a daily CSV in data/live/."""
    live_dir = "data/live"
    os.makedirs(live_dir, exist_ok=True)

    today = datetime.date.today().isoformat()
    filepath = os.path.join(live_dir, f"{today}.csv")

    write_header = not os.path.exists(filepath)
    df.to_csv(filepath, mode="a", header=write_header, index=False)
