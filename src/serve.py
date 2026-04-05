"""FastAPI inference endpoint."""

from fastapi import FastAPI

app = FastAPI(title="MLOps Autopilot")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict():
    return {"message": "Prediction not yet implemented"}
