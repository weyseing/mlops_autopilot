# mlops_autopilot

End-to-end MLOps pipeline with automated drift detection and model retraining.

## Project Structure

```
mlops_autopilot/
├── data/               # Raw and processed datasets
├── notebooks/          # Exploratory analysis
├── src/
│   ├── train.py        # Model training + MLflow logging
│   ├── serve.py        # FastAPI inference endpoint
│   ├── monitor.py      # Drift detection (Evidently)
│   └── retrain.py      # Auto-retraining trigger
├── pipelines/          # SageMaker Pipeline definitions
├── .github/workflows/  # CI/CD via GitHub Actions
├── Dockerfile
└── requirements.txt
```

## Tech Stack

- **ML tracking**: MLflow
- **Serving**: FastAPI + Docker + AWS ECR/ECS
- **Drift detection**: Evidently AI
- **Retraining pipeline**: SageMaker Pipelines
- **CI/CD**: GitHub Actions

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py

# Run inference server locally
uvicorn src.serve:app --reload

# Run drift check
python src/monitor.py

# Trigger retraining manually
python src/retrain.py
```

## How It Works

1. Model is trained and registered in MLflow Model Registry
2. FastAPI endpoint serves predictions and logs incoming data
3. Drift monitor runs on a schedule comparing live data vs training distribution
4. If drift score exceeds threshold, retraining is triggered automatically
5. New model is evaluated — promoted to production only if it beats current model

## Environment Variables

```
MLFLOW_TRACKING_URI=
AWS_REGION=
ECR_REPOSITORY=
DRIFT_THRESHOLD=0.15
```
