# mlops_autopilot

End-to-end MLOps pipeline: train, serve, detect drift, retrain on SageMaker.

## Structure

```bash
src/
├── train.py              # Train RandomForest + register in MLflow
├── serve.py              # FastAPI /predict endpoint + log live data
├── monitor.py            # Evidently drift detection (reference vs live)
└── retrain.py            # SageMaker spot training + MLflow promotion
pipelines/
└── train_script.py       # Standalone script SageMaker runs
```

## Commands (Docker)

```bash
docker compose up -d                      # Start stack (MLflow, MinIO, Postgres, API)
docker compose run --rm app train         # Train + register model
docker compose restart app                # Reload model in serving
docker compose run --rm app monitor       # Check drift (exit 1 = drifted)
docker compose run --rm retrain           # Retrain via SageMaker
```

## Flow

train → serve (logs live data) → monitor (Evidently) → retrain (SageMaker) → promote if RMSE improves

## Tech

- **ML tracking**: MLflow (Postgres + MinIO)
- **Serving**: FastAPI + Docker
- **Drift detection**: Evidently AI (Wasserstein / K-S auto-selected)
- **Retraining**: SageMaker SKLearn estimator (spot instances)
- **sklearn pinned to 1.2.1** to match SageMaker container
