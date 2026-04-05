"""Auto-retraining via SageMaker with MLflow promotion."""

import os
import tarfile
import tempfile

import boto3
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sagemaker.session import Session
from sagemaker.sklearn.estimator import SKLearn


# ---- Config from env ----
MODEL_NAME = os.getenv("MODEL_NAME", "mlops_autopilot_model")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
INSTANCE_TYPE = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.m5.large")
USE_SPOT = os.getenv("SAGEMAKER_USE_SPOT", "true").lower() == "true"
MAX_RUN = int(os.getenv("SAGEMAKER_MAX_RUN", "3600"))
MAX_WAIT = int(os.getenv("SAGEMAKER_MAX_WAIT", "7200"))
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

DATA_PATH = "data/raw/california_housing.csv"


def get_current_production_rmse() -> float | None:
    """Fetch RMSE of the current Production model from MLflow."""
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        return None
    run = client.get_run(versions[0].run_id)
    return float(run.data.metrics.get("rmse", float("inf")))


def upload_training_data(sagemaker_session: Session) -> str:
    """Upload training CSV to SageMaker default S3 bucket. Returns S3 URI."""
    bucket = sagemaker_session.default_bucket()
    prefix = "mlops-autopilot/data"
    s3_uri = sagemaker_session.upload_data(
        path=DATA_PATH,
        bucket=bucket,
        key_prefix=prefix,
    )
    print(f"Training data uploaded to {s3_uri}")
    return s3_uri


def launch_sagemaker_training(sagemaker_session: Session, train_s3_uri: str) -> str:
    """Launch SageMaker training job. Returns S3 URI of model artifact."""
    estimator = SKLearn(
        entry_point="train_script.py",
        source_dir="pipelines",
        framework_version="1.2-1",
        role=SAGEMAKER_ROLE_ARN,
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        use_spot_instances=USE_SPOT,
        max_run=MAX_RUN,
        max_wait=MAX_WAIT if USE_SPOT else None,
        hyperparameters={
            "n-estimators": 100,
            "max-depth": 10,
            "random-state": 42,
        },
    )

    estimator.fit({"train": train_s3_uri}, wait=True)

    model_artifact_uri = estimator.model_data
    print(f"Training complete. Model artifact: {model_artifact_uri}")
    return model_artifact_uri


def download_model(boto_session: boto3.Session, model_s3_uri: str):
    """Download model.tar.gz from S3, extract and return the model object."""
    parts = model_s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "model.tar.gz")
        boto_session.client("s3").download_file(bucket, key, tar_path)

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        model = joblib.load(os.path.join(tmpdir, "model.joblib"))

    return model


def evaluate_locally(model) -> dict:
    """Evaluate the downloaded model on the same test split for fair comparison."""
    df = pd.read_csv(DATA_PATH)
    target = "MedHouseVal"
    feature_names = [c for c in df.columns if c != target]

    X = df[feature_names]
    y = df[target]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"Local evaluation — RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def log_and_maybe_promote(model, metrics: dict, current_rmse: float | None):
    """Log retrained model to MLflow. Promote to Production if it beats current."""
    mlflow.set_experiment("mlops_autopilot")

    with mlflow.start_run(run_name="sagemaker_retrain") as run:
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "retrain_source": "sagemaker",
            "instance_type": INSTANCE_TYPE,
            "use_spot": USE_SPOT,
        })
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("r2", metrics["r2"])

        input_example = pd.read_csv(DATA_PATH).drop(columns=["MedHouseVal"]).iloc[:1]
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=input_example,
        )
        print(f"Logged to MLflow run {run.info.run_id}")

    # Decide whether to promote
    new_rmse = metrics["rmse"]

    if current_rmse is None:
        print("No existing Production model — promoting new model")
        promote = True
    elif new_rmse < current_rmse:
        print(f"New RMSE {new_rmse:.4f} < current {current_rmse:.4f} — promoting")
        promote = True
    else:
        print(f"New RMSE {new_rmse:.4f} >= current {current_rmse:.4f} — keeping existing model")
        promote = False

    if promote:
        client = mlflow.MlflowClient()
        latest = client.get_latest_versions(MODEL_NAME, stages=["None"])
        if latest:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=latest[0].version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"Model version {latest[0].version} promoted to Production")


def main():
    if not SAGEMAKER_ROLE_ARN:
        print("ERROR: SAGEMAKER_ROLE_ARN not set")
        return

    mlflow.set_tracking_uri(TRACKING_URI)

    # Use mlops_autopilot AWS profile (mounted from ~/.aws)
    profile = os.getenv("AWS_PROFILE", "mlops_autopilot")
    boto_session = boto3.Session(profile_name=profile, region_name=AWS_REGION)
    sagemaker_session = Session(boto_session=boto_session)

    print("=== Step 1: Get current Production model RMSE ===")
    current_rmse = get_current_production_rmse()
    if current_rmse:
        print(f"Current Production RMSE: {current_rmse:.4f}")
    else:
        print("No Production model found")

    print("\n=== Step 2: Upload training data to S3 ===")
    train_s3_uri = upload_training_data(sagemaker_session)

    print("\n=== Step 3: Launch SageMaker training job ===")
    model_s3_uri = launch_sagemaker_training(sagemaker_session, train_s3_uri)

    print("\n=== Step 4: Download and evaluate model ===")
    model = download_model(boto_session, model_s3_uri)
    metrics = evaluate_locally(model)

    print("\n=== Step 5: Log to MLflow and decide promotion ===")
    log_and_maybe_promote(model, metrics, current_rmse)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
