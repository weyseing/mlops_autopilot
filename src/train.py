"""Model training with MLflow logging."""

import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    model_name = os.getenv("MODEL_NAME", "mlops_autopilot_model")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # ---- Load data ----
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    feature_names = housing.feature_names
    target = "MedHouseVal"

    X = df[feature_names]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Save data for later use ----
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/raw/california_housing.csv", index=False)
    X_train.to_csv("data/processed/reference.csv", index=False)

    # ---- Train ----
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1,
    }

    mlflow.set_experiment("mlops_autopilot")

    with mlflow.start_run(run_name="initial_training") as run:
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        input_example = X_test.iloc[:1]
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=input_example,
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # ---- Promote to Production ----
    client = mlflow.MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=["None"])
    if latest_versions:
        version = latest_versions[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Model version {version} promoted to Production")


if __name__ == "__main__":
    main()
