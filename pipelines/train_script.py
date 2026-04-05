"""SageMaker training script for California Housing RandomForest.

SageMaker runs this script inside a managed container.
- Input data: /opt/ml/input/data/train/train.csv
- Output model: /opt/ml/model/model.joblib
- Hyperparameters passed via CLI args
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # SageMaker puts input data here
    input_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    # Load data — find whatever CSV is in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}: {os.listdir(input_dir)}")
    df = pd.read_csv(os.path.join(input_dir, csv_files[0]))
    print(f"Loaded {csv_files[0]} ({len(df)} rows)")
    target = "MedHouseVal"
    feature_names = [c for c in df.columns if c != target]

    X = df[feature_names]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )

    # Train
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # Save model — SageMaker packages everything in model_dir as model.tar.gz
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

    # Save metrics alongside model for retrieval after training
    pd.DataFrame([{"rmse": rmse, "mae": mae, "r2": r2}]).to_csv(
        os.path.join(model_dir, "metrics.csv"), index=False
    )
    print("Model saved to", model_dir)


if __name__ == "__main__":
    main()
