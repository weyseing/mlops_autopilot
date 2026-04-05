"""Drift detection using Evidently AI."""

import glob
import os
import sys

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
REFERENCE_PATH = "data/processed/reference.csv"
LIVE_DIR = "data/live"


def load_live_data() -> pd.DataFrame:
    """Load and concatenate all live inference logs."""
    files = sorted(glob.glob(os.path.join(LIVE_DIR, "*.csv")))
    if not files:
        print("No live data found in", LIVE_DIR)
        sys.exit(0)
    frames = [pd.read_csv(f) for f in files]
    live = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(live)} live samples from {len(files)} file(s)")
    return live


def main():
    # Load datasets
    reference = pd.read_csv(REFERENCE_PATH)
    print(f"Reference: {len(reference)} samples")
    live = load_live_data()

    # Run drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=live)
    result = report.as_dict()

    # metrics[0] = DatasetDriftMetric (summary), metrics[1] = DataDriftTable (per-column)
    summary = result["metrics"][0]["result"]
    drift_share = summary["share_of_drifted_columns"]
    n_drifted = summary["number_of_drifted_columns"]
    n_total = summary["number_of_columns"]

    print(f"\nDrift share: {drift_share:.2f} ({n_drifted}/{n_total} features)")

    # Show per-feature breakdown from DataDriftTable
    detail = result["metrics"][1]["result"]
    for col, col_result in detail["drift_by_columns"].items():
        status = "DRIFTED" if col_result["drift_detected"] else "ok"
        print(f"  {col:15s}  {status}  (score={col_result['drift_score']:.4f})")

    # Threshold decision
    if drift_share > DRIFT_THRESHOLD:
        print(f"\nDrift DETECTED (share {drift_share:.2f} > threshold {DRIFT_THRESHOLD})")
        sys.exit(1)
    else:
        print(f"\nNo significant drift (share {drift_share:.2f} <= threshold {DRIFT_THRESHOLD})")
        sys.exit(0)


if __name__ == "__main__":
    main()
