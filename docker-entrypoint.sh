#!/usr/bin/env bash
set -euo pipefail

wait_for_service() {
    local url="$1"
    local timeout="${2:-30}"
    local elapsed=0

    echo "Waiting for ${url} ..."
    until curl -sf "${url}" > /dev/null 2>&1; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ "${elapsed}" -ge "${timeout}" ]; then
            echo "ERROR: ${url} not reachable after ${timeout}s"
            exit 1
        fi
    done
    echo "${url} is ready."
}

case "${1:-serve}" in
    serve)
        wait_for_service "${MLFLOW_TRACKING_URI:-http://mlflow:5000}/health"
        exec uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
        ;;
    train)
        wait_for_service "${MLFLOW_TRACKING_URI:-http://mlflow:5000}/health"
        exec python -m src.train
        ;;
    monitor)
        exec python -m src.monitor
        ;;
    retrain)
        wait_for_service "${MLFLOW_TRACKING_URI:-http://mlflow:5000}/health"
        # Use real AWS credentials from ~/.aws for SageMaker
        export AWS_PROFILE="${AWS_PROFILE:-mlops_autopilot}"
        exec python -m src.retrain
        ;;
    mlflow)
        exec mlflow server \
            --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
            --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" \
            --host 0.0.0.0 \
            --port 5000
        ;;
    *)
        exec "$@"
        ;;
esac
