"""Simple test script for the inference API."""

import json
import requests

BASE_URL = "http://localhost:8000"

SAMPLE = {
    "MedInc": 3.5,
    "HouseAge": 25.0,
    "AveRooms": 5.0,
    "AveBedrms": 1.0,
    "Population": 1500.0,
    "AveOccup": 3.0,
    "Latitude": 37.0,
    "Longitude": -122.0,
}


def log_response(label, r):
    print(f"\n{'='*50}")
    print(f"TEST: {label}")
    print(f"{'='*50}")
    print(f"Status: {r.status_code}")
    print(f"Response: {json.dumps(r.json(), indent=2)}")


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    log_response("GET /health", r)
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    print("RESULT: PASS")


def test_predict():
    r = requests.post(f"{BASE_URL}/predict", json={"features": SAMPLE})
    log_response("POST /predict (single)", r)
    print(f"Input:  {SAMPLE}")
    print(f"Output: predicted house value = ${r.json()['prediction'][0] * 100_000:,.0f}")
    assert r.status_code == 200
    print("RESULT: PASS")


def test_predict_batch():
    sample_expensive = {**SAMPLE, "MedInc": 8.0, "Latitude": 34.0, "Longitude": -118.0}
    payload = {"instances": [SAMPLE, sample_expensive]}
    r = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    log_response("POST /predict/batch (2 instances)", r)
    preds = r.json()["predictions"]
    for i, p in enumerate(preds):
        print(f"  Instance {i+1}: ${p * 100_000:,.0f}")
    assert r.status_code == 200
    assert len(preds) == 2
    print("RESULT: PASS")


def test_predict_missing_features():
    r = requests.post(f"{BASE_URL}/predict", json={"features": {"MedInc": 3.5}})
    log_response("POST /predict (missing features)", r)
    assert r.status_code == 422
    print("RESULT: PASS (correctly rejected)")


def test_predict_invalid_body():
    r = requests.post(f"{BASE_URL}/predict", json={"wrong_key": 123})
    log_response("POST /predict (invalid body)", r)
    assert r.status_code == 422
    print("RESULT: PASS (correctly rejected)")


if __name__ == "__main__":
    print("MLOps Autopilot - Inference API Tests")
    print(f"Target: {BASE_URL}\n")

    test_health()
    test_predict()
    test_predict_batch()
    test_predict_missing_features()
    test_predict_invalid_body()

    print(f"\n{'='*50}")
    print("ALL TESTS PASSED")
    print(f"{'='*50}")
