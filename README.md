# Setup Guide
- Create **AWS Service Acc user**
    - **Permission :** `AmazonSageMakerFullAccess`
    - Add to `~/.aws/credentials`
```bash
[mlops_autopilot]
region = ap-southeast-1
aws_access_key_id = AKI**************
aws_secret_access_key = O6li*************
```

- Create **AWS Role** for ENV `SAGEMAKER_ROLE_ARN`
    - **Trusted entity type :** `AWS Service` > `Sagemaker`
    - **Permission :** `AmazonSageMakerFullAccess`

- Copy `.env` file for values below
```bash
SAGEMAKER_ROLE_ARN=     # SageMaker Role
```

# Quick Start

```bash
docker compose up -d                      # Start stack
docker compose run --rm app train         # Train model
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [3.5, 25, 5, 1, 1500, 3, 37, -122]}'
docker compose run --rm app monitor       # Check drift
docker compose run --rm retrain           # Retrain via SageMaker
```

