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