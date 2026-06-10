# Credit Card Fraud Detection

End-to-end ML system that classifies transactions as fraudulent or legitimate in real time. Python handles training and model artifacts; TypeScript handles inference serving and infrastructure.

## Architecture

- **Containerized model** — XGBoost trained on the Kaggle credit card fraud dataset, packaged in a custom Docker container and deployed on Lambda
- **Serverless inference** — TypeScript Lambda fetches online features from Redis, invokes the container model Lambda, and logs predictions to DynamoDB
- **Feature store** — rolling window aggregates per card (tx count, avg spend, z-score) served from Upstash Redis
- **CI/CD** — GitHub Actions pipelines for training, evaluation with promotion gates, container build/push, and canary rollout via Lambda weighted aliases
- **Monitoring** — Evidently drift detection on a scheduled Lambda, auto-triggers retraining when drift exceeds threshold

## Stack

| Layer | Tool |
|---|---|
| Model | XGBoost, scikit-learn |
| Serving | Lambda (custom Docker container) |
| API | API Gateway + TypeScript Lambda |
| Features | Upstash Redis (online), S3 + Parquet (offline) |
| CI/CD | GitHub Actions |
| IaC | AWS SST v3 / CDK |
| Monitoring | Evidently AI, CloudWatch |

## Local Dev

```bash
sst dev              # live-reload Lambda handlers
docker compose up    # Redis + model container + DynamoDB Local
```

## Phases

1. Feature engineering (rolling aggregates, offline/online split)
2. Model training (SMOTE, PR-AUC, MLflow tracking)
3. Custom container with FastAPI + Lambda RIC
4. TypeScript inference API with feature merge
5. GitHub Actions CI/CD with canary deployment
6. Drift monitoring and automated retraining
