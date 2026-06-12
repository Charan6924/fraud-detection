# 🏦 Real-Time Credit Card Fraud Detection System

> An end-to-end ML project covering feature engineering, model training, serverless TypeScript inference deployment, and a full CI/CD pipeline with automated retraining.

---

## Project Overview

Build a production-grade **fraud detection system** that classifies credit card transactions as fraudulent or legitimate in real time (< 100ms). The system ingests live transaction events through serverless TypeScript functions, scores them against a trained ML model, logs predictions to a managed database, and automatically retrains when data drift is detected — all wired together with a CI/CD pipeline on GitHub Actions.

The architecture is deliberately split: **Python handles everything ML** (training, evaluation, model artifacts), and **TypeScript handles everything application** (API, event routing, orchestration, monitoring). This reflects how real production teams are structured.

This project touches every layer of the ML lifecycle:

- **Data engineering** — streaming ingestion, feature stores, offline/online split
- **Model training** — imbalanced classification, experiment tracking, promotion gates
- **Inference deployment** — serverless TypeScript API calling a hosted model endpoint
- **MLOps / CI/CD** — automated retraining, model promotion, staged rollout
- **Monitoring** — drift detection, alerting, dashboards

---

## System Architecture

### Production Flow

Transactions arrive as events from a client or upstream payment processor. An **API Gateway** routes each request to a TypeScript Lambda function, which fetches pre-computed online features from a Redis-backed feature store, invokes a containerized model Lambda, and writes the prediction to a database.

### CI/CD Flow

Every push to `main` triggers GitHub Actions. The pipeline validates data quality, runs the Python training job on a compute instance, evaluates the new model against promotion gates, and — if all gates pass — deploys the updated Lambda functions and registers the new model version. The deployment uses a weighted Lambda alias to canary 10% of traffic before full rollout.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language (serving) | TypeScript, Node.js 20 |
| Serverless framework | AWS SST v3 (or Serverless Framework) |
| API layer | AWS API Gateway + Lambda |
| Event ingestion | AWS EventBridge, SQS |
| Feature store | Upstash Redis (online), S3 + Parquet (offline) |
| Model hosting | Custom Docker container on Lambda (container image) |
| Training language | Python 3.11 |
| Training framework | XGBoost, scikit-learn, imbalanced-learn |
| Experiment tracking | MLflow (self-hosted on EC2 or managed) |
| Model registry | MLflow Model Registry |
| Database | DynamoDB (predictions log) |
| CI/CD | GitHub Actions |
| Monitoring | AWS CloudWatch, Evidently AI, custom Lambda metrics |
| IaC | AWS CDK (via SST) |

---

## Dataset

Use the **[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** as the seed dataset. It contains 284,807 transactions with 492 frauds (0.17% positive class) — a classic imbalanced classification problem that forces deliberate choices around sampling, thresholds, and evaluation metrics.

Columns include anonymized PCA features `V1–V28`, a raw `Amount`, a `Time` offset, and a binary `Class` label. In the project you simulate a streaming pipeline by replaying this dataset through EventBridge as if transactions were arriving live.

---

## Project Structure

```
fraud-detection/
├── container/                   # Custom model inference container
│   ├── Dockerfile               # Multi-stage: Python + XGBoost + model artifact
│   ├── app.py                   # FastAPI inference server with Lambda RIE
│   └── requirements.txt         # xgboost, scikit-learn, fastapi, etc.
│
├── packages/
│   ├── functions/               # TypeScript Lambda functions
│   │   ├── predict/             # Inference handler (API Gateway → container Lambda)
│   │   ├── ingest/              # Event ingestion (EventBridge consumer)
│   │   ├── monitor/             # Drift check scheduler (EventBridge cron)
│   │   └── retrain-trigger/     # Fires retraining workflow via GitHub API
│   │
│   └── core/                    # Shared TypeScript types, feature client, DB client
│
├── ml/                          # Python ML workspace
│   ├── features/                # Feature engineering & Feast definitions
│   ├── training/                # Train, evaluate, promote scripts
│   └── monitoring/              # Drift detection with Evidently
│
├── infra/                       # SST / CDK infrastructure definitions
│   ├── api.ts                   # API Gateway + Lambda config
│   ├── queues.ts                # SQS + EventBridge
│   ├── storage.ts               # DynamoDB tables, S3 buckets
│   └── container.ts             # ECR repo + container Lambda definition
│
├── .github/
│   └── workflows/
│       ├── ci.yaml              # Lint, type-check, unit tests
│       ├── train.yaml           # Python training pipeline
│       └── deploy.yaml          # Container build + Lambda deploy + canary rollout
│
├── tests/
│   ├── unit/                    # Jest unit tests for Lambda handlers
│   └── integration/             # End-to-end API tests (local Docker Compose)
│
├── sst.config.ts                # SST entry point
├── docker-compose.yml           # Local dev: Redis + model container + DynamoDB Local
└── package.json
```

---

## Phase 1 — Data & Feature Engineering (Python)

### Goals

Build two versions of every feature: an **offline** version stored as Parquet on S3 for training, and an **online** version served from Redis for sub-millisecond lookup at inference time. Keeping these in sync is critical — training-serving skew (where the model was trained on different data than it sees in production) is one of the most common silent failure modes.

### Features to Engineer

On top of the raw PCA columns, the pipeline builds rolling aggregates per card over a 1-hour window: transaction count, average spend, maximum spend, a Z-score for the current amount relative to card history, time elapsed since the last transaction, and a pre-computed merchant category risk score. These behavioral features carry far more signal than the raw PCA columns alone.

### Data Validation

Before any training run, a Great Expectations checkpoint validates the dataset. It checks for nulls in the PCA features, ensures `Amount` is non-negative, confirms `Class` is binary, and checks that feature distributions fall within expected ranges. If any check fails the training pipeline is aborted — data quality is a hard gate, not a warning.

---

## Phase 2 — Model Training (Python)

### Approach

The core challenge is severe class imbalance: only 0.17% of transactions are fraudulent. The training pipeline addresses this with SMOTE oversampling of the minority class combined with a class weight penalty in XGBoost. The model is evaluated on **Average Precision (PR-AUC)** rather than ROC-AUC, since PR-AUC is far more informative when positives are rare.

All training runs are tracked in MLflow — hyperparameters, metrics, the trained artifact, and the dataset version used. This creates a full audit trail so you can always reproduce any model version.

### Promotion Gates

A new model is registered in the MLflow Model Registry and marked as a candidate for deployment only if every gate passes. The gates are: Average Precision ≥ 0.85, F1 Score on the fraud class ≥ 0.78, False Negative Rate ≤ 5%, p99 inference latency ≤ 80ms, and a bias check confirming the performance gap across demographic proxies is ≤ 3%. If any gate fails the run is logged but the current production model is left untouched.

---

## Phase 3 — Model Hosting (Custom Docker Container on Lambda)

### Why Build Your Own Container

Rather than using a managed inference service, the trained model is packaged into a **custom Docker container** and deployed as a Lambda function using container image support. This forces you to own the entire serving stack: model loading on startup, request serialization, the HTTP server interface, and the Lambda Runtime Interface Client (RIC). No black boxes.

The container uses a multi-stage build: the first stage installs Python dependencies (XGBoost, scikit-learn, FastAPI) and copies the serialized model artifact from S3, the second stage strips build tooling to minimize image size. The resulting image is pushed to Amazon ECR and registered as a Lambda function.

### Container Design

```
container/
├── Dockerfile
├── app.py
└── requirements.txt
```

The `app.py` runs a FastAPI server inside the Lambda Runtime Interface. On startup it loads the XGBoost model into memory (warm start). Each invocation receives a JSON payload of pre-computed features, runs `model.predict()`, and returns the prediction score and confidence. The Lambda RIC adapter handles the translation between API Gateway's `Invoke` payload and the HTTP server.

### Cold Start Mitigation

Container Lambda cold starts are the main challenge — the image must be pulled and extracted before the first request. Mitigations include:

- **Provisioned Concurrency** — keep 1 container warm for steady-state traffic
- **Multi-stage Docker build** — minimize image size by excluding training code and dev dependencies
- **Model pre-loading** — the model artifact is copied into the image at build time, not downloaded at runtime (avoids S3 latency on cold start)

### Canary Rollouts with Lambda Aliases

Lambda weighted aliases handle gradual rollouts. A new container version is published as `$LATEST`, then the production alias shifts traffic: 10% → 50% → 100% over a monitoring window. If the error rate or latency exceeds thresholds, the alias snaps back to the previous version. No SageMaker required.

---

## Phase 4 — Inference API (TypeScript + Lambda)

### Predict Lambda

The core of the serving layer is a TypeScript Lambda function sitting behind API Gateway. When a transaction request arrives, the handler pulls the card's online features from Upstash Redis, merges them with the payload, invokes the container Lambda via the AWS SDK `Invoke` command, and writes the prediction result to DynamoDB. The full round-trip — feature fetch, model call, DB write — is designed to complete in under 100ms at p99.

The handler is strongly typed end-to-end: the request schema, the feature shape, the container Lambda response, and the DynamoDB record are all defined as TypeScript types in the `core` package, shared across all Lambda functions.

### Ingest Lambda

A separate Lambda function subscribes to the EventBridge event bus and handles the async path: consuming raw transaction events, computing and writing online features to Redis, and forwarding events to SQS for downstream processing. This decouples feature computation from the synchronous prediction path.

### Monitor Lambda

A scheduled Lambda runs every 6 hours on an EventBridge cron. It reads the last N predictions from DynamoDB, computes a drift report against the training reference distribution using the Python Evidently library (invoked as a subprocess or called via a lightweight REST wrapper), and publishes drift metrics to CloudWatch. If the drift ratio exceeds 30% it calls the GitHub Actions API to dispatch the retraining workflow.

---

## Phase 5 — CI/CD Pipeline (GitHub Actions)

### CI Pipeline

Triggered on every push and pull request. Runs ESLint and TypeScript type-checking across all packages, Jest unit tests for every Lambda handler, Python linting and mypy type-checking across the ML workspace, and the Great Expectations data checkpoint. The pipeline must be fully green before any merge to `main`.

### Training Pipeline

Triggered on merges to `main` that touch the `ml/` directory, on a weekly schedule (Sunday 2am), or manually via `workflow_dispatch` (which the Monitor Lambda calls when drift is detected). The pipeline spins up a compute instance, runs the full training job, evaluates against the promotion gates, and — if promoted — packages the model artifact and uploads it to S3. It emits a repository dispatch event that triggers the deploy pipeline.

### Deploy Pipeline

Triggered by a successful training run. Builds the Docker image using the multi-stage Dockerfile, pushes it to ECR, runs `sst deploy` to update all Lambda functions (including the container Lambda with the new image URI), shifts 10% of traffic to the new Lambda alias, monitors CloudWatch metrics for 30 minutes, and rolls to 100% if no alarms fire. On failure at any step the pipeline snaps the alias back to the previous container version and alerts via a Slack Lambda.

---

## Phase 6 — Monitoring

### What to Monitor

**Model health** is tracked by the Monitor Lambda using Evidently: it watches input feature distributions for drift using the Population Stability Index, monitors prediction distribution shift (is the model suddenly calling everything fraud or nothing fraud?), and tracks data quality metrics like null rates.

**Infrastructure health** is tracked in CloudWatch: Lambda invocation errors, p50/p95/p99 latency for the Predict Lambda, container Lambda cold start frequency and duration, SQS queue depth for the ingest path, and DynamoDB read/write capacity.

**Business metrics** are tracked with a delay: chargebacks are joined back to predictions weekly to compute the real false negative rate, closing the ground-truth feedback loop.

### Alerting

CloudWatch alarms notify a Slack channel (via a simple webhook Lambda) when: prediction Lambda error rate exceeds 1%, p99 latency exceeds 150ms, the container Lambda error rate exceeds 1%, or the drift ratio exceeds the retraining threshold. The monitoring Lambda also writes a weekly summary report to S3 as a JSON artifact for auditing.

---

## Local Development

SST's local development mode (`sst dev`) proxies Lambda invocations to your local machine, so you can iterate on TypeScript handlers with live reload without deploying to AWS. The Python training stack runs entirely locally via a virtual environment and a local MLflow server. DynamoDB Local and a local Redis instance (via Docker Compose) back the feature store and predictions log during development.

---

## Learning Outcomes

By completing this project you will have hands-on experience with:

- **Imbalanced classification** — SMOTE, class weights, threshold tuning, PR-AUC vs ROC-AUC
- **Feature stores** — online vs offline serving, preventing training-serving skew
- **Custom model container** — Docker multi-stage builds, Lambda container image support, cold start profiling
- **TypeScript on AWS** — Lambda handlers, SST/CDK infrastructure-as-code, strong typing end-to-end
- **Event-driven architecture** — EventBridge, SQS, DynamoDB Streams
- **Experiment tracking** — MLflow runs, parameters, metrics, artifact storage
- **CI/CD for ML** — GitHub Actions, automated gates, canary deployments, weighted traffic splitting
- **Drift monitoring** — Evidently, CloudWatch, automated retraining triggers

---

## Extensions

| Extension | What you learn |
|---|---|
| Add a **human review queue** for borderline predictions | Step Functions, active learning |
| Add **explainability** (SHAP values) returned in the API response | Model interpretability at inference time |
| Build an **A/B test** between two model versions | Statistical significance, Lambda weighted aliases |
| Add a **streaming feature pipeline** using Kinesis | Real-time feature computation at scale |
| Add a **graph neural network** to detect fraud rings | Graph ML, PyTorch Geometric |
| **GPU inference** via ECS Fargate with NVIDIA | GPU-accelerated model serving, EFS for model storage |

