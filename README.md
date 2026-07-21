# Credit Card Fraud Detection

End-to-end ML system that classifies transactions as fraudulent or legitimate in real time. Python handles training and model artifacts; TypeScript/Next.js handles inference serving and infrastructure.

Trained on the **IEEE-CIS Fraud Detection** dataset — a large-scale, real-world transaction dataset with extensive identity and transactional features. Uses a stacking ensemble of XGBoost, Random Forest, and Logistic Regression with a meta-model for final prediction.

## Architecture

```
Client / Payment Processor
        │
        ▼
  Next.js API (API Routes)
  ┌─────────────────────────────┐
  │  POST /api/predict          │
  │  GET  /api/transactions     │
  │  GET  /api/health           │
  └──────┬──────────────────────┘
         │ HTTP
         ▼
  ECS Fargate (Model Container)
  ┌─────────────────────────────┐
  │  FastAPI server             │
  │  Ensemble: XGBoost + RF + LR│
  │  + meta-model               │
  │  POST /drift (Evidently AI)  │
  │  POST /feedback              │
  └──────┬──────────────────────┘
         │ write / read
         ▼
  DynamoDB (Predictions Log)
  ──────────────────────
  ▲ Monitor Lambda (cron, 6h)
  │  POST /drift → if > 30% drifted
  │  → GitHub dispatch retrain
  │
  └──► Train Pipeline (Fargate RunTask)
       Download features → train ensemble
       → upload models to S3 → dispatch deploy
```

- **Ensemble model** — XGBoost (handles NaN natively) + Random Forest + Logistic Regression, stacked with a Logistic Regression meta-model. Imputation via `SimpleImputer` for RF/LR. Threshold tuned at 0.10 for fraud class recall.
- **Inference API** — Next.js 15 App Router API routes. Protected by API key auth (`x-api-key` header). Accepts transaction JSON, checks rate limit and dedup cache in Upstash Redis, forwards to the model container over HTTP, writes prediction + metadata to DynamoDB.
- **Container** — FastAPI server in a Docker container running on ECS Fargate. Protected by an internal shared secret (`x-model-secret` header) so only the Next.js API can call it. Loads all model artifacts at startup. Preprocessing (log transform, scaling, imputation) happens in Python in the container to prevent training-serving skew.
- **Feature store** — Upstash Redis provides per-card rate limiting (10 txns/60s window) and prediction dedup caching (5-minute TTL) to reduce model invocation costs.
- **Drift detection** — Evidently AI compares production features against a training reference dataset. Monitor Lambda runs every 6 hours, calls POST /drift, dispatches retrain if drift exceeds 30%.
- **Label feedback** — POST /feedback attaches ground-truth labels to predictions in DynamoDB for retraining.
- **Retrain loop** — GitHub Actions builds a training Docker image, runs ensemble training as a Fargate task, uploads new model artifacts to S3, then dispatches deploy.
- **Infrastructure** — SST v3 (AWS CDK). Defines DynamoDB table, S3 bucket, ECS Fargate cluster + service with ALB, Monitor Lambda + Cron, and Next.js deployment.
- **CI/CD** — GitHub Actions: CI (type-check + lint), deploy (push or repo_dispatch, downloads models from S3), train (monitor-triggered, scheduled, or push to ml/).
- **Local dev** — Docker Compose with DynamoDB Local, the model container, and the Next.js API all wired together.

## Stack

| Layer | Technology | Purpose |
|---|---|---|
| Model training | Python, XGBoost, scikit-learn, imbalanced-learn | Stacking ensemble (XGBoost + RF + LR + meta) |
| Data | IEEE-CIS Fraud Detection (transaction + identity) | Real-world transaction dataset with feature-rich columns |
| Inference container | Python, FastAPI, Docker, ECS Fargate | Model serving via HTTP |
| API server | Next.js 15 (App Router), TypeScript | Predict endpoint, transaction history, health check |
| Database | DynamoDB (pay-per-request) | Prediction log with full input/output/timestamp |
| Infrastructure | SST v3 / AWS CDK | DynamoDB, ECS Fargate, Next.js deployment |
| Monitoring | Evidently AI, CloudWatch, Monitor Lambda | Drift detection, alerting, retrain trigger |
| CI/CD | GitHub Actions | Type-check, lint, train (Fargate), deploy (SST) |
| Artifacts | S3 | Features, model artifacts, reference dataset |
| Feature store | Upstash Redis | Rate limiting + prediction dedup cache |
| Local dev | Docker Compose | DynamoDB Local + model container + API |

## Project Structure

```
fraud-detection/
├── container/                       # Model inference container
│   ├── Dockerfile                   # Python 3.12 slim, uvicorn
│   ├── app.py                       # FastAPI server (/predict, /drift, /feedback, /health)
│   ├── inference.py                 # Ensemble inference logic
│   ├── preprocess.py                # Feature preprocessing (log, scale, impute)
│   ├── schemas.py                   # Pydantic request schemas
│   ├── features.py                  # Feature column definitions
│   ├── drift.py                     # Evidently AI drift computation
│   ├── reference.parquet            # Training reference for drift comparison
│   ├── requirements.txt
│   └── model/                       # Trained artifacts (joblib)
│       ├── xgboost_model.joblib
│       ├── random_forest_model.joblib
│       ├── logistic_reg_model.joblib
│       ├── meta_model.joblib
│       ├── scaler.joblib
│       └── imputer.joblib
│
├── packages/
│   ├── api/                         # Next.js 15 API
│   │   └── src/
│   │       ├── middleware.ts        # API key auth (x-api-key header)
│   │       └── app/api/
│   │           ├── predict/route.ts # POST — run inference
│   │           ├── transactions/route.ts# GET — query predictions
│   │           └── health/route.ts  # GET — health check
│   ├── core/                        # Shared TypeScript types + Redis client
│   │   └── src/
│   │       ├── index.ts             # TransactionInput, PredictionResult, etc.
│   │       └── redis.ts             # Upstash Redis client (rate limit + cache)
│   └── monitor/                     # Drift monitoring Lambda
│       └── src/index.ts             # Handler: calls /drift, dispatches retrain
│
├── ml/
│   ├── requirements.txt             # Training dependencies
│   ├── Dockerfile.train             # Training container image
│   ├── features/
│   │   ├── build_features.py        # Full feature pipeline (IEEE-CIS dataset)
│   │   ├── feature_selection.py     # Feature importance analysis
│   │   └── generate_reference.py    # Reference dataset for drift detection
│   └── training/
│       ├── train.py                 # XGBoost baseline training
│       ├── ensemble.py              # Stacking ensemble training + CV
│       ├── fargate_train.py         # Fargate entrypoint: S3 in/out, full training
│       ├── grid_search.py           # Hyperparameter search
│       ├── threshold_tuning.py      # Fraud-class threshold optimization
│       ├── tune_meta.py             # Meta-model tuning
│       └── feature_select.py        # Feature selection utilities
│
├── sst.config.ts                    # SST v3 infrastructure definition
├── docker-compose.yml               # Local dev environment
├── .github/workflows/
│   ├── ci.yaml                      # Type-check + lint
│   └── deploy.yaml                  # SST deploy to AWS
└── package.json
```

## API Endpoints

### Next.js API (public)

All endpoints except `/api/health` require an `x-api-key` header.

| Method | Path | Description |
|---|---|---|
| POST | `/api/predict` | Submit a transaction, get fraud prediction |
| GET | `/api/transactions` | List recent predictions (limit 50) |
| GET | `/api/transactions?id=<uuid>` | Get a specific prediction |
| GET | `/api/health` | API + container health check (no auth required) |

### Container (internal, via ALB)

The container rejects requests without a valid `x-model-secret` header. Only the Next.js API knows this secret.

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Run inference (called by Next.js API) |
| POST | `/drift` | Compute Evidently drift report vs reference dataset |
| POST | `/feedback` | Attach ground-truth label `{ prediction_id, label }` to a prediction |
| GET | `/health` | Container health check (no auth required) |

### Predict Request

```json
{
  "TransactionAmt": 150.00,
  "card1": 12345,
  "card2": 500.0,
  "addr1": 123.0,
  "P_emaildomain": 5.0
}
```

All fields except `TransactionAmt` are optional — missing values are handled by the preprocessing pipeline.

### Predict Response

```json
{
  "prediction": 1,
  "probability": 0.87,
  "probabilities_per_model": {
    "xgboost": 0.82,
    "random_forest": 0.79,
    "logistic_regression": 0.71
  },
  "model_version": "ensemble_v1"
}
```

## Local Development

```bash
# Start all services (DynamoDB Local + model container + API)
docker compose up

# The API is available at http://localhost:3000
# The model container is at http://localhost:8001

# SST dev (live Lambda reload for AWS dev)
npm run dev
```

## Drift Detection & Retrain Loop

The system automatically detects when production data has drifted from the training distribution and retrains the model without manual intervention.

### How it works

```
                    ┌──────────────────────────────────────┐
                    │  Reference Dataset (training split)  │
                    │  - 80% of features.parquet           │
                    │  - Bundled in container as           │
                    │    reference.parquet                  │
                    └──────────┬───────────────────────────┘
                               │
  Production ─────► DynamoDB   │
  Predictions      (features)  │
    stored with    ────────────┤
    input features             │
       │                       │
       ▼                       ▼
  ┌─────────────────────────────────┐
  │  POST /drift (FastAPI container)│
  │  - Evidently DataDriftPreset   │
  │  - Compares each feature's     │
  │    distribution (PSI)          │
  │  - Reports drift_share,        │
  │    drifted_features list       │
  └──────────────┬──────────────────┘
                 │
      drift_share > 0.30?
            /        \
          NO         YES
          │           │
     ┌────┘           └──────────────────┐
     ▼                                    ▼
  No action                  Monitor Lambda (cron, 6h)
                             ┌──────────────────────────┐
                             │  1. POST /drift          │
                             │  2. Check drift_share    │
                             │  3. If > 30%: POST       │
                             │     GitHub dispatch      │
                             │     event_type: retrain  │
                             └──────────┬───────────────┘
                                        │ dispatch
                                        ▼
                        ┌───────────────────────────────┐
                        │  Train Pipeline (train.yaml)  │
                        │  1. Build training image      │
                        │  2. Push to ECR               │
                        │  3. Run Fargate task:         │
                        │     a. Download features.parquet│
                        │        from S3                │
                        │     b. Train ensemble (5-fold │
                        │        CV, XGBoost + RF + LR) │
                        │     c. Upload models to S3    │
                        │     d. Upload new reference   │
                        │        dataset to S3          │
                        │  4. Dispatch deploy event     │
                        └──────────┬────────────────────┘
                                   │ dispatch
                                   ▼
                        ┌───────────────────────────────┐
                        │  Deploy Pipeline (deploy.yaml)│
                        │  1. Download models from S3   │
                        │  2. npx sst deploy            │
                        │     (zero-downtime rolling    │
                        │      update)                  │
                        └───────────────────────────────┘
```

### Key components

**Reference dataset** — Generated from the training split (80%) of `features.parquet`. Stored as `reference.parquet` and bundled into the container image. This is the baseline distribution that production features are compared against.

**Drift metric** — Uses Evidently AI's `DataDriftPreset` with Population Stability Index (PSI). Each feature column is compared between the production batch and the reference. If the fraction of drifted columns exceeds 30%, drift is flagged.

**Monitor Lambda** — A TypeScript Lambda triggered every 6 hours by an EventBridge cron. It calls `POST /drift` on the model container's ALB. If drift is detected, it posts a `repository_dispatch` event to the GitHub API with event type `retrain`.

**Training pipeline** — A Fargate task that downloads the latest `features.parquet` from S3, runs the full ensemble training (5-fold cross-validated stacking with XGBoost, Random Forest, and Logistic Regression), uploads the new model artifacts and a fresh reference dataset to S3, then dispatches a deploy event.

**Deploy pipeline** — Downloads the latest model artifacts from S3 into `container/model/` before running `sst deploy`, so the new Docker image ships with the freshly trained models.

### Trigger sources

The retrain loop can be triggered three ways:

| Trigger | Source | When |
|---|---|---|
| Monitor Lambda | GitHub API dispatch | Drift detected (> 30% drifted features) |
| Scheduled | `train.yaml` cron | Weekly (Sunday 2am — configured in workflow) |
| Manual | GitHub UI | `workflow_dispatch` via Actions tab |

## Training

The ML pipeline is in `ml/`. To run training locally:

```bash
# Feature engineering (requires IEEE-CIS CSV data in data/)
cd ml/features && python build_features.py

# Train stacking ensemble
cd ml/training && python ensemble.py

# Threshold tuning
python threshold_tuning.py
```

Trained models are copied to `container/model/` for packaging in the Docker image.

## CI/CD

- **CI** — Runs on every push/PR to `main`. TypeScript type-check (`tsc --noEmit`) across the API package, and Python lint (`ruff check`) on container and ML code.
- **CD** — On push to `main` or `repository_dispatch` (from train pipeline), downloads latest model artifacts from S3 into `container/model/`, then runs `sst deploy --stage production` to build and deploy the inference container.
- **Train** — Triggered by `repository_dispatch` (from Monitor Lambda when drift > 30%), `workflow_dispatch`, or push to `ml/`. Builds the training Docker image, runs ensemble training as a Fargate task, uploads new model artifacts + reference dataset to S3, then dispatches the deploy workflow.

## What's Next

- [x] API key authentication (Next.js middleware + x-api-key header)
- [x] Internal model service auth (shared secret between API and container)
- [x] Rate limiting + prediction cache (Upstash Redis, per-card 60s window)
- [ ] Event-driven ingestion (EventBridge + SQS for async transaction processing)
- [ ] MLflow experiment tracking + model registry
- [ ] Inference explainability (SHAP values in response)
- [ ] Monitoring hold in deploy pipeline (30-min CloudWatch watch + rollback)
- [ ] Human review queue for borderline predictions
