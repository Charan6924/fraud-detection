# Weekend Plan: Drift Detection → Retrain → Canary Deploy Pipeline

## Context

The fraud detection system is deployed end-to-end (Fargate container serving ensemble model + Next.js API + DynamoDB + CI/CD). But it's a static deployment — if fraud patterns shift, the model degrades silently. We need the monitoring-and-retrain loop: detect when production data is statistically different from training, trigger a retrain on fresh data, and deploy the new model with zero downtime.

## Architecture Overview

```
Train Data ──► Reference Dataset (S3)
                    │
Production ──► DynamoDB (predictions with features)
                    │
       Monitor Lambda (cron, every 6h) ──► POST /drift ──► Evidently
                    │                                    compares prod vs ref
                    │
            drift > 30%?
                    │
               GitHub API ──► train.yaml ──► Fargate RunTask ──► deploy.yaml
                    │                                            │
               S3 (new model artifacts)               rebuild container + canary deploy
```

## Four Work Items

---

### 1. Drift Detection Endpoint (`/drift`) + Reference Dataset

**What changes:**
- `ml/generate_reference.py` — Load `features.parquet`, take training split (80%), save as `container/reference.parquet`. Run this locally once.
- `container/requirements.txt` — Add `evidently`, `boto3`
- `container/drift.py` — **New file.** Evidently `DataDriftPreset` comparing production feature batch vs reference
- `container/app.py` — Add `POST /drift` route. Accepts JSON array of production rows, runs drift, returns `{ drift_ratio, drifted_features, n_drifted, n_total }`
- Reference dataset bundled into the Docker image (copied in Dockerfile)

**SST config:**
- Link `predictionsTable` to the container service (`link: [predictionsTable]`) for drift + feedback endpoints

---

### 2. Label Feedback Endpoint (`/feedback`)

**What changes:**
- `container/app.py` — Add `POST /feedback` route. Accepts `{ prediction_id: string, label: 0 | 1 }`. Updates DynamoDB predictions table to add a `label` field.

---

### 3. Monitor Lambda (TypeScript, cron)

**What changes:**
- `packages/monitor/src/index.ts` — **New file.** Lambda handler:
  1. Query DynamoDB scan for latest predictions (limit 500)
  2. Extract feature arrays from `Item.input`
  3. Call `POST /drift` on the model container ALB
  4. If `drift_ratio > 0.30`, call GitHub API to dispatch `train.yaml`
- `sst.config.ts` — Add Cron + Function resources for the monitor
- New `packages/monitor/package.json` with dependencies

---

### 4. Training Fargate Task + Train Pipeline

**What changes:**
- `ml/Dockerfile.train` — Same Python base as `container/Dockerfile`, copies training scripts.
- `ml/training/fargate_train.py` — **New file.** Entrypoint for Fargate:
  1. Download `features.parquet` from S3
  2. Train ensemble (same logic as ensemble.py but `device="cpu"`, simplified CV)
  3. Upload model artifacts + new reference dataset to S3
- `.github/workflows/train.yaml` — **New file.** Triggered by `workflow_dispatch`, schedule, or push to `ml/`:
  1. Build & push training image to ECR
  2. Run Fargate task via `aws ecs run-task`
  3. Poll until completed
  4. If success, trigger `deploy.yaml` via repository_dispatch
- `sst.config.ts` — Add S3 bucket for model artifacts

---

### 5. Deploy Pipeline Update (Rolling with Monitoring Hold)

**What changes:**
- `.github/workflows/deploy.yaml` — Modified deploy:
  1. Download latest model artifacts from S3 (`aws s3 cp`)
  2. Copy into `container/model/`
  3. Build & push inference container to ECR
  4. Run `npx sst deploy`
  5. **Monitoring hold** — 30-minute CloudWatch watch
  6. Rollback via previous ECR image tag if alarms fire
- Rolling update (SST default). True weighted canary would need CodeDeploy — scope it separately.

---

### Files Modified / Created

| File | Action |
|---|---|
| `container/requirements.txt` | Add `evidently`, `boto3` |
| `container/drift.py` | **New** — Evidently drift computation |
| `container/app.py` | Add `/drift`, `/feedback` routes |
| `container/reference.parquet` | **New** — generated training reference |
| `container/Dockerfile` | Copy drift.py, reference.parquet |
| `sst.config.ts` | Link DynamoDB to container, add S3 bucket, add Cron + Monitor Function |
| `packages/monitor/src/index.ts` | **New** — Monitor Lambda handler |
| `packages/monitor/package.json` | **New** |
| `packages/monitor/tsconfig.json` | **New** |
| `packages/core/src/index.ts` | Add types for drift request/response, feedback request |
| `ml/generate_reference.py` | Update to generate reference.parquet from training split |
| `ml/Dockerfile.train` | **New** — Training container image |
| `ml/training/fargate_train.py` | **New** — Fargate training entrypoint |
| `.github/workflows/train.yaml` | **New** — Retrain pipeline |
| `.github/workflows/deploy.yaml` | Updated — pull from S3, monitoring hold |

### Build Order

Work items must be built in order: 1 → 2 → 3 → 4 → 5, because each depends on infra or artifacts from the previous one.

### Verification

1. **Local drift test**: Start container locally, call `POST /drift` with sample features, verify drift report returned
2. **Local feedback test**: `POST /feedback { prediction_id, label }`, verify DynamoDB record updated
3. **Monitor Lambda**: Deploy SST, verify cron triggers, check CloudWatch logs
4. **Full loop**: Set low drift threshold → submit drifted predictions → verify retrain triggers → verify new model deploys
