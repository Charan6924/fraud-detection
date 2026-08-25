# Codebase Gaps

This is the current follow-up list after the P0 deployment fixes. Items are
ordered roughly by production impact. Each item includes the evidence location
and a practical acceptance condition for closing it.

## High Priority

### GAP-01: Local Compose prediction flow is incomplete

Status:  Done for Upstash local development; offline Redis mode remains optional.

Evidence:

- `docker-compose.yml` has DynamoDB, model, and API services but no Redis service.
- The API service does not receive `API_KEY` or `MODEL_SECRET`.
- The model service does not receive `MODEL_SECRET`.
- Redis is initialized from `UPSTASH_REDIS_REST_URL` and
  `UPSTASH_REDIS_REST_TOKEN`.

Impact: `docker compose up` does not provide a working local prediction path.
Health may work, but authenticated prediction and Redis operations do not.

Acceptance:

- Compose starts API, model, DynamoDB, and a local Redis-compatible service.
- A documented local request can authenticate, predict, cache, rate-limit, and
  persist a record without AWS, GitHub, Upstash, or other cloud credentials.

Evidence files: `docker-compose.yml`, `packages/api/src/middleware.ts`,
`packages/core/src/redis.ts`, `container/app.py`

### GAP-02: Deployment secrets are not passed to SST

Status: 

Evidence: `sst.config.ts` reads `API_KEY`, `MODEL_SECRET`, Upstash variables,
and `GH_TOKEN` from `process.env`, but `.github/workflows/deploy.yaml` only
configures AWS credentials before running `sst deploy`.

Impact: A deployment can complete with missing application authentication,
Redis, or monitoring configuration.

Acceptance:

- The deploy workflow explicitly maps required GitHub secrets into the SST
  process environment.
- The workflow fails before deployment when required values are missing.
- Secret values never appear in logs.

Evidence files: `.github/workflows/deploy.yaml`, `sst.config.ts`

### GAP-03: Drift data is on incompatible scales

Status: Open

Evidence: training features are log-transformed, encoded, and standardized in
`ml/features/build_features.py`. Production drift data is read directly from
the raw `input` stored by the API. `container/drift.py` compares these two
representations.

Impact: Drift results can be false positives or false negatives, causing missed
or unnecessary retraining.

Acceptance:

- Production features and the reference dataset use the same preprocessing
  pipeline and column order.
- A contract test proves identical-distribution production/reference data does
  not report drift.
- A known shifted feature reliably reports drift.

Evidence files: `ml/features/build_features.py`, `container/app.py`,
`container/drift.py`, `ml/features/generate_reference.py`

### GAP-04: Retraining evaluates a different threshold than serving

Status: Open

Evidence: `ml/training/fargate_train.py` uses `meta.predict()`, which applies a
default threshold of `0.5`. `container/inference.py` serves fraud when the
probability is at least `0.10`.

Impact: Reported F1/FNR and any future promotion decision do not represent
actual production behavior.

Acceptance:

- Threshold is a versioned model artifact or configuration value.
- Training evaluation and inference use the same threshold.
- Promotion gates evaluate the same prediction rule used in serving.

Evidence files: `ml/training/fargate_train.py`, `container/inference.py`,
`ml/training/tune_meta.py`

### GAP-05: Model health is not a readiness check

Status: Open

Evidence: `GET /health` returns `{"status": "ok"}` without loading or
validating model artifacts. Models are loaded lazily on the first prediction.

Impact: ECS can mark a container healthy and route traffic to a container that
cannot serve predictions.

Acceptance:

- Readiness validates all required model artifacts and reference data.
- ECS health checks use readiness, not only process liveness.
- Missing or incompatible artifacts produce a non-healthy status before traffic
  is accepted.

Evidence files: `container/app.py`, `container/inference.py`,
`container/preprocess.py`, `container/Dockerfile`

### GAP-06: Model secret is sent through an HTTP ALB listener

Status: Open

Evidence: SST configures the model service load balancer with `80/http`, while
the API sends `x-model-secret` to the model URL.

Impact: The shared model secret can be exposed if traffic crosses an untrusted
network or the ALB is publicly reachable.

Acceptance:

- Model traffic uses HTTPS or remains entirely inside a private network with
  security-group restrictions.
- The model ALB is not publicly reachable unless HTTPS and access controls are
  explicitly configured.

Evidence file: `sst.config.ts`

## Medium Priority

### GAP-07: Prediction cache identity is incomplete

Status: Open

Evidence: Cache keys contain only `card1` and `TransactionAmt`.

Impact: Two different transactions with the same card and amount can receive a
stale or incorrect prediction.

Acceptance:

- Cache identity includes every input field that affects model output, or
  caching is removed until a canonical request hash is implemented.
- A test proves different transaction payloads do not share results.

Evidence files: `packages/core/src/redis.ts`,
`packages/api/src/app/api/predict/route.ts`

### GAP-08: Missing card identifiers share one cache/rate-limit key

Status: Open

Evidence: `String(input.card1)` converts every missing `card1` value to the
same string, `"undefined"`.

Impact: Unrelated requests without `card1` share rate-limit and cache state.

Acceptance:

- Requests require a stable identity field, or a different safe identity/rate
  limiting strategy is used when `card1` is absent.
- Missing-identity behavior is documented and tested.

Evidence file: `packages/api/src/app/api/predict/route.ts`

### GAP-09: Cache is written before prediction persistence

Status: Open

Evidence: The API writes the prediction cache before writing the DynamoDB
record, with no transaction or recovery path.

Impact: A DynamoDB failure can leave a cache entry for a prediction that was
never persisted.

Acceptance:

- Persistence and cache ordering is deliberate and tested.
- A failed database write cannot create an apparently successful durable cache
  result, or the cache entry is removed on failure.

Evidence file: `packages/api/src/app/api/predict/route.ts`

### GAP-10: DynamoDB failure test does not exercise the route client

Status: Open

Evidence: The test obtains a new mocked `DynamoDBDocumentClient` instance after
the route module has already captured another instance.

Impact: The test can pass without proving the route's actual DynamoDB failure
behavior.

Acceptance:

- The test controls the exact client used by the route.
- The expected API behavior for database failure is explicitly defined and
  verified.

Evidence files: `tests/unit/api/predict.test.ts`,
`packages/api/src/app/api/predict/route.ts`

### GAP-11: Transaction listing is not reliably the latest 50

Status: Open

Evidence: The endpoint scans with `Limit: 50` and sorts only after the scan.
There is no timestamp index or pagination.

Impact: Results may omit newer records and table scans become increasingly
expensive as data grows.

Acceptance:

- Add a queryable timestamp/index strategy or explicitly document the intended
  access pattern.
- Implement pagination and prove the returned page ordering.

Evidence file: `packages/api/src/app/api/transactions/route.ts`

### GAP-12: Drift monitoring samples only one arbitrary DynamoDB page

Status: Open

Evidence: The model service calls `Scan(Limit=500)` and ignores
`LastEvaluatedKey`. It also does not order records by timestamp.

Impact: Drift monitoring may repeatedly analyze an old or unrepresentative
subset of predictions.

Acceptance:

- Monitor a defined recent time window or use a timestamp-based access pattern.
- Handle pagination or explicitly bound the sampling strategy.

Evidence file: `container/app.py`

### GAP-13: Forks dispatch retraining to the original repository

Status: Open

Evidence: The monitor hardcodes
`https://api.github.com/repos/Charan6924/fraud-detection/dispatches`.

Impact: An open-source fork can trigger workflows in the upstream repository,
or fail to trigger its own workflow.

Acceptance:

- Repository owner/name is supplied through deployment configuration or derived
  from the deployment context.
- A fork-safe test covers the dispatch target.

Evidence file: `packages/monitor/src/index.ts`

### GAP-14: Monitor network calls have no timeout or failure boundary

Status: Open

Evidence: Monitor `fetch` calls have no timeout, retry policy, or outer
exception handling.

Impact: A stalled model or GitHub request can consume the full Lambda timeout
and produce an opaque scheduled-job failure.

Acceptance:

- Add bounded timeouts and controlled error results.
- Emit enough structured context to distinguish model, GitHub, and parsing
  failures.
- Test timeout and malformed-response paths.

Evidence file: `packages/monitor/src/index.ts`

### GAP-15: SST and workflow use different artifact buckets

Status: Open

Evidence: SST creates an `Artifacts` bucket and returns its generated name,
while workflows use the hardcoded bucket
`fraud-detection-artifacts-charan`.

Impact: Infrastructure can create one bucket while CI/CD reads and writes
another, leaving deployments without artifacts.

Acceptance:

- Define one source of truth for the artifact bucket.
- Training, publishing, and deployment all use the same resolved bucket name.
- A fresh environment test proves the bucket wiring end to end.

Evidence files: `sst.config.ts`, `.github/workflows/deploy.yaml`,
`.github/workflows/train.yaml`, `scripts/publish_artifacts.sh`

### GAP-16: CI does not build the deployable artifacts

Status: Open

Evidence: CI runs type-checking, unit tests, Ruff, and pytest, but does not run
the Next.js production build, model Docker build, API Docker build, or a local
Compose smoke test.

Impact: CI can pass while deployment fails due to missing artifacts, Docker
build errors, runtime environment problems, or packaging issues.

Acceptance:

- Add a safe CI build validation that uses generated/test artifacts where
  appropriate.
- Add a Compose or container smoke test that does not require cloud secrets.
- Keep real AWS deployment as a separate protected environment job.

Evidence file: `.github/workflows/ci.yaml`

### GAP-17: Production model version is hardcoded

Status: Open

Evidence: Every inference response returns `"ensemble_v1"` regardless of the
actual artifact revision.

Impact: Predictions cannot be reliably traced to a model artifact, and rollback
verification is difficult.

Acceptance:

- Publish an immutable model version with the artifact set.
- Return that version in inference responses and prediction records.
- Include the version in deployment logs and monitoring.

Evidence file: `container/inference.py`

### GAP-18: Deployment uses mutable `latest` training images

Status: Open

Evidence: The training workflow pushes and registers the image tag `latest`.

Impact: Runs are not reproducible and an old or overwritten image cannot be
unambiguously identified for rollback.

Acceptance:

- Tag images with commit SHA or model version.
- Deploy by immutable image digest or version tag.
- Retain enough metadata to reproduce a training/deployment pair.

Evidence file: `.github/workflows/train.yaml`

## Notes

- Cloud credentials are intentionally not stored in this public repository.
- AWS/GitHub/Upstash-dependent items should remain opt-in and secret-gated.
- The artifact bootstrap changes that generated the local reference dataset are
  currently uncommitted and should be reviewed separately.
