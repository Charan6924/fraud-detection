#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_BUCKET="${ARTIFACTS_BUCKET:?Set ARTIFACTS_BUCKET to the S3 bucket name}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-ap-southeast-2}}"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

command -v aws >/dev/null || { echo "aws CLI is required" >&2; exit 1; }
command -v "$PYTHON" >/dev/null || { echo "Python interpreter not found: $PYTHON" >&2; exit 1; }

FEATURES="$PROJECT_ROOT/data/features.parquet"
REFERENCE="$PROJECT_ROOT/container/reference.parquet"
MODEL_DIR="$PROJECT_ROOT/container/model"

test -s "$FEATURES" || { echo "Missing $FEATURES" >&2; exit 1; }

for artifact in \
  meta_model.joblib \
  xgboost_model.joblib \
  random_forest_model.joblib \
  logistic_reg_model.joblib \
  imputer.joblib \
  scaler.joblib; do
  test -s "$MODEL_DIR/$artifact" || {
    echo "Missing $MODEL_DIR/$artifact" >&2
    exit 1
  }
done

"$PYTHON" "$PROJECT_ROOT/ml/features/generate_reference.py" \
  --input "$FEATURES" \
  --output "$REFERENCE"

aws s3 cp "$FEATURES" "s3://$ARTIFACTS_BUCKET/features.parquet" --region "$REGION"
aws s3 sync "$MODEL_DIR/" "s3://$ARTIFACTS_BUCKET/models/" --region "$REGION"
aws s3 cp "$REFERENCE" "s3://$ARTIFACTS_BUCKET/reference.parquet" --region "$REGION"

echo "Published model artifacts, features, and reference dataset to s3://$ARTIFACTS_BUCKET"
