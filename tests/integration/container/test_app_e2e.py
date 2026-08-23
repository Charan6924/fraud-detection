"""End-to-end container tests against the real model artifacts.

Skipped by default (RUN_INTEGRATION=1 to run) because they need the
gitignored container/model/*.joblib files and reference.parquet.
"""

import os

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app

MODEL_DIR = os.path.join(os.path.dirname(app_module.__file__), "model")
REFERENCE = os.path.join(os.path.dirname(app_module.__file__), "reference.parquet")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.path.isdir(MODEL_DIR) or not os.path.isfile(REFERENCE),
        reason="model artifacts or reference.parquet not present",
    ),
]

SECRET = "e2e-secret"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("MODEL_SECRET", SECRET)
    monkeypatch.setenv("PREDICTIONS_TABLE", "e2e-table")
    return TestClient(app)


class TestPredictRoundTrip:
    def test_health(self, client):
        assert client.get("/health").status_code == 200

    def test_real_prediction_response_shape(self, client):
        response = client.post(
            "/predict",
            json={"TransactionAmt": 25.0, "card1": 12345, "addr1": 1001},
            headers={"x-model-secret": SECRET},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in (0, 1)
        assert 0.0 <= body["probability"] <= 1.0
        assert set(body["probabilities_per_model"]) == {
            "xgboost",
            "random_forest",
            "logistic_regression",
        }
        assert body["model_version"] == "ensemble_v1"
