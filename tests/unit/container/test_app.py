"""Tests for the FastAPI app: auth middleware and route behavior."""

import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

import app as app_module
from app import app

pytestmark = pytest.mark.mock_dynamodb

SECRET = "test-secret"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("MODEL_SECRET", SECRET)
    return TestClient(app)


class TestAuthMiddleware:
    def test_health_is_open(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_missing_secret_header_is_rejected(self, client):
        response = client.post("/predict", json={"TransactionAmt": 10.0})
        assert response.status_code == 401

    def test_wrong_secret_is_rejected(self, client):
        response = client.post(
            "/predict",
            json={"TransactionAmt": 10.0},
            headers={"x-model-secret": "wrong"},
        )
        assert response.status_code == 401

    def test_unset_model_secret_rejects_all(self, client, monkeypatch):
        monkeypatch.delenv("MODEL_SECRET")
        response = client.post(
            "/predict",
            json={"TransactionAmt": 10.0},
            headers={"x-model-secret": SECRET},
        )
        assert response.status_code == 401


class TestPredictRoute:
    def test_predict_returns_model_result(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "preprocess", lambda transaction: np.zeros((1, 10)))
        monkeypatch.setattr(app_module, "load_models", lambda: {"a": 1})
        monkeypatch.setattr(
            app_module,
            "inference",
            lambda features, **models: {"prediction": 1, "probability": 0.9},
        )
        response = client.post(
            "/predict",
            json={"TransactionAmt": 42.0},
            headers={"x-model-secret": SECRET},
        )
        assert response.status_code == 200
        assert response.json() == {"prediction": 1, "probability": 0.9}

    def test_predict_rejects_invalid_payload(self, client):
        response = client.post(
            "/predict",
            json={"not_a_field": 1},
            headers={"x-model-secret": SECRET},
        )
        assert response.status_code == 422


class TestDriftRoute:
    def test_empty_production_features_reports_no_drift(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "get_production_features", lambda: pd.DataFrame())
        response = client.post(
            "/drift", json={}, headers={"x-model-secret": SECRET}
        )
        assert response.status_code == 200
        assert response.json() == {
            "drift_detected": False,
            "drift_share": 0.0,
            "drifted_features": [],
            "n_drifted": 0,
            "n_total": 0,
        }


class TestFeedbackRoute:
    class FakeTable:
        def __init__(self, error=None):
            self.error = error
            self.calls = []

        def update_item(self, **kwargs):
            self.calls.append(kwargs)
            if self.error:
                raise self.error

    def test_feedback_updates_label(self, client, monkeypatch):
        table = self.FakeTable()
        monkeypatch.setattr(app_module, "_get_table", lambda: table)
        response = client.post(
            "/feedback",
            json={"prediction_id": "abc-123", "label": 1},
            headers={"x-model-secret": SECRET},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "prediction_id": "abc-123", "label": 1}
        assert table.calls[0]["Key"] == {"id": "abc-123"}

    def test_feedback_missing_prediction_is_404(self, client, monkeypatch):
        error = ClientError(
            {"Error": {"Code": "ConditionalCheckFailedException", "Message": "missing"}},
            "UpdateItem",
        )
        monkeypatch.setattr(app_module, "_get_table", lambda: self.FakeTable(error))
        response = client.post(
            "/feedback",
            json={"prediction_id": "nope", "label": 0},
            headers={"x-model-secret": SECRET},
        )
        assert response.status_code == 404
