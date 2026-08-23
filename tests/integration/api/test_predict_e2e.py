"""End-to-end test of the Next.js API against a live model service.

Skipped by default: requires RUN_INTEGRATION=1 plus a running API
(MODEL_SERVICE_URL pointing at the container) and its secret.
"""

import os

import httpx
import pytest

pytestmark = pytest.mark.integration

SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "")
SECRET = os.environ.get("MODEL_SECRET", "e2e-secret")


@pytest.fixture(scope="module")
def service():
    if not SERVICE_URL:
        pytest.skip("MODEL_SERVICE_URL not set")
    return SERVICE_URL


class TestApiToModelFlow:
    def test_health(self, service):
        response = httpx.get(f"{service}/health", timeout=10)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_predict_through_live_service(self, service):
        response = httpx.post(
            f"{service}/predict",
            json={"TransactionAmt": 25.0, "card1": 12345},
            headers={"x-model-secret": SECRET},
            timeout=30,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in (0, 1)
        assert "probability" in body
        assert body["model_version"] == "ensemble_v1"

    def test_predict_rejects_bad_secret(self, service):
        response = httpx.post(
            f"{service}/predict",
            json={"TransactionAmt": 25.0},
            headers={"x-model-secret": "wrong-secret"},
            timeout=30,
        )
        assert response.status_code == 401
