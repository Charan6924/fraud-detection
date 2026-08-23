"""Edge case tests for preprocessing extreme or degenerate payloads."""

import numpy as np
import pytest

from features import FEATURE_COLUMNS
import preprocess
from schemas import TransactionInput


class IdentityScaler:
    def transform(self, df):
        return df.to_numpy()


@pytest.fixture(autouse=True)
def fake_scaler(monkeypatch):
    monkeypatch.setattr(preprocess, "get_scaler", lambda: IdentityScaler())


class TestAmountExtremes:
    def test_zero_amount_stays_finite(self):
        out = preprocess.preprocess(TransactionInput(TransactionAmt=0.0))
        assert np.isfinite(out[0, FEATURE_COLUMNS.index("TransactionAmt")])

    def test_large_amount_stays_finite(self):
        out = preprocess.preprocess(TransactionInput(TransactionAmt=1e9))
        assert np.isfinite(out[0, FEATURE_COLUMNS.index("TransactionAmt")])

    def test_negative_amount_does_not_crash(self):
        """log1p(-1) = -inf is current behavior; assert no crash, not a fix."""
        out = preprocess.preprocess(TransactionInput(TransactionAmt=-1.0))
        assert np.isneginf(out[0, FEATURE_COLUMNS.index("TransactionAmt")])


class TestDegeneratePayloads:
    def test_minimal_payload(self):
        out = preprocess.preprocess(TransactionInput(TransactionAmt=100.0))
        assert out.shape == (1, len(FEATURE_COLUMNS))

    def test_every_optional_field_none(self):
        payload = TransactionInput(TransactionAmt=100.0)
        out = preprocess.preprocess(payload)
        assert out.shape == (1, len(FEATURE_COLUMNS))

    def test_extra_schema_fields_are_tolerated(self):
        payload = TransactionInput(TransactionAmt=100.0, unknown_field=1.0)
        out = preprocess.preprocess(payload)
        assert out.shape == (1, len(FEATURE_COLUMNS))
