"""Tests for the feature preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS
import preprocess
from schemas import TransactionInput


class IdentityScaler:
    """Stand-in for the fitted StandardScaler; returns input unchanged."""

    def transform(self, df):
        return df.to_numpy()


@pytest.fixture(autouse=True)
def fake_scaler(monkeypatch):
    monkeypatch.setattr(preprocess, "get_scaler", lambda: IdentityScaler())


def _payload(**overrides):
    base = {"TransactionAmt": 100.0}
    base.update(overrides)
    return TransactionInput(**base)


class TestImputation:
    def test_log1p_transforms_amount(self):
        out = preprocess.preprocess(_payload(TransactionAmt=99.0))
        assert out[0, FEATURE_COLUMNS.index("TransactionAmt")] == np.log1p(99.0)

    def test_categorical_nans_filled_with_zero(self):
        out = preprocess.preprocess(_payload())
        df = pd.DataFrame(out, columns=FEATURE_COLUMNS)
        for col in preprocess._IMPUTE_CAT:
            assert df[col].iloc[0] == 0.0

    def test_numeric_nans_filled_with_minus_one(self):
        out = preprocess.preprocess(_payload())
        df = pd.DataFrame(out, columns=FEATURE_COLUMNS)
        for col in preprocess._IMPUTE_NEG1:
            assert df[col].iloc[0] == -1.0

    def test_provided_values_are_not_overwritten(self):
        v_col = next(c for c in preprocess._IMPUTE_NEG1 if c.startswith("V"))
        out = preprocess.preprocess(_payload(card2=5.0, **{v_col: 1.5}, ProductCD=2.0))
        df = pd.DataFrame(out, columns=FEATURE_COLUMNS)
        assert df["card2"].iloc[0] == 5.0
        assert df[v_col].iloc[0] == 1.5
        assert df["ProductCD"].iloc[0] == 2.0


class TestOutputShape:
    def test_single_row_with_feature_column_order(self):
        out = preprocess.preprocess(_payload())
        assert out.shape == (1, len(FEATURE_COLUMNS))
        assert list(FEATURE_COLUMNS[:5]) == [
            "TransactionAmt",
            "ProductCD",
            "card1",
            "card2",
            "card3",
        ]

    def test_all_values_finite_when_every_field_provided(self):
        fields = {name: 0.5 for name in TransactionInput.model_fields if name != "TransactionAmt"}
        payload = TransactionInput(TransactionAmt=100.0, **fields)
        out = preprocess.preprocess(payload)
        assert np.isfinite(out).all()
