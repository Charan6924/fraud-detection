"""Tests for Evidently-based drift detection."""

import numpy as np
import pandas as pd
import pytest

from drift import check_drift
from features import FEATURE_COLUMNS

pytestmark = pytest.mark.slow


def _synthetic(rows, seed, shift_col=None):
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1, size=(rows, len(FEATURE_COLUMNS)))
    if shift_col is not None:
        idx = FEATURE_COLUMNS.index(shift_col)
        data[:, idx] += 3.0
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


def _report(current, reference, tmp_path):
    ref_path = tmp_path / "reference.parquet"
    reference.to_parquet(ref_path)
    return check_drift(current, ref_path=str(ref_path))


class TestNoDrift:
    def test_identical_data_reports_no_drift(self, tmp_path):
        reference = _synthetic(rows=200, seed=42)
        report = _report(reference, reference, tmp_path)
        assert report["drift_share"] == 0.0
        assert not any(v["drift_detected"] for v in report["drift_by_columns"].values())
        assert report["n_total"] == len(FEATURE_COLUMNS)


class TestDrift:
    def test_shifted_column_is_detected(self, tmp_path):
        reference = _synthetic(rows=200, seed=42)
        current = _synthetic(rows=200, seed=42, shift_col="V2")
        report = _report(current, reference, tmp_path)
        assert report["n_drifted"] >= 1
        assert report["drift_by_columns"]["V2"]["drift_detected"] is True
