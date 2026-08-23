"""Tests for the ensemble inference pipeline."""

import numpy as np

from inference import inference, THRESHOLD


class FixedProb:
    """Model stub returning a fixed fraud probability."""

    def __init__(self, prob):
        self.prob = prob
        self.inputs = []

    def predict_proba(self, inp):
        self.inputs.append(np.array(inp))
        return np.array([[1 - self.prob, self.prob]])


class RecordingImputer:
    def __init__(self):
        self.inputs = []

    def transform(self, inp):
        self.inputs.append(np.array(inp))
        return np.array(inp)


def _stubs(xgb_prob=0.2, rf_prob=0.3, lr_prob=0.4, meta_prob=0.05):
    stubs = {
        "lr": FixedProb(lr_prob),
        "rf": FixedProb(rf_prob),
        "xgb": FixedProb(xgb_prob),
        "imputer": RecordingImputer(),
    }
    meta = FixedProb(meta_prob)
    return stubs, meta


class TestEnsembleComposition:
    def test_probabilities_per_model_from_stubs(self):
        stubs, meta = _stubs(xgb_prob=0.2, rf_prob=0.3, lr_prob=0.4)
        inp = np.array([[0.1, -0.5]])
        result = inference(inp, meta=meta, **stubs)
        assert result["probabilities_per_model"] == {
            "xgboost": 0.2,
            "random_forest": 0.3,
            "logistic_regression": 0.4,
        }

    def test_meta_receives_stacked_probabilities(self):
        stubs, meta = _stubs(xgb_prob=0.2, rf_prob=0.3, lr_prob=0.4)
        inp = np.array([[0.1, -0.5]])
        inference(inp, meta=meta, **stubs)
        np.testing.assert_allclose(meta.inputs[0], np.array([[0.2, 0.3, 0.4]]))

    def test_imputer_used_for_tree_and_linear_models(self):
        stubs, meta = _stubs()
        inp = np.array([[0.1, -0.5]])
        inference(inp, meta=meta, **stubs)
        np.testing.assert_allclose(stubs["imputer"].inputs[0], inp)

    def test_model_version_is_ensemble_v1(self):
        stubs, meta = _stubs()
        result = inference(np.array([[0.0, 0.0]]), meta=meta, **stubs)
        assert result["model_version"] == "ensemble_v1"


class TestThreshold:
    def test_below_threshold_is_legitimate(self):
        stubs, meta = _stubs(meta_prob=THRESHOLD - 0.0001)
        result = inference(np.array([[0.0, 0.0]]), meta=meta, **stubs)
        assert result["prediction"] == 0
        assert result["probability"] == THRESHOLD - 0.0001

    def test_at_threshold_is_fraud(self):
        stubs, meta = _stubs(meta_prob=THRESHOLD)
        result = inference(np.array([[0.0, 0.0]]), meta=meta, **stubs)
        assert result["prediction"] == 1
        assert result["probability"] == THRESHOLD
