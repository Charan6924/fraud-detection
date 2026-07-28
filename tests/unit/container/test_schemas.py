"""Tests for Pydantic schema validation."""

import pytest
from pydantic import ValidationError
from schemas import TransactionInput, FeedbackInput


class TestTransactionInput:
    def test_minimal_valid(self):
        txn = TransactionInput(TransactionAmt=100.5)
        assert txn.TransactionAmt == 100.5

    def test_all_optional_fields_none_by_default(self):
        txn = TransactionInput(TransactionAmt=50.0)
        assert txn.C1 is None
        assert txn.card1 is None
        assert txn.V2 is None
        assert txn.id_01 is None
        assert txn.ProductCD is None

    def test_with_optional_fields(self):
        txn = TransactionInput(
            TransactionAmt=250.0,
            card1=12345,
            card2=678,
            ProductCD=3.0,
            addr1=1001,
            dist1=50.0,
            P_emaildomain=5.0,
            C1=0.5,
            D1=1.0,
            M2=0.0,
            V2=-0.3,
            id_01=1.0,
        )
        assert txn.TransactionAmt == 250.0
        assert txn.card1 == 12345
        assert txn.dist1 == 50.0
        assert txn.C1 == 0.5

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            TransactionInput()

    def test_required_field_none_raises(self):
        with pytest.raises(ValidationError):
            TransactionInput(TransactionAmt=None)

    def test_wrong_type_raises(self):
        with pytest.raises(ValidationError):
            TransactionInput(TransactionAmt="not-a-number")

    def test_extra_fields_allowed_by_pydantic_v2(self):
        txn = TransactionInput(TransactionAmt=10.0, unknown_field="value")
        assert txn.TransactionAmt == 10.0

    def test_negative_amount(self):
        txn = TransactionInput(TransactionAmt=-1.0)
        assert txn.TransactionAmt == -1.0

    def test_zero_amount(self):
        txn = TransactionInput(TransactionAmt=0.0)
        assert txn.TransactionAmt == 0.0

    def test_large_amount(self):
        txn = TransactionInput(TransactionAmt=1e9)
        assert txn.TransactionAmt == 1e9


class TestFeedbackInput:
    def test_valid_label_0(self):
        fb = FeedbackInput(prediction_id="abc-123", label=0)
        assert fb.prediction_id == "abc-123"
        assert fb.label == 0

    def test_valid_label_1(self):
        fb = FeedbackInput(prediction_id="def-456", label=1)
        assert fb.prediction_id == "def-456"
        assert fb.label == 1

    def test_invalid_label_raises(self):
        with pytest.raises(ValidationError):
            FeedbackInput(prediction_id="abc", label=2)

    def test_non_integer_label_raises(self):
        with pytest.raises(ValidationError):
            FeedbackInput(prediction_id="abc", label="fraud")

    def test_missing_prediction_id_raises(self):
        with pytest.raises(ValidationError):
            FeedbackInput(label=1)

    def test_missing_label_raises(self):
        with pytest.raises(ValidationError):
            FeedbackInput(prediction_id="abc")

    def test_empty_prediction_id(self):
        fb = FeedbackInput(prediction_id="", label=0)
        assert fb.prediction_id == ""
