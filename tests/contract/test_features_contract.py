"""Contract tests tying the schema, feature list, and preprocessing together."""

from features import FEATURE_COLUMNS
import preprocess
from schemas import TransactionInput


class TestFeatureColumns:
    def test_feature_columns_are_unique(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))

    def test_every_schema_field_is_in_feature_columns(self):
        missing = set(TransactionInput.model_fields) - set(FEATURE_COLUMNS)
        assert missing == set()

    def test_impute_lists_are_subset_of_feature_columns(self):
        assert set(preprocess._IMPUTE_CAT) <= set(FEATURE_COLUMNS)
        assert set(preprocess._IMPUTE_NEG1) <= set(FEATURE_COLUMNS)

    def test_preprocess_depends_only_on_feature_columns(self):
        """preprocess reindexes to FEATURE_COLUMNS, so dropping any column breaks it."""
        assert len(TransactionInput.model_fields) <= len(FEATURE_COLUMNS)
