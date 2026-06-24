"""Drift detection using Evidently AI data drift presets."""

from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset
from features import FEATURE_COLUMNS
import pandas as pd
import os


def check_drift(current: pd.DataFrame):
    schema = DataDefinition(numerical_columns=FEATURE_COLUMNS)

    current_data = Dataset.from_pandas(current, data_definition=schema)
    ref_path = os.path.join(os.path.dirname(__file__), "reference.parquet")
    reference = pd.read_parquet(ref_path)
    reference_data = Dataset.from_pandas(reference, data_definition=schema)

    report = Report([DataDriftPreset()])
    report.run(current_data=current_data, reference_data=reference_data)
    return report.as_dict()
