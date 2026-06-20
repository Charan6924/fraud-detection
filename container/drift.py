from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset
from features import FEATURE_COLUMNS
import pandas as pd


def check_drift(current: pd.DataFrame):
    schema = DataDefinition(numerical_columns=FEATURE_COLUMNS)

    current_data = Dataset.from_pandas(current, data_definition=schema)
    reference = pd.read_parquet("reference.parquet")
    reference_data = Dataset.from_pandas(reference, data_definition=schema)

    report = Report([DataDriftPreset()])
    report.run(current_data=current_data, reference_data=reference_data)
    return report.as_dict()
