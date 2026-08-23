"""Drift detection using Evidently AI data drift presets."""

from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset
from features import FEATURE_COLUMNS
import pandas as pd
import os


DRIFT_THRESHOLD = 0.05


def check_drift(current: pd.DataFrame, ref_path: str = None) -> dict:
    schema = DataDefinition(numerical_columns=FEATURE_COLUMNS)

    current_data = Dataset.from_pandas(current, data_definition=schema)
    if ref_path is None:
        ref_path = os.path.join(os.path.dirname(__file__), "reference.parquet")
    reference = pd.read_parquet(ref_path)
    reference_data = Dataset.from_pandas(reference, data_definition=schema)

    snapshot = Report([DataDriftPreset()]).run(
        current_data=current_data, reference_data=reference_data
    )
    return _summarize(snapshot.dict())


def _summarize(report_dict: dict) -> dict:
    """Collapse the DataDriftPreset snapshot into share + per-column map."""
    drift_share = 0.0
    drift_by_columns = {}
    for metric in report_dict["metrics"]:
        name = metric.get("metric_name", "")
        if name.startswith("ValueDrift(column="):
            column = name.split("column=")[1].split(",")[0]
            p_value = metric["value"]
            drift_by_columns[column] = {
                "drift_detected": bool(isinstance(p_value, float) and p_value < DRIFT_THRESHOLD)
            }
        elif name.startswith("DriftedColumnsCount"):
            drift_share = metric["value"]["share"]
    return {
        "drift_share": round(float(drift_share), 4),
        "drift_by_columns": drift_by_columns,
        "n_drifted": sum(v["drift_detected"] for v in drift_by_columns.values()),
        "n_total": len(drift_by_columns),
    }
