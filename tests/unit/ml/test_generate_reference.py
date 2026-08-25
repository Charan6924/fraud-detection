import importlib.util
from pathlib import Path

import pandas as pd

_MODULE_PATH = Path(__file__).parents[3] / "ml" / "features" / "generate_reference.py"
_SPEC = importlib.util.spec_from_file_location("generate_reference", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
generate = _MODULE.generate


def test_generate_preserves_feature_columns(tmp_path):
    source = tmp_path / "features.parquet"
    output = tmp_path / "reference.parquet"
    pd.DataFrame(
        {
            "TransactionAmt": list(range(1, 11)),
            "card1": list(range(10, 20)),
            "isFraud": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        }
    ).to_parquet(source)

    generate(source, output)

    reference = pd.read_parquet(output)
    assert list(reference.columns) == ["TransactionAmt", "card1"]
    assert len(reference) == 8
