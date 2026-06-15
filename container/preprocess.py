import pandas as pd
import numpy as np
from schemas import TransactionInput
import joblib
import os
from features import FEATURE_COLUMNS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
SCALER = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

# Columns filled with -1 in training (same as build_features.py)
_IMPUTE_NEG1 = (
    ["card2", "card3", "card5", "addr1", "addr2"]
    + [f"V{i}" for i in range(95, 138)]
    + [f"V{i}" for i in range(279, 322)]
)
_IMPUTE_NEG1 = [c for c in _IMPUTE_NEG1 if c in FEATURE_COLUMNS]

# Categorical columns filled with mode in training
_IMPUTE_CAT = ["ProductCD", "card4", "card6", "P_emaildomain"]
_IMPUTE_CAT = [c for c in _IMPUTE_CAT if c in FEATURE_COLUMNS]


def preprocess(transaction: TransactionInput) -> np.ndarray:
    df = pd.DataFrame([transaction.model_dump()])
    df = df[FEATURE_COLUMNS]

    df["TransactionAmt"] = np.log1p(df["TransactionAmt"])

    # Fill categorical defaults (0 = first factorized category)
    df[_IMPUTE_CAT] = df[_IMPUTE_CAT].fillna(0)

    # Fill numeric defaults with -1
    df[_IMPUTE_NEG1] = df[_IMPUTE_NEG1].fillna(-1)

    # Other NaN stays as-is: XGBoost handles it, imputer handles it for RF/LR
    return SCALER.transform(df)
