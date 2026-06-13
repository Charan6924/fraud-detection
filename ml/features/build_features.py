import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


DROP_V = [f"V{i}" for i in range(138, 167)] + [f"V{i}" for i in range(322, 340)]

DROP_D = ["D6", "D7", "D8", "D9", "D11", "D12", "D13", "D14"]

DROP_ID = [
    "id_03", "id_04", "id_07", "id_08", "id_09", "id_10",
    "id_18", "id_21", "id_22", "id_23", "id_24", "id_25",
    "id_26", "id_27", "id_30", "id_32", "id_33", "id_34",
]

DROP_META = ["TransactionID", "TransactionDT"]

KEEP_NAN_V = (
    [f"V{i}" for i in range(1, 12)]
    + [f"V{i}" for i in range(12, 35)]
    + [f"V{i}" for i in range(35, 53)]
    + [f"V{i}" for i in range(53, 95)]
    + [f"V{i}" for i in range(167, 279)]
)

KEEP_NAN_D = ["D1", "D2", "D3", "D4", "D5", "D10", "D15"]

KEEP_NAN_ID = [
    "id_01", "id_02", "id_05", "id_06", "id_11", "id_12",
    "id_13", "id_14", "id_15", "id_16", "id_17", "id_19",
    "id_20", "id_28", "id_29", "id_31", "id_35", "id_36",
    "id_37", "id_38",
]

KEEP_NAN_M = [f"M{i}" for i in range(1, 10)]

KEEP_NAN_OTHER = ["dist1"]

IMPUTE_V = [f"V{i}" for i in range(95, 138)] + [f"V{i}" for i in range(279, 322)]

IMPUTE_NEG1 = ["card2", "card3", "card5", "addr1", "addr2"] + IMPUTE_V

IMPUTE_CAT = {
    "ProductCD": "W",
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "Unknown",
}


BUILD_FLAGS = ["has_identity"]

ALL_DROP = set(DROP_V + DROP_D + DROP_ID + DROP_META)

ALL_KEEP_NAN = set(
    KEEP_NAN_V + KEEP_NAN_D + KEEP_NAN_ID + KEEP_NAN_M + KEEP_NAN_OTHER
)

ALL_IMPUTE_NEG1 = IMPUTE_NEG1

def build_dataset():
    txn = pd.read_csv('/Users/charan/Documents/fraud-detection/data/dataset/train_transaction.csv')
    id = pd.read_csv('/Users/charan/Documents/fraud-detection/data/dataset/train_identity.csv')

    merged = pd.merge(txn,id,how='left',on='TransactionID')

    merged["has_identity"] = merged["id_01"].notna().astype(int)

    merged = merged.drop(columns=[c for c in ALL_DROP if c in merged.columns])

    merged["TransactionAmt"] = np.log1p(merged["TransactionAmt"])

    for col, val in IMPUTE_CAT.items():
        merged[col] = merged[col].fillna(val)

    for col in ALL_IMPUTE_NEG1:
        if col in merged.columns:
            merged[col] = merged[col].fillna(-1)

    # Encode all remaining string columns to numeric
    for col in merged.select_dtypes(include="object").columns:
        merged[col] = pd.factorize(merged[col])[0]

    feature_cols = [c for c in merged.columns if c != "isFraud"]
    numeric_cols = merged[feature_cols].select_dtypes(include="number").columns.tolist()
    scaler = StandardScaler()
    merged[numeric_cols] = scaler.fit_transform(merged[numeric_cols])

    merged.to_parquet("data/features.parquet", index=False)




if __name__ == "__main__":
    build_dataset()
    print('Done')
