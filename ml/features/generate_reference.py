import pandas as pd
from sklearn.model_selection import train_test_split


def generate():
    df = pd.read_parquet('/Users/charan/Documents/fraud-detection/data/features.parquet')
    X = df.drop(columns=["isFraud"]).values
    y = df["isFraud"].values

    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    out = "/Users/charan/Documents/fraud-detection/container/reference.parquet"
    pd.DataFrame(X_train).to_parquet(out)


if __name__ == "__main__":
    generate()
