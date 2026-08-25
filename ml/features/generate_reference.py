import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def generate(input_path: str | Path, output_path: str | Path) -> None:
    df = pd.read_parquet(input_path)
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(output_path, index=False)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate the drift reference dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "features.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "container" / "reference.parquet",
    )
    args = parser.parse_args()
    generate(args.input, args.output)
