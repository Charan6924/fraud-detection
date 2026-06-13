import os
import pandas as pd
import xgboost

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def get_features():
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/features.parquet"))
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    model = xgboost.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        importance_type="gain", eval_metric="logloss",
        tree_method="hist", device="cuda",
        random_state=42,
    )
    model.fit(X, y)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return importance

if __name__ == "__main__":
    imp = get_features()
    imp.to_csv(os.path.join(PROJECT_ROOT, "data/feature_importances.csv"), index=False)
    print(imp.head(20))
    print(f"\nFeatures with zero importance: {(imp['importance'] == 0).sum()}")
