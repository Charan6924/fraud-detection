import joblib
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
import pandas as pd
import xgboost
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import make_scorer
import time

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}

def fine_tune():
    print("Loading data")
    df = pd.read_parquet('/Users/charan/Documents/fraud-detection/data/features.parquet')
    print(f"  Loaded {len(df)} rows, {len(df.columns)} cols")

    print("Splitting features and target")
    X = df.drop(columns=["isFraud"]).values
    y = df["isFraud"].values
    print(f"  X shape: {X.shape}, y fraud rate: {y.mean():.4%}")

    print("[Train/test split")
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    print("Building pipeline")
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", xgboost.XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
            device="cuda",
        )),
    ])

    print("Setting up grid search")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scorer = make_scorer(average_precision_score)

    total_fits = 72 * 3
    print(f"  {total_fits} total fits (72 params × 3 folds)")
    t1 = time.time()
    search = GridSearchCV(pipeline, param_grid=param_grid, scoring=scorer, cv=cv, n_jobs=-1, verbose=10)

    print("Fitting grid search")
    search.fit(X_train, y_train)
    print(f"  Done in {time.time() - t1:.1f}s")
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV score: {search.best_score_:.4f}")

    print("Evaluating on test set")
    y_prob = search.best_estimator_.predict_proba(X_test)[:, 1]
    y_pred = search.best_estimator_.predict(X_test)

    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"  PR-AUC:     {pr_auc:.4f}")
    print(f"  F1 (fraud): {f1:.4f}")
    print(f"  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"  FNR:        {fn / (fn + tp):.4%}")

    print("Saving model")
    joblib.dump(search.best_estimator_, "models/finetuned_model.joblib")
    print("Done!")


if __name__ == "__main__":
    fine_tune()
