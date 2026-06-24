"""Train XGBoost + Random Forest + Logistic Regression ensemble with meta-model."""

import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import joblib
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from sklearn.impute import SimpleImputer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def ensemble_modelling():
    df = pd.read_parquet(os.path.join(PROJECT_ROOT, "data/features.parquet"))
    X = df.drop(columns="isFraud")
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    xgboost_model = xgboost.XGBClassifier(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=1,
        tree_method="hist",
        device="cuda",
        random_state=42,
    )
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    xgboost_model.set_params(scale_pos_weight=scale)

    random_forest = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    logistic_regression = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.1,
        random_state=42,
    )

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_xgb = np.zeros((X_train.shape[0], 1))
    oof_lr = np.zeros((X_train.shape[0], 1))
    oof_rf = np.zeros((X_train.shape[0], 1))

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        print(f"Fold : {fold + 1}")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        X_tr_imp, X_val_imp = X_train_imp[train_idx], X_train_imp[val_idx]
        y_tr, _ = y_train.iloc[train_idx], y_train.iloc[val_idx]

        xgboost_model.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgboost_model.predict_proba(X_val)[:, 1].reshape(-1, 1)

        random_forest.fit(X_tr_imp, y_tr)
        oof_rf[val_idx] = random_forest.predict_proba(X_val_imp)[:, 1].reshape(-1, 1)

        logistic_regression.fit(X_tr_imp, y_tr)
        oof_lr[val_idx] = logistic_regression.predict_proba(X_val_imp)[:, 1].reshape(
            -1, 1
        )

    stacked = np.column_stack([oof_xgb, oof_rf, oof_lr])

    meta.fit(stacked, y_train)

    test_preds = np.column_stack(
        [
            xgboost_model.predict_proba(X_test)[:, 1],
            random_forest.predict_proba(X_test_imp)[:, 1],
            logistic_regression.predict_proba(X_test_imp)[:, 1],
        ]
    )
    y_prob = meta.predict_proba(test_preds)[:, 1]
    y_pred = meta.predict(test_preds)

    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"  PR-AUC:     {pr_auc:.4f}")
    print(f"  F1 (fraud): {f1:.4f}")
    print(f"  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"  FNR:        {fn / (fn + tp):.4%}")

    print("Saving model")
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(meta, os.path.join(model_dir, "meta_model.joblib"))
    joblib.dump(xgboost_model, os.path.join(model_dir, "xgboost_model.joblib"))
    joblib.dump(random_forest, os.path.join(model_dir, "random_forest_model.joblib"))
    joblib.dump(logistic_regression, os.path.join(model_dir, "logistic_reg_model.joblib"))
    print("Done!")


if __name__ == "__main__":
    ensemble_modelling()
