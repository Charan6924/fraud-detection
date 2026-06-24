"""Ensemble training pipeline for Fargate: downloads features from S3, trains, uploads artifacts."""

import os
import joblib
import pandas as pd
import numpy as np
import boto3
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
import xgboost

s3 = boto3.client("s3")
BUCKET = os.environ["ARTIFACTS_BUCKET"]


def train():
    s3.download_file(BUCKET, "features.parquet", "/tmp/features.parquet")
    df = pd.read_parquet("/tmp/features.parquet")
    X = df.drop(columns="isFraud")
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, 
                                                        random_state=42, stratify=y)
    
    xgb_model = xgboost.XGBClassifier(
          n_estimators=500, max_depth=12, learning_rate=0.2,
          subsample=0.8, colsample_bytree=1.0, min_child_weight=1,
          tree_method="hist", device="cpu", random_state=42,
      )
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model.set_params(scale_pos_weight=scale)

    rf_model = RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )

    lr_model = LogisticRegression(
        max_iter=1000, class_weight="balanced", C=0.1, random_state=42,
    )

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_xgb = np.zeros((X_train.shape[0], 1))
    oof_lr = np.zeros((X_train.shape[0], 1))
    oof_rf = np.zeros((X_train.shape[0], 1))

    for _ , (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        X_tr_imp, X_val_imp = X_train_imp[train_idx], X_train_imp[val_idx]
        y_tr = y_train.iloc[train_idx]

        xgb_model.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
        rf_model.fit(X_tr_imp, y_tr)
        oof_rf[val_idx] = rf_model.predict_proba(X_val_imp)[:, 1].reshape(-1, 1)
        lr_model.fit(X_tr_imp, y_tr)
        oof_lr[val_idx] = lr_model.predict_proba(X_val_imp)[:, 1].reshape(-1, 1)

    meta.fit(np.column_stack([oof_xgb, oof_rf, oof_lr]), y_train)
    test_preds = np.column_stack([
          xgb_model.predict_proba(X_test)[:, 1],
          rf_model.predict_proba(X_test_imp)[:, 1],
          lr_model.predict_proba(X_test_imp)[:, 1],
      ])
    y_prob = meta.predict_proba(test_preds)[:, 1]
    y_pred = meta.predict(test_preds)

    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fnr = fn / (fn + tp)

    print(f"PR-AUC: {pr_auc:.4f}  F1: {f1:.4f}  FNR: {fnr:.4%}")

    # Save artifacts
    os.makedirs("/tmp/models", exist_ok=True)
    joblib.dump(meta, "/tmp/models/meta_model.joblib")
    joblib.dump(xgb_model, "/tmp/models/xgboost_model.joblib")
    joblib.dump(rf_model, "/tmp/models/random_forest_model.joblib")
    joblib.dump(lr_model, "/tmp/models/logistic_reg_model.joblib")
    joblib.dump(imputer, "/tmp/models/imputer.joblib")

    # Upload to S3
    for f in os.listdir("/tmp/models"):
        s3.upload_file(f"/tmp/models/{f}", BUCKET, f"models/{f}")

    # Generate & upload reference dataset
    X_train_ref = pd.DataFrame(X_train)
    X_train_ref.to_parquet("/tmp/reference.parquet", index=False)
    s3.upload_file("/tmp/reference.parquet", BUCKET, "reference.parquet")

    print("Done — artifacts uploaded to s3://%s/models/" % BUCKET)


if __name__ == "__main__":
    train()
