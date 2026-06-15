import xgboost
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
from sklearn.impute import SimpleImputer
import os
import joblib


def tune_meta():
    df = pd.read_parquet('/home/cxv166/fraud-detection/data/features.parquet')
    X = df.drop(columns='isFraud')
    y = df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    xgboost_model = xgboost.XGBClassifier(
        n_estimators=500, max_depth=12, learning_rate=0.2,
        subsample=0.8, colsample_bytree=1.0, min_child_weight=1,
        tree_method="hist", device="cuda",
        random_state=42)
    scale = (y_train == 0).sum() / (y_train == 1).sum()
    xgboost_model.set_params(scale_pos_weight=scale)

    random_forest = RandomForestClassifier(
        n_estimators=500, max_depth=15,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    imputer = SimpleImputer(strategy="median")

    oof_xgb = np.zeros((X_train.shape[0], 1))
    oof_rf = np.zeros((X_train.shape[0], 1))
    oof_lr = np.zeros((X_train.shape[0], 1))

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        print(f'OOF Fold : {fold + 1}')
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        X_tr_imp, X_val_imp = X_train_imp[train_idx], X_train_imp[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        xgboost_model.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgboost_model.predict_proba(X_val)[:, 1].reshape(-1, 1)

        random_forest.fit(X_tr_imp, y_tr)
        oof_rf[val_idx] = random_forest.predict_proba(X_val_imp)[:, 1].reshape(-1, 1)

        lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1, random_state=42)
        lr.fit(X_tr_imp, y_tr)
        oof_lr[val_idx] = lr.predict_proba(X_val_imp)[:, 1].reshape(-1, 1)

    stacked = np.column_stack([oof_xgb, oof_rf, oof_lr])

    print("\nGrid searching meta-model...")
    meta_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }
    meta_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        meta_grid, cv=5, scoring="average_precision", verbose=1
    )
    meta_search.fit(stacked, y_train)
    print(f"Best meta params: {meta_search.best_params_}")
    print(f"Best CV score:    {meta_search.best_score_:.4f}")

    xgboost_model.fit(X_train, y_train)
    random_forest.fit(X_train_imp, y_train)
    lr_full = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1, random_state=42)
    lr_full.fit(X_train_imp, y_train)

    test_preds = np.column_stack([
        xgboost_model.predict_proba(X_test)[:, 1],
        random_forest.predict_proba(X_test_imp)[:, 1],
        lr_full.predict_proba(X_test_imp)[:, 1],
    ])
    y_prob = meta_search.best_estimator_.predict_proba(test_preds)[:, 1]

    print(f"{'thresh':>6} | {'F1':>5} | {'FNR':>6} | {'FP':>4} | {'FN':>4} | {'TP':>4} | {'cost':>6}")
    print("-" * 55)

    cost_fraud, cost_review = 150, 5
    results = []
    for thresh in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fnr = fn / (fn + tp)
        f1 = f1_score(y_test, y_pred)
        cost = fn * cost_fraud + fp * cost_review
        results.append({"threshold": thresh, "cost": cost, "fnr": fnr, "f1": f1, "fp": fp, "fn": fn, "tp": tp})
        print(f"{thresh:>5.2f}  | {f1:.4f} | {fnr:.2%} | {fp:>4} | {fn:>4} | {tp:>4} | {cost:>5}")

    candidates = [r for r in results if r['fp'] <= 900]
    best = min(candidates, key=lambda r: r['cost'])
    print(f"\nBest (cost): threshold={best['threshold']:.2f}, FNR={best['fnr']:.2%}, F1={best['f1']:.4f}, FP={best['fp']}")

    print("\nSaving tuned ensemble...")                                                                                                 
    model_dir = '/home/cxv166/fraud-detection/models'                                                                                   
    os.makedirs(model_dir, exist_ok=True)                                                                                               
    joblib.dump(meta_search.best_estimator_, os.path.join(model_dir, "meta_model.joblib"))                                              
    joblib.dump(xgboost_model, os.path.join(model_dir, "xgboost_model.joblib"))                                                         
    joblib.dump(random_forest, os.path.join(model_dir, "random_forest_model.joblib"))                                                   
    joblib.dump(lr_full, os.path.join(model_dir, "linear_reg_model.joblib"))                                                            
    print(f"Saved. Recommended threshold = {best['threshold']:.2f}")   
if __name__ == '__main__':
    tune_meta()


'''
Best meta params: {'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear'}
Best CV score:    0.7983
thresh |    F1 |    FNR |   FP |   FN |   TP |   cost
-------------------------------------------------------
 0.05  | 0.7799 | 20.20% | 1026 |  835 | 3298 | 130380
 0.10  | 0.8068 | 22.70% |  592 |  938 | 3195 | 143660
 0.15  | 0.8107 | 23.93% |  479 |  989 | 3144 | 150745
 0.20  | 0.8111 | 24.82% |  421 | 1026 | 3107 | 156005
 0.25  | 0.8132 | 25.21% |  378 | 1042 | 3091 | 158190
 0.30  | 0.8131 | 25.62% |  354 | 1059 | 3074 | 160620
 0.35  | 0.8125 | 26.20% |  325 | 1083 | 3050 | 164075
 0.40  | 0.8123 | 26.57% |  305 | 1098 | 3035 | 166225
 0.45  | 0.8122 | 27.05% |  276 | 1118 | 3015 | 169080
 0.50  | 0.8101 | 27.58% |  263 | 1140 | 2993 | 172315
 0.55  | 0.8096 | 27.99% |  243 | 1157 | 2976 | 174765
 0.60  | 0.8083 | 28.45% |  227 | 1176 | 2957 | 177535
 0.65  | 0.8068 | 28.89% |  214 | 1194 | 2939 | 180170
 0.70  | 0.8063 | 29.35% |  190 | 1213 | 2920 | 182900
 0.75  | 0.8025 | 30.10% |  178 | 1244 | 2889 | 187490
 0.80  | 0.7937 | 31.67% |  159 | 1309 | 2824 | 197145
 0.85  | 0.7870 | 33.08% |  130 | 1367 | 2766 | 205700
 0.90  | 0.7653 | 36.46% |  104 | 1507 | 2626 | 226570

Best (cost): threshold=0.10, FNR=22.70%, F1=0.8068, FP=592
'''
