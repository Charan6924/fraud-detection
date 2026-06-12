from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import xgboost
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    df = pd.read_parquet('/Users/charan/Documents/fraud-detection/data/features.parquet')

    
    X = df.drop(columns=["isFraud"]).values
    y = df["isFraud"].values

    X_train,y_train,X_test,y_test = train_test_split(X,y,test_size = 0.2,stratify=y, random_state=42)

    scale = (y_train == 0).sum() / (y_train == 1).sum() 
    model = xgboost.XGBClassifier(n_estimators = 200,max_depth = 5, learning_rate = 0.05, scale_pos_weight = scale, eval_metric = 'logloss')

    smote = SMOTE(random_state = 42)
    X_train_res, y_train_res = smote.fit_resample(X_train,y_train) #type: ignore

    print('starting training')
    model.fit(X_train_res,y_train_res)

    print('finished training')
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    pr_auc = average_precision_score(y_test,y_prob)
    f1 = f1_score(y_test,y_pred)
    tn,fp,fn,tp = confusion_matrix(y_test,y_pred).ravel()

    print(f"pr_auc : {pr_auc:.4f}")
    print(f"f1 : {f1:.4f}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    print(f"fnr: {fn / (fn + tp):.4%}")

    joblib.dump(model, "models/model.joblib")

if __name__ == "__main__":
    train_model()
    print('Done training')


