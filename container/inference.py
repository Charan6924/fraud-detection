import joblib
from preprocess import preprocess
import os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")       
THRESHOLD = 0.10

lr_path = os.path.join(MODEL_DIR, 'logistic_reg_model.joblib')
rf_path = os.path.join(MODEL_DIR, 'random_forest_model.joblib')
xgb_path = os.path.join(MODEL_DIR, 'xgboost_model.joblib')
meta_path = os.path.join(MODEL_DIR, 'meta_model.joblib')
imputer_path = os.path.join(MODEL_DIR, 'imputer.joblib')

lr = joblib.load(lr_path)
rf = joblib.load(rf_path)
xgb = joblib.load(xgb_path)
meta = joblib.load(meta_path)
imputer = joblib.load(imputer_path) 

def inference(inp, lr, rf, xgb, meta, imputer):
    # XGBoost runs on raw features (handles NaN natively)
    xgb_probs = xgb.predict_proba(inp)[0, 1]

    # RF and LR need imputation
    inp_imp = imputer.transform(inp)
    rf_probs = rf.predict_proba(inp_imp)[0, 1]
    lr_probs = lr.predict_proba(inp_imp)[0, 1]
    
    meta_input = np.column_stack((xgb_probs,rf_probs,lr_probs))

    #threshold = 0.10
    prob = meta.predict_proba(meta_input)[0, 1]
    prediction = int(prob >= THRESHOLD)  
    
    return {
        'prediction': prediction,
        'probability': float(prob),
        'probabilities_per_model': {
            'xgboost': float(xgb_probs),
            'random_forest': float(rf_probs),
            'logistic_regression': float(lr_probs),
        },
        'model_version': 'ensemble_v1',
    }
