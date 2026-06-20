from schemas import TransactionInput
from preprocess import preprocess
from inference import inference, lr, rf, xgb, meta, imputer
from fastapi import FastAPI
from drift import check_drift
import boto3
import os
import pandas as pd

app = FastAPI()


def get_production_features():                                                                                                          
      table = boto3.resource("dynamodb").Table(os.environ["PREDICTIONS_TABLE"])                                                           
      response = table.scan(Limit=500)                                                                                                    
      items = response.get("Items", [])                                                                                                                                                                                                                                                                                                                   
      features = [item["input"] for item in items if "input" in item]                                                                     
      return pd.DataFrame(features)   


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: TransactionInput):
    features = preprocess(transaction=transaction)
    result = inference(features, lr, rf, xgb, meta, imputer)
    return result


@app.post("/drift")
def drift():
    production_features = get_production_features()
    result = check_drift(production_features)
    drift_share = result["metrics"][0]["result"]["drift_share"]
    drift_by_col = result["metrics"][0]["result"]["drift_by_columns"]
    drifted = [col for col, v in drift_by_col.items() if v["drift_detected"]]
    return {
        "drift_detected": drift_share > 0.3,
        "drift_share": round(drift_share, 4),
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "n_total": len(drift_by_col),
    }
