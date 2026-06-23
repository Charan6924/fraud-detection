from preprocess import preprocess
from inference import inference, lr, rf, xgb, meta, imputer
from fastapi import FastAPI
from drift import check_drift
import boto3
import os
from schemas import TransactionInput, FeedbackInput
from fastapi import HTTPException
from botocore.exceptions import ClientError
import pandas as pd

app = FastAPI()
_dynamo_table = None

def _get_table():
    global _dynamo_table
    if _dynamo_table is None:
        _dynamo_table = boto3.resource("dynamodb").Table(os.environ["PREDICTIONS_TABLE"])
    return _dynamo_table

def get_production_features():
    table = _get_table()
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
    if production_features.empty:
        return {"drift_detected": False, "drift_share": 0.0, "drifted_features": [], "n_drifted": 0, "n_total": 0}
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


@app.post("/feedback")
def feedback(fb: FeedbackInput):
    table = _get_table()
    try:
        table.update_item(
            Key={"id": fb.prediction_id},
            UpdateExpression="SET #l = :l",
            ExpressionAttributeNames={"#l": "label"},
            ExpressionAttributeValues={":l": fb.label},
        )
        return {"status": "ok", "prediction_id": fb.prediction_id, "label": fb.label}
    except ClientError as e:
        raise HTTPException(status_code=404, detail=f"prediction_id not found: {e}")
