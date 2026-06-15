#Fast API server
from mangum import Mangum
from schemas import TransactionInput
from preprocess import preprocess
from inference import inference,lr,rf,xgb,meta,imputer
from fastapi import FastAPI

app = FastAPI()

@app.get('/health')
def health():
    return {'status':'ok'}

@app.post('/predict')
def predict(transaction: TransactionInput):
    features = preprocess(transaction=transaction)
    result = inference(features,lr,rf,xgb,meta,imputer)
    return result

handler = Mangum(app)
