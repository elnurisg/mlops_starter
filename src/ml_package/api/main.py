import os
import sys
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from src.ml_package.models.logistic_regression import load_model

# Add the src directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

app = FastAPI()

# Load the model at startup
@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")

# Define the request body model
class PredictRequest(BaseModel):
    data: conlist(float, min_length=4, max_length=4)
    
@app.post("/predict")
async def predict(request: PredictRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    prediction = model.predict([request.data])
    return {"prediction": prediction.tolist()}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the MLOps starter API!"}