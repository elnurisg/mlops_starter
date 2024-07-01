import os
import sys
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from contextlib import asynccontextmanager

from src.ml_package.models.logistic_regression import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model = None

# Loading the model on startup would help in avoiding the overhead of loading the model for every request
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")

app = FastAPI(lifespan=lifespan)

# Defining the request body corresponding the data
class PredictRequest(BaseModel):
    data: conlist(float, min_length=4, max_length=4)
    
@app.post("/predict") # Model inference is a CPU-bound operation.
def predict(request: PredictRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    prediction = model.predict([request.data])
    return {"prediction": prediction.tolist()}

@app.get("/")
def read_root():
    return {"message": "Welcome to the MLOps starter API!"}