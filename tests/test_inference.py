import json
from fastapi.testclient import TestClient
from src.ml_package.api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the MLOps starter API!"}

def test_inference_correct_input():
    payload = {
        "data": [5.1, 3.5, 1.4, 0.2]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert isinstance(result["prediction"], int)

def test_inference_incorrect_input():
    payload = {
        "data": [5.1, 3.5, 1.4]  # Incorrect input size
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_inference_empty_input():
    payload = {
        "data": []
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
