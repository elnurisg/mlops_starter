from sklearn.linear_model import LogisticRegression
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'logistic_regression_model.pkl')

def create_model():
    return LogisticRegression()

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None