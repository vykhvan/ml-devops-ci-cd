import pytest
import sklearn
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from src.ml.data import process_data
from src.ml.model import (train_model, 
        compute_model_metrics, inference)

    
def test_train_model(train_test_data):
    X_train, X_test, y_train, y_test = [*train_test_data] 
    model = train_model(X_train, y_train)
    assert isinstance(model, sklearn.ensemble._gb.GradientBoostingClassifier)
    joblib.dump(model, "test_model.pkl")

def test_inference(train_test_data):
    model = joblib.load("test_model.pkl")
    X_train, X_test, y_train, y_test = [*train_test_data] 
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)


def test_compute_model_metrics(train_test_data):
    X_train, X_test, y_train, y_test = [*train_test_data] 
    model = joblib.load("test_model.pkl")
    preds = inference(model, X_test)  
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

