import sklearn
import numpy as np
import joblib
import os
from src.ml.model import (train_model,
                          compute_model_metrics, inference)


def test_train_model(processed_data):
    X_train = processed_data["X_train"]
    y_train = processed_data["y_train"]
    model = train_model(X_train, y_train)
    assert isinstance(model, sklearn.ensemble._gb.GradientBoostingClassifier)
    joblib.dump(model, "test_model.pkl")


def test_inference(processed_data):
    X_test = processed_data["X_test"]
    model = joblib.load("test_model.pkl")
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)


def test_compute_model_metrics(processed_data):
    X_test = processed_data["X_test"]
    y_test = processed_data["y_test"]
    model = joblib.load("test_model.pkl")
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    os.remove("test_model.pkl")
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
