import pytest
import numpy as np
import sklearn
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from src.ml.data import process_data

def test_train_model(data, categorical_features):
    
    X_train, y_train, encoder, lb = process_data(
            data,
            categorical_features=categorical_features,
            label="salary",
            training=True)
    
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder)
    assert isinstance(lb, sklearn.preprocessing._label.LabelBinarizer)
   

