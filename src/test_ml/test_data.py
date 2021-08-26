import pytest
import numpy as np
import sklearn
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from src.ml.data import process_data

def test_process_data(data, categorical_features):
    """Test process_data"""    
    X, y, encoder, lb = process_data(
            data,
            categorical_features=categorical_features,
            label="salary",
            training=True)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(encoder, sklearn.preprocessing._encoders.OneHotEncoder)
    assert isinstance(lb, sklearn.preprocessing._label.LabelBinarizer)
