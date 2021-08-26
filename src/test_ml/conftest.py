import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.ml.data import process_data

@pytest.fixture(scope="session")
def data():
    df = pd.read_csv("~/ml-devops-ci-cd/data/cleaned_data.csv")
    return df

@pytest.fixture(scope="session")
def categorical_features():
    cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
    ]
    return cat_features

@pytest.fixture(scope="session")
def train_test_data(data, categorical_features):

    train, test = train_test_split(data, test_size=0.3)

    X_train, y_train, encoder, lb = process_data(
            train,
            categorical_features=categorical_features,
            label="salary",
            training=True)

    X_test, y_test, encoder, lb = process_data(
            test,
            categorical_features=categorical_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb)
    return X_train, X_test, y_train, y_test
