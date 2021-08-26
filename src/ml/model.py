from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    """ 
    n_estimators = [100, 200, 300]
    max_depth = [3, 4, 5]
    param_grid = {"n_estimators": n_estimators, "max_depth": max_depth}
    estimator = GradientBoostingClassifier()
    search = GridSearchCV(estimator=estimator, 
                          param_grid=param_grid,
                          n_jobs = -1,
                          cv=5, 
                          verbose=3)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    """
    model = GradientBoostingClassifier(n_estimators=250, max_depth=4)
    model.fit(X_train, y_train)

    return model 


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X) 
    return preds
