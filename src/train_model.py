# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import logging
import joblib
from ml.data import process_data
from ml.model import train_model
from ml.model import compute_model_metrics
from ml.model import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()
# Add code to load in the data.

logger.info("Data ingestion step")
data = pd.read_csv("../data/cleaned_data.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info("Data segregation step")
train, test = train_test_split(data, test_size=0.20)

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

logger.info("Data preprocessing step for train set")
X_train, y_train, encoder, lb = process_data(
    train, 
    categorical_features=cat_features, 
    label="salary", 
    training=True
)

# Proces the test data with the process_data function.
logger.info("Data preprocessing step for test set")
X_test, y_test, encoder, lb  = process_data(
    train, 
    categorical_features=cat_features, 
    label="salary", 
    training=False, 
    encoder=encoder,
    lb=lb
)

# Train and save a model.
logger.info("Train model step")
model = train_model(X_train, y_train)

logger.info("Inference step")
preds = inference(model, X_test)

logger.info("Scoring step")
precision, recall, fbeta =  compute_model_metrics(y_test, preds)

logger.info(f"precision: {precision}")
logger.info(f"recall: {recall}")
logger.info(f"fbeta: {fbeta}")

logger.info("Save model")
joblib.dump(model, "../model/model.pkl")

logger.info("Save one-hot encoder")
joblib.dump(encoder, "../model/encoder.pkl")

logger.info("Save label encoder")
joblib.dump(lb, "../model/lb.pkl")
