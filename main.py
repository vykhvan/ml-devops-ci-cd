from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from src.ml.data import process_data
from src.ml.model import inference


app = FastAPI(title="Census Bureau Income Prediction")

model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")

cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
]

class Person(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

@app.get("/")
async def root():
    return {"message": "Greetings!"}


@app.post("/prediction/")
async def get_prediction(person: Person):
    data = { 
            "age": [person.age],
            "workclass": [person.workclass],
            "fnlgt": [person.fnlgt],
            "education": [person.education],
            "education-num": [person.education_num],
            "marital-status": [person.marital_status],
            "occupation": [person.occupation],
            "relationship": [person.relationship],
            "race": [person.race],
            "sex": [person.sex],
            "capital-gain": [person.capital_gain],
            "capital-loss": [person.capital_loss],
            "hours-per-week": [person.hours_per_week],
            "native-country": [person.native_country]
    }

    features = pd.DataFrame.from_dict(data=data)
    X, _, _, _ = process_data(
            X=features,
            categorical_features = cat_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb)
    prediction = inference(model, X)
    salary = lb.inverse_transform(prediction)[0]  
    return {"salary": salary} 

