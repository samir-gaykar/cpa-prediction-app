# Third-party imports
from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Local application imports
from boolean_to_string_transformer import BooleanToStringTransformer
from column_selector import ColumnSelector
from data_loader import DataLoader
from drop_columns_transformer import DropColumnsTransformer
from interaction_transformer import InteractionTransformer
from log_transformer import LogTransformer
from standard_scaler_transformer import StandardScalerTransformer
from target_encoder_transformer import TargetEncoderTransformer
from temporal_feature_engineer_transformer import (
    TemporalFeatureEngineerTransformer
)

# List of (estimator, filename) tuples
estimators = [
    ("data_loader", "data_loader.pkl"),
    (
        "temporal_feature_engineer_transformer",
        "temporal_feature_engineer_transformer.pkl"
    ),
    ("drop_columns_transformer", "drop_columns_transformer.pkl"),
    ("log_transformer", "log_transformer.pkl"),
    ("boolean_to_string_transformer", "boolean_to_string_transformer.pkl"),
    ("target_encoder_transformer", "target_encoder_transformer.pkl"),
    ("standard_scaler_transformer", "standard_scaler_transformer.pkl"),
    ("interaction_transformer", "interaction_transformer.pkl"),
    ("column_selector", "column_selector.pkl"),
    ("model", "model.joblib"),
    ("inference_pipeline", "inference_pipeline.pkl")
]

# Load each estimator using a for loop
for var_name, filename in estimators:
  globals()[var_name] = joblib.load(filename)

app = FastAPI()

class Data(BaseModel):
    date: str
    category_id: str
    industry: str
    publisher: str
    market_id: str

@app.get("/")
def read_root():
    return {"message": "CPA Prediction API"}

@app.post('/predict')
def predict(data: Data):
    columns = [
    'date',
    'category_id',
    'industry',
    'publisher',
    'market_id'
]

    data = [[
        data.date,
        data.category_id,
        data.industry,
        data.publisher,
        data.market_id
    ]]

    df = pd.DataFrame(data=data, columns=columns)

    prediction = inference_pipeline.predict(df)
    return round(prediction.tolist()[0], 4)