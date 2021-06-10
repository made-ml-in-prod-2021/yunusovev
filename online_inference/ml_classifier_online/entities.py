from typing import Any, Dict

import yaml
import pandas as pd
from pydantic import BaseModel


class AppParams(BaseModel):
    path_to_model: str


class RequestsParams(BaseModel):
    path_to_data: str
    host: str = '0.0.0.0'
    port: int = 8000


def read_config(path_to_config: str) -> Dict[str, Any]:
    with open(path_to_config, 'r') as input_stream:
        config = yaml.safe_load(input_stream)
    return config


def read_app_params(path_to_config: str) -> AppParams:
    config = read_config(path_to_config)
    return AppParams(**config)


def read_requests_params(path_to_config: str) -> RequestsParams:
    config = read_config(path_to_config)
    return RequestsParams(**config)


class Sample(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class Response(BaseModel):
    predict: int


def to_pandas(sample: Sample) -> pd.DataFrame:
    return pd.DataFrame([sample.dict()])
