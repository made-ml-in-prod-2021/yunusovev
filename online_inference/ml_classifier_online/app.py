import sys
import logging

from fastapi import FastAPI
from ml_classifier.models.model import load_model

from ml_classifier_online.entities import read_app_params, to_pandas, Sample, Response


CONFIG_PATH = 'configs/app_config.yaml'

app = FastAPI()
depends = {}

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load():
    logger.info('Starting service')
    app_params = read_app_params(CONFIG_PATH)
    depends['app_params'] = app_params
    logger.info(f'loading model: {app_params.path_to_model}')
    depends['model'] = load_model(depends['app_params'].path_to_model)
    logger.info('model loaded')


@app.post("/predict/", response_model=Response)
def predict(request: Sample):
    sample = to_pandas(request)
    result = depends['model'].predict(sample)
    response = Response(predict=result)
    return response
