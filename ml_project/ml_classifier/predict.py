import sys
import logging

import pandas as pd
import click

from ml_classifier.models.model import load_model, predict_model
from ml_classifier.data.dataset import read_data, save_data
from ml_classifier.utils.utils import read_predict_params


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@click.command(name="predict")
@click.argument("path_to_config")
def predict(path_to_config: str) -> None:
    logger.info(f'Start predicting by config: {path_to_config}')
    params = read_predict_params(path_to_config)
    data = read_data(params.input_data_path)
    logger.info(f"Loaded dataset")
    model = load_model(params.model_path)
    logger.info(f"Loaded model")
    predicts = predict_model(model, data)
    predicts = pd.DataFrame(predicts, columns=['predict'])
    save_data(predicts, params.output_data_path)
    logger.info(f"Saved predict to {params.output_data_path}")


if __name__ == '__main__':
    predict()