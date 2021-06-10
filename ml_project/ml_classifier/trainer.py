import sys
import json
from dataclasses import asdict

import click
import logging

from ml_classifier.utils.utils import read_training_params
from ml_classifier.configs.config import TrainingPipelineParams
from ml_classifier.data.dataset import read_data, split_data
from ml_classifier.models.model import (
    save_model,
    evaluate_model,
    extract_target,
    predict_model,
    train_model
)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train(training_params: TrainingPipelineParams) -> None:
    """Run train model"""
    df = read_data(training_params.input_data_path)
    logger.info(f"Loaded dataset")
    train, val = split_data(df, training_params.splitting_params)
    logger.info('train size: %s, val size: %s', train.shape[0], val.shape[0])

    y_train = extract_target(train, training_params)
    y_val = extract_target(val, training_params)

    logger.info('Start training model (%s)', training_params.clf_params)
    model = train_model(
        train,
        y_train,
        training_params.clf_params,
        training_params.feature_params
    )
    logger.info('End training model')
    pred_val = predict_model(model, val)

    save_model(training_params.output_model_path, model)
    logger.info('Saved model to %s', training_params.output_model_path)

    metrics = evaluate_model(y_val, pred_val)
    logger.info('metrics on validate: %s', metrics)

    with open(training_params.metrics_path, 'w') as fio:
        json.dump(asdict(metrics), fio)
    logger.info('Saved metrics to %s', training_params.metrics_path)
    return metrics


@click.command(name="train")
@click.argument('path_to_config')
def train_command(path_to_config: str) -> None:
    logger.info("Start training by config: %s", path_to_config)
    training_params = read_training_params(path_to_config)
    metrics = train(training_params)


if __name__ == '__main__':
    train_command()
