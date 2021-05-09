from typing import Dict, Any

import yaml
from marshmallow_dataclass import class_schema

from src.configs.config import TrainingPipelineParams, PredictParams


def read_config(path_to_config: str) -> Dict[str, Any]:
    with open(path_to_config, 'r') as input_stream:
        config = yaml.safe_load(input_stream)
    return config


def read_training_params(path_to_config: str) -> TrainingPipelineParams:
    TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)
    schema = TrainingPipelineParamsSchema()
    config = read_config(path_to_config)
    training_params = schema.load(config)
    return training_params


def read_predict_params(path_to_config: str) -> PredictParams:
    PredictParamsSchema = class_schema(PredictParams)
    schema = PredictParamsSchema()
    config = read_config(path_to_config)
    predict_params = schema.load(config)
    return predict_params
