import os

import pytest
from faker import Faker
import pandas as pd

from ml_classifier.data.dataset import save_data
from ml_classifier.trainer import train
from ml_classifier.configs.config import (
    TrainingPipelineParams,
    ClfParams,
    FeaturesParams,
    SplittingParams,
)

faker = Faker()

DATASET_SIZE = 100
TARGET = 'target'
INPUT_DATA_PATH = 'tmp_data.csv'
OUTPUT_MODEL_PATH = 'tmp_model.pkl'
METRICS_PATH = 'tmp_metrics.json'


@pytest.fixture()
def tmp_feature_params() -> FeaturesParams:
    tmp_params = FeaturesParams(
        bin_features=['bin_field_1', 'bin_field_2'],
        cat_features=['cat_field_1', 'cat_field_2'],
        num_features=['num_field_1', 'num_field_2']
    )
    return tmp_params


@pytest.fixture()
def tmp_dataset() -> pd.DataFrame:
    tmp_data = pd.DataFrame({
        'bin_field_1': [faker.random.randint(0, 1) for _ in range(DATASET_SIZE)],
        'bin_field_2': [faker.random.randint(0, 1) for _ in range(DATASET_SIZE)],
        'cat_field_1': [faker.random.randint(0, 4) for _ in range(DATASET_SIZE)],
        'cat_field_2': [faker.random.randint(0, 4) for _ in range(DATASET_SIZE)],
        'num_field_1': [faker.random.randint(0, 99) for _ in range(DATASET_SIZE)],
        'num_field_2': [faker.random.randint(0, 99) for _ in range(DATASET_SIZE)],
        TARGET: [faker.random.randint(0, 1) for _ in range(DATASET_SIZE)]
    })
    return tmp_data


@pytest.fixture()
def tmp_training_pipeline_params(tmpdir, tmp_feature_params):
    tmp_train_params = TrainingPipelineParams(
        input_data_path=tmpdir.join(INPUT_DATA_PATH),
        output_model_path=tmpdir.join(OUTPUT_MODEL_PATH),
        metrics_path=tmpdir.join(METRICS_PATH),
        splitting_params=SplittingParams(),
        clf_params=ClfParams(),
        feature_params=tmp_feature_params,
        target=TARGET
    )
    return tmp_train_params


def test_train_end2end(tmpdir, tmp_training_pipeline_params, tmp_dataset):
    save_data(tmp_dataset, tmp_training_pipeline_params.input_data_path)
    metrics = train(tmp_training_pipeline_params)
    assert os.path.exists(tmp_training_pipeline_params.output_model_path)
    assert os.path.exists(tmp_training_pipeline_params.metrics_path)
    assert metrics['accuracy'] > 0
