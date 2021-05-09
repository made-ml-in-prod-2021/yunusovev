from typing import Tuple

import pandas as pd
import pytest
from faker import Faker

from ml_classifier.configs.config import FeaturesParams
from ml_classifier.features.preprocessing import (
    build_bin_pipeline,
    build_cat_pipeline,
    build_num_pipeline,
    build_column_selector,
    build_transformer
)

faker = Faker()

DATASET_SIZE = 100


@pytest.fixture()
def tmp_feature_params() -> FeaturesParams:
    params = FeaturesParams(
        bin_features=['bin_field_1', 'bin_field_2'],
        cat_features=['cat_field_1', 'cat_field_2'],
        num_features=['num_field_1', 'num_field_2']
    )
    return params


@pytest.fixture()
def tmp_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    tmp_train = pd.DataFrame({
        'bin_field_1': [faker.random.randint(0, 1) for _ in range(DATASET_SIZE)],
        'bin_field_2': [faker.random.randint(0, 1) for _ in range(DATASET_SIZE)],
        'cat_field_1': [faker.random.randint(0, 4) for _ in range(DATASET_SIZE)],
        'cat_field_2': [faker.random.randint(0, 4) for _ in range(DATASET_SIZE)],
        'num_field_1': [faker.random.randint(0, 99) for _ in range(DATASET_SIZE)],
        'num_field_2': [faker.random.randint(0, 99) for _ in range(DATASET_SIZE)],
        'target': [faker.random.randint(0, 1) for _ in range(DATASET_SIZE)]
    })
    tmp_val = pd.DataFrame({
        'bin_field_1': [faker.random.randint(0, 1) for _ in range(DATASET_SIZE)],
        'bin_field_2': [faker.random.randint(0, 1) for _ in range(DATASET_SIZE)],
        'cat_field_1': [faker.random.randint(0, 10) for _ in range(DATASET_SIZE)],
        'cat_field_2': [faker.random.randint(0, 10) for _ in range(DATASET_SIZE)],
        'num_field_1': [faker.random.randint(0, 99) for _ in range(DATASET_SIZE)],
        'num_field_2': [faker.random.randint(0, 99) for _ in range(DATASET_SIZE)],

    })
    return tmp_train, tmp_val


def test_build_cat_pipeline(tmp_dataset, tmp_feature_params):
    cat_pipeline = build_cat_pipeline()
    tmp_train, tmp_val = tmp_dataset
    tmp_train = tmp_train[tmp_feature_params.cat_features]
    tmp_val = tmp_val[tmp_feature_params.cat_features]
    cat_pipeline.fit(tmp_train)
    assert cat_pipeline.transform(tmp_train).shape[0] == tmp_train.shape[0]
    assert cat_pipeline.transform(tmp_val).shape[0] == tmp_val.shape[0]
    assert cat_pipeline.transform(tmp_train).shape[1] == cat_pipeline.transform(tmp_val).shape[1]


def test_build_num_pipeline(tmp_dataset, tmp_feature_params):
    num_pipeline = build_num_pipeline()
    tmp_train, tmp_val = tmp_dataset
    tmp_train = tmp_train[tmp_feature_params.num_features]
    tmp_val = tmp_val[tmp_feature_params.num_features]
    num_pipeline.fit(tmp_train)
    assert num_pipeline.transform(tmp_train).shape[0] == tmp_train.shape[0]
    assert num_pipeline.transform(tmp_val).shape[0] == tmp_val.shape[0]
    assert num_pipeline.transform(tmp_train).shape[1] == num_pipeline.transform(tmp_val).shape[1]


def test_build_bin_pipeline(tmp_dataset, tmp_feature_params):
    bin_pipeline = build_bin_pipeline()
    tmp_train, tmp_val = tmp_dataset
    tmp_train = tmp_train[tmp_feature_params.bin_features]
    tmp_val = tmp_val[tmp_feature_params.bin_features]
    bin_pipeline.fit(tmp_train)
    assert bin_pipeline.transform(tmp_train).shape[0] == tmp_train.shape[0]
    assert bin_pipeline.transform(tmp_val).shape[0] == tmp_val.shape[0]
    assert bin_pipeline.transform(tmp_train).shape[1] == bin_pipeline.transform(tmp_val).shape[1]


def test_columns_selector(tmp_dataset, tmp_feature_params):
    column_selector = build_column_selector(tmp_feature_params)
    tmp_train, tmp_val = tmp_dataset
    column_selector.fit(tmp_train)
    assert column_selector.transform(tmp_train).shape[0] == tmp_train.shape[0]
    assert column_selector.transform(tmp_val).shape[0] == tmp_val.shape[0]
    assert min(column_selector.transform(tmp_train).columns == column_selector.transform(tmp_val).columns)


def test_build_transformer(tmp_dataset, tmp_feature_params):
    transformer = build_transformer(tmp_feature_params)
    tmp_train, tmp_val = tmp_dataset
    transformer.fit(tmp_train)
    assert transformer.transform(tmp_train).shape[0] == tmp_train.shape[0]
    assert transformer.transform(tmp_val).shape[0] == tmp_val.shape[0]
    assert transformer.transform(tmp_train).shape[1] == transformer.transform(tmp_val).shape[1]
