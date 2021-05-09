from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from faker import Faker

from ml_classifier.configs.config import ClfParams, TrainingPipelineParams, FeaturesParams
from ml_classifier.models.model import (
    build_classifier,
    build_model,
    train_model,
    save_model,
    load_model,
    predict_model,
    evaluate_model
)

faker = Faker()
DATASET_SIZE = 100


@pytest.fixture()
def tmp_clf_params() -> Tuple[ClfParams, ClfParams]:
    sgd_params = ClfParams(model_type='SGDClassifier', random_state=218, alpha=0.001)
    rf_params = ClfParams(model_type='RandomForestClassifier', random_state=313, n_estimators=20)
    return sgd_params, rf_params


@pytest.fixture()
def tmp_training_pipeline_params() -> TrainingPipelineParams:
    training_params = TrainingPipelineParams(target='target')
    return training_params


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
    })
    tmp_y = np.asarray([faker.random.randint(0, 1) for _ in range(DATASET_SIZE)])
    return tmp_train, tmp_y


def test_build_classifier(tmp_clf_params):
    sgd_params, rf_params = tmp_clf_params
    sgd = build_classifier(sgd_params)
    rf = build_classifier(rf_params)
    assert isinstance(sgd, SGDClassifier)
    assert isinstance(rf, RandomForestClassifier)
    assert sgd.alpha == sgd_params.alpha
    assert rf.random_state == rf_params.random_state


def test_train_predict_evaluate_model(
        tmp_dataset,
        tmp_feature_params,
        tmp_clf_params
):
    tmp_train, tmp_y = tmp_dataset
    sgd_params, rf_params = tmp_clf_params
    model = train_model(tmp_train, tmp_y, sgd_params, tmp_feature_params)
    predicts = model.predict(tmp_train)
    predicts_from_func = predict_model(model, tmp_train)
    metrics = evaluate_model(predicts, tmp_y)
    assert predicts.shape == tmp_y.shape
    assert all(predicts == predicts_from_func)
    assert all([metric > 0 for metric in metrics.values()])



def test_save_load_model(tmpdir, tmp_clf_params, tmp_feature_params):
    path = tmpdir.join('model.pkl')
    sgd_params, rf_params = tmp_clf_params
    model = build_model(sgd_params, tmp_feature_params)
    save_model(path, model)
    model_dump = load_model(path)
    assert model.__repr__() == model_dump.__repr__()
