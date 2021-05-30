from typing import Tuple
from dataclasses import asdict

import pytest
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from ml_classifier.configs.config import ClfParams
from ml_classifier.models.model import (
    build_classifier,
    build_model,
    train_model,
    save_model,
    load_model,
    predict_model,
    evaluate_model
)


@pytest.fixture(scope='session')
def tmp_clf_params(test_params) -> Tuple[ClfParams, ClfParams]:
    sgd_params = test_params.sgd_params
    rf_params = test_params.rf_params
    return sgd_params, rf_params


def test_build_classifier(tmp_clf_params):
    sgd_params, rf_params = tmp_clf_params
    sgd = build_classifier(sgd_params)
    rf = build_classifier(rf_params)
    assert isinstance(sgd, SGDClassifier)
    assert isinstance(rf, RandomForestClassifier)
    assert sgd.alpha == sgd_params.alpha
    assert rf.random_state == rf_params.random_state


def test_train_predict_evaluate_model(
        tmp_clf_params,
        train_dataset,
        test_params
):
    X_train = train_dataset.drop(test_params.target, axis=1)
    y_train = train_dataset[test_params.target].values
    sgd_params, rf_params = tmp_clf_params
    model = train_model(X_train, y_train, sgd_params, test_params.feature_params)
    predicts = model.predict(X_train)
    predicts_from_func = predict_model(model, X_train)
    metrics = asdict(evaluate_model(predicts, y_train))
    assert predicts.shape == y_train.shape
    assert all(predicts == predicts_from_func)
    assert all([metric > 0 for metric in metrics.values()])


def test_save_load_model(tmpdir, tmp_clf_params, test_params):
    path = tmpdir.join('model.pkl')
    sgd_params, rf_params = tmp_clf_params
    model = build_model(sgd_params, test_params.feature_params)
    save_model(path, model)
    model_dump = load_model(path)
    assert model.__repr__() == model_dump.__repr__()
