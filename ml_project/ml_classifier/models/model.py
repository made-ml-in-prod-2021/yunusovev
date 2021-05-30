import pickle
from dataclasses import dataclass
from typing import Dict, Union


from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd

from ml_classifier.features.preprocessing import build_transformer
from ml_classifier.configs.config import ClfParams, FeaturesParams, TrainingPipelineParams, Metrics

SklearnClf = Union[RandomForestClassifier, SGDClassifier]


def build_classifier(clf_params: ClfParams) -> SklearnClf:
    """
    Build sklearn classifier (RF/SGD) by config parameters
    """
    if clf_params.model_type == 'SGDClassifier':
        clf = SGDClassifier(
            loss=clf_params.loss,
            max_iter=clf_params.max_iter,
            alpha=clf_params.alpha,
            random_state=clf_params.random_state,
        )
    elif clf_params.model_type == 'RandomForestClassifier':
        clf = RandomForestClassifier(
            n_estimators=clf_params.n_estimators,
            max_depth=clf_params.max_depth,
            random_state=clf_params.random_state
        )
    else:
        raise NotImplementedError()
    return clf


def build_model(
        clf_params: ClfParams,
        feature_params: FeaturesParams
) -> Pipeline:
    """
    Build model with features preprocessing
    """
    transformer = build_transformer(feature_params)
    clf = build_classifier(clf_params)
    model = Pipeline([
        ('features', transformer),
        ('clf', clf)
        ])
    return model


def extract_target(
        df: pd.DataFrame,
        training_pipeline_params: TrainingPipelineParams
) -> np.ndarray:
    return df[training_pipeline_params.target].values


def train_model(
        df_features: pd.DataFrame,
        y: np.ndarray,
        clf_params: ClfParams,
        features_params: FeaturesParams
) -> Pipeline:
    model = build_model(clf_params, features_params)
    model.fit(df_features, y)
    return model


def predict_model(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(df)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Metrics:
    return Metrics(
        accuracy=accuracy_score(y_true=target, y_pred=predicts),
        precision=precision_score(y_true=target, y_pred=predicts),
        recall=recall_score(y_true=target, y_pred=predicts),
    )


def save_model(path_to_model: str, model: Pipeline) -> None:
    with open(path_to_model, 'wb') as fio:
        pickle.dump(model, fio)


def load_model(path_to_model: str) -> Pipeline:
    with open(path_to_model, 'rb') as fio:
        model = pickle.load(fio)
    return model
