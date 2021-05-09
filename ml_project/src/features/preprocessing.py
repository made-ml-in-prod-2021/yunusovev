from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator

from src.configs.config import FeaturesParams


def build_num_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scaler', StandardScaler())
        ]
    )
    return num_pipeline


def build_cat_pipeline() -> Pipeline:
    cat_pipeline = Pipeline(
        [
            ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore')),
        ]
    )
    return cat_pipeline


def build_bin_pipeline() -> Pipeline:
    bin_pipeline = Pipeline(
        [
            ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ]
    )
    return bin_pipeline


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Applies transformers to columns of pandas DataFrame.

    This estimator allows to extract a fixed set of columns from a DataFrame
    """
    def __init__(self, features):
        self.features = features

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x[self.features]


def build_column_selector(feature_params: FeaturesParams) -> ColumnSelector:
    features = []
    for feature_list in asdict(feature_params).values():
        if feature_list is not None:
            features.extend(feature_list)
    column_selector = ColumnSelector(features)
    return column_selector


def build_transformer(feature_params: FeaturesParams) -> ColumnTransformer:
    column_transformer = ColumnTransformer(
        [
            (
                'categorical_pipeline',
                build_cat_pipeline(),
                feature_params.cat_features,
            ),
            (
                'numerical_pipeline',
                build_num_pipeline(),
                feature_params.num_features,
            ),
            (
                'bin_pipeline',
                build_bin_pipeline(),
                feature_params.bin_features
            ),
        ]
    )
    transformer = Pipeline([
        ('selector', build_column_selector(feature_params)),
        ('column_transformer', column_transformer)
    ])
    return transformer
