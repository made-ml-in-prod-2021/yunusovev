from typing import Dict

import pytest
import pandas as pd
from faker import Faker
from ml_classifier.utils.utils import read_test_params

PATH_TO_CONFIG = 'tests/test_config.yaml'


@pytest.fixture(scope='session')
def faker():
    return Faker()


@pytest.fixture(scope='session')
def test_params():
    return read_test_params(PATH_TO_CONFIG)


@pytest.fixture(scope='session')
def samples() -> Dict[str, int]:
    def inner(include_target: bool, faker, test_params):
        sample = {}
        features = test_params.feature_params
        for cat_feature in features.cat_features:
            sample[cat_feature] = faker.random.randint(0, 5)
        for bin_feature in features.bin_features:
            sample[bin_feature] = faker.random.randint(0, 1)
        for num_feature in features.num_features:
            sample[num_feature] = faker.random.randint(0, 100)
        if include_target:
            sample['target'] = faker.random.randint(0, 1)
        return sample
    return inner


@pytest.fixture(scope='session')
def train_dataset(
        samples,
        faker,
        test_params,
        include_target: bool = True
) -> pd.DataFrame:
    dataset = []
    for i in range(test_params.train_size):
        dataset.append(samples(include_target, faker, test_params))
    return pd.DataFrame(dataset)


@pytest.fixture(scope='session')
def val_dataset(
        samples,
        faker,
        test_params,
        include_target: bool = False
) -> pd.DataFrame:
    dataset = []
    for i in range(test_params.val_size):
        dataset.append(samples(include_target, faker, test_params))
    return pd.DataFrame(dataset)
