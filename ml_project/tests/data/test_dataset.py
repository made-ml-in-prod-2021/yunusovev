import pytest
from faker import Faker
import pandas as pd

from src.configs.config import SplittingParams
from src.data.dataset import (
    read_data,
    save_data,
    split_data
)

DATASET_SIZE = 100
VAL_SIZE = 0.2
RANDOM_STATE = 42
faker = Faker()


@pytest.fixture()
def fake_dataset(tmpdir):
    path = tmpdir.join('fake_dataset.csv')
    df_fake = pd.DataFrame([faker.random.randint(0, 100) for _ in range(DATASET_SIZE)], columns=['field'])
    save_data(df_fake, path)
    return df_fake


def test_read_save_data(tmpdir, fake_dataset):
    path = tmpdir.join('fake_dataset.csv')
    df_fake = read_data(path)
    assert fake_dataset.equals(df_fake)


def test_split_data(fake_dataset):
    params = SplittingParams(val_size=VAL_SIZE, random_state=RANDOM_STATE)
    assert params.val_size == VAL_SIZE
    assert params.random_state == RANDOM_STATE
    train, val = split_data(fake_dataset, params)
    assert train.shape[0] == int(DATASET_SIZE * (1 - VAL_SIZE))
    assert val.shape[0] == int(DATASET_SIZE * VAL_SIZE)
