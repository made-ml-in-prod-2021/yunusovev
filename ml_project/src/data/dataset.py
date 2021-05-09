from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs.config import SplittingParams


def read_data(path_to_data: str) -> pd.DataFrame:
    return pd.read_csv(path_to_data)


def split_data(
    data: pd.DataFrame,
    params: SplittingParams
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, val = train_test_split(
        data, 
        test_size=params.val_size,
        random_state=params.random_state,
        shuffle=params.shuffle
        )
    return train, val


def save_data(data: pd.DataFrame, output_path: str) -> None:
    data.to_csv(output_path, index=False)
