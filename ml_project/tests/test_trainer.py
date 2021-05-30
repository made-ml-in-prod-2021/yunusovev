import os

import pytest

from ml_classifier.data.dataset import save_data
from ml_classifier.trainer import train
from ml_classifier.configs.config import (
    TrainingPipelineParams,
    ClfParams,
    SplittingParams,
)


@pytest.fixture()
def tmp_training_pipeline_params(tmpdir, test_params):
    return TrainingPipelineParams(
        input_data_path=tmpdir.join(test_params.input_data_path),
        output_model_path=tmpdir.join(test_params.model_path),
        metrics_path=tmpdir.join(test_params.metrics_path),
        splitting_params=SplittingParams(),
        clf_params=ClfParams(),
        feature_params=test_params.feature_params,
        target=test_params.target
    )


def test_train_end2end(tmp_training_pipeline_params, train_dataset, test_params):
    save_data(train_dataset, tmp_training_pipeline_params.input_data_path)
    metrics = train(tmp_training_pipeline_params)
    assert os.path.exists(tmp_training_pipeline_params.output_model_path)
    assert os.path.exists(tmp_training_pipeline_params.metrics_path)
    assert metrics.accuracy > 0
