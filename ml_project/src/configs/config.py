from typing import Optional, List
from dataclasses import dataclass, field


@dataclass()
class ClfParams:
    model_type: str = field(default='SGDClassifier')
    random_state: int = field(default=42)
    n_estimators: int = field(default=40)
    max_depth: int = field(default=3)
    loss: str = field(default='log')
    alpha: float = field(default=0.001)
    max_iter: int = field(default=300)


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=42)
    shuffle: bool = field(default=True)


@dataclass
class FeaturesParams:
    num_features: Optional[List[str]]
    cat_features: Optional[List[str]]
    bin_features: Optional[List[str]]


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metrics_path: str
    clf_params: ClfParams
    splitting_params: SplittingParams
    feature_params: FeaturesParams
    target: str


@dataclass()
class PredictParams:
    input_data_path: str
    output_data_path: str
    model_path: str
