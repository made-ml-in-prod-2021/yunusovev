ml_project
==============================

Package for classification task

Requirements: Python 3.8+

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install .
~~~
Train:
~~~
# SGDClassidier
python ml_classifier/trainer.py configs/train_config_sgd.yaml

#RandomForestClassifier
python -m ml_classifier.trainer configs/train_config_rf.yaml
~~~

Predict:
~~~
python -m ml_classifier.predict configs/predict_config.yaml
~~~

Test:
~~~
pytest -v
~~~

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── configs            <- Configs example for train and predict.
    ├── data               <- The original, immutable data dump.
    ├── tests              <- Tests
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── setup.py           <- makes project pip installable (pip install .) so src can be imported
    ├── setup.cfg          
    ├── ml_classifier      <- Source code for use in this project.
    │   ├── __init__.py    
    │   │
    │   ├── configs        <- Dataclases for working with configs
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │                  
    │   ├── utils          <- Help functions
    │   │
    │   ├── models         <- code to train models and then use trained models to make
    └── 


--------

<p><small>Project based on the <a target="_blank" href="https://github.com/made-ml-in-prod-2021/ml_project_example">ml_example</a>.

