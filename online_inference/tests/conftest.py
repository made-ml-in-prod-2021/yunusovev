import pytest


@pytest.fixture()
def positive_request():
    return {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1
        }


@pytest.fixture()
def negative_request():
    return {
        'age': 57,
        'sex': 0,
        'cp': 1,
        'trestbps': 130,
        'chol': 236,
        'fbs': 0,
        'restecg': 0,
        'thalach': 174,
        'exang': 0,
        'oldpeak': 0.0,
        'slope': 1,
        'ca': 1,
        'thal': 2
    }


@pytest.fixture()
def bad_request():
    return {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": "a",
            "slope": 0,
            "ca": 0,
            "thal": 1
        }