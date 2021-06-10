from fastapi.testclient import TestClient
from ml_classifier_online.app import app


def test_main():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200


def test_predict_posittive(positive_request):
    with TestClient(app) as client:
        response = client.post(url='/predict/', json=positive_request)
    assert response.status_code == 200
    assert response.json()['predict'] == 1


def test_predict_negative(negative_request):
    with TestClient(app) as client:
        response = client.post(url='/predict/', json=negative_request)
    assert response.status_code == 200
    assert response.json()['predict'] == 0


def test_predict_unprocessible(bad_request):
    with TestClient(app) as client:
        response = client.post(url='/predict/', json=bad_request)
    assert response.status_code == 422


def test_predict_bad_request():
    with TestClient(app) as client:
        response = client.post(url='/predict/', data=4)
    assert response.status_code == 400
