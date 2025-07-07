from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_predict_spam():
    response = client.post("/predict", json={"text": "Win free money now!"})
    assert response.status_code == 200
    assert response.json()["prediction"] == "spam"

def test_predict_ham():
    response = client.post("/predict", json={"text": "Meeting at 10 am"})
    assert response.status_code == 200
    assert response.json()["prediction"] == "ham"

def test_predict_no_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Validation error from FastAPI

def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "Metin boş olamaz."