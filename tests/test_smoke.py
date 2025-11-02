from fastapi.testclient import TestClient
from app import app  # imports your FastAPI app object

client = TestClient(app)

def test_version():
    r = client.get("/version")
    assert r.status_code == 200
    assert "model_version" in r.json()

def test_predict_minimal():
    payload = {
        "areaSqm": 45,
        "latitude": 52.0907,
        "longitude": 5.1214,
        "city": "utrecht",
        "pc4": "3511",
        "propertyType": "apartment",
        "furnish": "furnished",
        "internet": "yes",
        "kitchen": "own",
        "shower": "own",
        "toilet": "own",
        "living": "own",
        "smokingInside": "no",
        "pets": "no"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "predicted_rent" in body
    assert isinstance(body["predicted_rent"], (int, float))
