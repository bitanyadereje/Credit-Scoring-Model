import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    sample = {
        "total_debit": 5000.0,
        "total_credit": 0.0,
        "net_amount": 5000.0,
        "avg_debit": 1250.0,
        "txn_count": 4,
        "std_debit": 500.0,
        "recency_days": 10,
        "avg_amount_abs": 1250.0,
        "fraud_count": 0,
        "fraud_rate": 0.0,
        "mode_hour": 14
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert "credit_score" in data
    assert "risk_grade" in data
    assert 0 <= data["risk_probability"] <= 1
    assert 300 <= data["credit_score"] <= 850
