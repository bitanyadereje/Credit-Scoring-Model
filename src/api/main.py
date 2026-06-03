from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import numpy as np
import joblib
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

app = FastAPI(
    title="🏦 Bati Bank Credit Risk API",
    description="Predicts default probability and credit score using RFM-based proxy target.",
    version="2.0.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

# Load model and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

def probability_to_score(p, base_points=600, odds_at_base=50, pdo=20):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    odds = (1 - p) / p
    score = base_points + pdo * (np.log2(odds) - np.log2(odds_at_base))
    return int(round(score))

def score_to_grade(score):
    if score >= 700:
        return "A (Low Risk)"
    elif score >= 600:
        return "B (Medium-Low Risk)"
    elif score >= 500:
        return "C (Medium Risk)"
    else:
        return "D (High Risk)"

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Bati Bank Credit Risk API</title></head>
    <body style="font-family: Arial; margin: 40px;">
        <h1>🏦 Bati Bank Credit Risk API</h1>
        <p>Use <a href="/docs">/docs</a> for interactive documentation.</p>
    </body>
    </html>
    """

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    try:
        input_array = np.array([[
            features.total_debit, features.total_credit, features.net_amount,
            features.avg_debit, features.txn_count, features.std_debit,
            features.recency_days, features.avg_amount_abs, features.fraud_count,
            features.fraud_rate, features.mode_hour
        ]])
        input_scaled = scaler.transform(input_array)
        prob = model.predict_proba(input_scaled)[0, 1]
        score = probability_to_score(prob)
        grade = score_to_grade(score)
        return PredictionResponse(risk_probability=prob, credit_score=score, risk_grade=grade)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
