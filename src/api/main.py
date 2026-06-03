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
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bati Bank | Credit Risk API</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: #e2e8f0;
                padding: 2rem;
                min-height: 100vh;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 {
                font-size: 2.5rem;
                font-weight: 600;
                background: linear-gradient(120deg, #38bdf8, #818cf8);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                margin-bottom: 0.5rem;
                display: inline-block;
            }
            .subtitle {
                color: #94a3b8;
                margin-bottom: 2rem;
                border-left: 3px solid #38bdf8;
                padding-left: 1rem;
            }
            .grid {
                display: flex;
                flex-wrap: wrap;
                gap: 2rem;
            }
            .form-card {
                flex: 2;
                min-width: 300px;
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(8px);
                border-radius: 1.5rem;
                padding: 1.5rem;
                border: 1px solid rgba(56, 189, 248, 0.2);
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            .result-card {
                flex: 1;
                min-width: 280px;
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(8px);
                border-radius: 1.5rem;
                padding: 1.5rem;
                border: 1px solid rgba(56, 189, 248, 0.2);
                display: flex;
                flex-direction: column;
                justify-content: center;
                text-align: center;
            }
            h2 {
                font-size: 1.25rem;
                font-weight: 500;
                margin-bottom: 1rem;
                color: #cbd5e1;
                letter-spacing: -0.01em;
            }
            .form-row {
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                margin-bottom: 1rem;
            }
            .input-group {
                flex: 1;
                min-width: 120px;
            }
            label {
                display: block;
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #94a3b8;
                margin-bottom: 0.25rem;
            }
            input {
                width: 100%;
                padding: 0.5rem 0.75rem;
                background: #0f172a;
                border: 1px solid #334155;
                border-radius: 0.75rem;
                color: #f1f5f9;
                font-size: 0.875rem;
                transition: all 0.2s;
            }
            input:focus {
                outline: none;
                border-color: #38bdf8;
                box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
            }
            button {
                background: linear-gradient(120deg, #38bdf8, #818cf8);
                border: none;
                border-radius: 2rem;
                padding: 0.7rem 1.5rem;
                font-weight: 600;
                color: white;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                width: 100%;
                margin-top: 0.5rem;
                font-size: 1rem;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px -5px rgba(56, 189, 248, 0.4);
            }
            .result-value {
                font-size: 3rem;
                font-weight: 700;
                margin: 0.5rem 0;
                background: linear-gradient(120deg, #facc15, #f97316);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
            }
            .result-label {
                font-size: 0.875rem;
                text-transform: uppercase;
                color: #94a3b8;
            }
            .grade-box {
                background: rgba(15, 23, 42, 0.6);
                border-radius: 1rem;
                padding: 0.5rem;
                margin-top: 1rem;
            }
            .footer {
                margin-top: 2rem;
                text-align: center;
                font-size: 0.75rem;
                color: #475569;
            }
            .badge {
                display: inline-block;
                background: #38bdf8;
                color: #0f172a;
                border-radius: 9999px;
                padding: 0.2rem 0.8rem;
                font-size: 0.7rem;
                font-weight: 600;
                margin-bottom: 1rem;
            }
            @media (max-width: 768px) {
                body { padding: 1rem; }
                .grid { flex-direction: column; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="badge">🔐 Basel II Compliant</div>
            <h1>🏦 Bati Bank Credit Risk API</h1>
            <div class="subtitle">Real‑time credit scoring for buy‑now‑pay‑later • Powered by Random Forest & MLflow</div>
            
            <div class="grid">
                <div class="form-card">
                    <h2>📋 Customer Risk Profile</h2>
                    <form id="predictionForm">
                        <div class="form-row">
                            <div class="input-group"><label>💰 Total Debit</label><input type="number" id="total_debit" value="5000" step="any"></div>
                            <div class="input-group"><label>💳 Total Credit</label><input type="number" id="total_credit" value="0" step="any"></div>
                            <div class="input-group"><label>⚖️ Net Amount</label><input type="number" id="net_amount" value="5000" step="any"></div>
                        </div>
                        <div class="form-row">
                            <div class="input-group"><label>📊 Avg Debit</label><input type="number" id="avg_debit" value="1250" step="any"></div>
                            <div class="input-group"><label>🔄 Tx Count</label><input type="number" id="txn_count" value="4" step="1"></div>
                            <div class="input-group"><label>📈 Std Debit</label><input type="number" id="std_debit" value="500" step="any"></div>
                        </div>
                        <div class="form-row">
                            <div class="input-group"><label>📅 Recency (days)</label><input type="number" id="recency_days" value="10" step="1"></div>
                            <div class="input-group"><label>💵 Avg Amount Abs</label><input type="number" id="avg_amount_abs" value="1250" step="any"></div>
                        </div>
                        <div class="form-row">
                            <div class="input-group"><label>⚠️ Fraud Count</label><input type="number" id="fraud_count" value="0" step="1"></div>
                            <div class="input-group"><label>📉 Fraud Rate</label><input type="number" id="fraud_rate" value="0" step="0.01"></div>
                            <div class="input-group"><label>⏰ Mode Hour</label><input type="number" id="mode_hour" value="14" step="1"></div>
                        </div>
                        <button type="button" id="predictBtn">🔮 Predict Credit Risk</button>
                    </form>
                </div>

                <div class="result-card" id="resultCard">
                    <h2>📊 Risk Assessment</h2>
                    <div id="predictionResult" style="width: 100%;">
                        <div class="result-label">Risk Probability</div>
                        <div class="result-value" id="probValue">—</div>
                        <div class="result-label">Credit Score</div>
                        <div class="result-value" id="scoreValue" style="font-size: 2.5rem;">—</div>
                        <div class="grade-box">
                            <div class="result-label">Risk Grade</div>
                            <div id="gradeValue" style="font-size: 1.5rem; font-weight: 600;">—</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="footer">
                <span>⚡ 24/7 API | 📈 Model version: Random Forest (ROC‑AUC 0.9999) | 🐳 Docker ready</span>
            </div>
        </div>

        <script>
            document.getElementById('predictBtn').addEventListener('click', async () => {
                const payload = {
                    total_debit: parseFloat(document.getElementById('total_debit').value),
                    total_credit: parseFloat(document.getElementById('total_credit').value),
                    net_amount: parseFloat(document.getElementById('net_amount').value),
                    avg_debit: parseFloat(document.getElementById('avg_debit').value),
                    txn_count: parseInt(document.getElementById('txn_count').value),
                    std_debit: parseFloat(document.getElementById('std_debit').value),
                    recency_days: parseInt(document.getElementById('recency_days').value),
                    avg_amount_abs: parseFloat(document.getElementById('avg_amount_abs').value),
                    fraud_count: parseInt(document.getElementById('fraud_count').value),
                    fraud_rate: parseFloat(document.getElementById('fraud_rate').value),
                    mode_hour: parseInt(document.getElementById('mode_hour').value)
                };
                
                document.getElementById('probValue').innerHTML = '🔄 ...';
                document.getElementById('scoreValue').innerHTML = '...';
                document.getElementById('gradeValue').innerHTML = '...';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await response.json();
                    if (response.ok) {
                        document.getElementById('probValue').innerHTML = (data.risk_probability * 100).toFixed(2) + '%';
                        document.getElementById('scoreValue').innerHTML = data.credit_score;
                        document.getElementById('gradeValue').innerHTML = data.risk_grade;
                    } else {
                        document.getElementById('probValue').innerHTML = '❌ Error';
                        document.getElementById('scoreValue').innerHTML = data.detail || 'Failed';
                        document.getElementById('gradeValue').innerHTML = '—';
                    }
                } catch (err) {
                    document.getElementById('probValue').innerHTML = '⚠️ Network error';
                    document.getElementById('scoreValue').innerHTML = '—';
                    document.getElementById('gradeValue').innerHTML = '—';
                }
            });
        </script>
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