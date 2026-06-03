from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    total_debit: float
    total_credit: float
    net_amount: float
    avg_debit: float
    txn_count: int
    std_debit: float
    recency_days: int
    avg_amount_abs: float
    fraud_count: int
    fraud_rate: float
    mode_hour: int

class PredictionResponse(BaseModel):
    risk_probability: float
    credit_score: int
    risk_grade: str
