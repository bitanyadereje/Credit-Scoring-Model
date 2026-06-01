# Credit Risk Model – Bati Bank x eCommerce BNPL

End-to-end implementation of a credit scoring model using RFM-based proxy target, MLflow tracking, and containerized FastAPI deployment.

## Credit Scoring Business Understanding

### How does the Basel II Accord’s emphasis on risk measurement influence the need for an interpretable and well-documented model?

Basel II requires financial institutions to validate, document, and explain their risk models. For credit scoring, this means every feature transformation and model parameter must be justified. The model must produce interpretable outputs that risk officers can explain to regulators. Documentation must cover variable selection, performance monitoring, and model limitations.

### Without a direct “default” label, why is a proxy variable necessary, and what business risks does proxy‑based prediction introduce?

The raw transaction data contains no historical default flag. We therefore need a proxy target (e.g., disengaged customers labeled as high-risk) to approximate credit risk. Business risks include label bias (proxy may not align with true default), concept drift (customer behavior changes), regulatory scrutiny (must be disclosed), and adverse selection (wrong approvals/rejections).

### What are the key trade‑offs between a simple, interpretable model (e.g., Logistic Regression with WoE) and a high‑performance model (e.g., Gradient Boosting) in a regulated financial context?

| Aspect | Logistic Regression + WoE | Gradient Boosting |
|--------|---------------------------|-------------------|
| Interpretability | High – coefficients show impact | Low – needs SHAP |
| Regulatory acceptance | Preferred | Acceptable with explainability |
| Predictive performance | Moderate | Higher |
| Risk of overfitting | Lower | Higher |
| Maintenance | Stable | Needs frequent recalibration |
 
We will train both and compare ROC-AUC, F1, and precision-recall.
