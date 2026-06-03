import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class CustomerFeatureAggregator(BaseEstimator, TransformerMixin):
    """Aggregate transaction data to customer-level features."""
    
    def __init__(self, id_col='CustomerId', amount_col='Amount', 
                 time_col='TransactionStartTime', fraud_col='FraudResult'):
        self.id_col = id_col
        self.amount_col = amount_col
        self.time_col = time_col
        self.fraud_col = fraud_col
        self.snapshot_date = None
    
    def fit(self, X, y=None):
        X_copy = X.copy()
        X_copy[self.time_col] = pd.to_datetime(X_copy[self.time_col])
        self.snapshot_date = X_copy[self.time_col].max()
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.time_col] = pd.to_datetime(X[self.time_col])
        
        # Split amounts into debit (positive) and credit (negative)
        X['debit'] = X[self.amount_col].apply(lambda x: x if x > 0 else 0)
        X['credit'] = X[self.amount_col].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Extract hour as a separate column before grouping
        X['hour'] = X[self.time_col].dt.hour
        
        # Group by customer
        grp = X.groupby(self.id_col)
        
        features = pd.DataFrame()
        features['total_debit'] = grp['debit'].sum()
        features['total_credit'] = grp['credit'].sum()
        features['net_amount'] = grp[self.amount_col].sum()
        features['avg_debit'] = grp['debit'].mean()
        features['txn_count'] = grp[self.time_col].count()
        features['std_debit'] = grp['debit'].std().fillna(0)
        features['recency_days'] = (self.snapshot_date - grp[self.time_col].max()).dt.days
        features['avg_amount_abs'] = grp[self.amount_col].apply(lambda x: x.abs().mean())
        features['fraud_count'] = grp[self.fraud_col].sum()
        features['fraud_rate'] = grp[self.fraud_col].mean()
        
        # Mode of transaction hour
        def mode_hour(series):
            mode_vals = series.mode()
            return mode_vals.iloc[0] if len(mode_vals) > 0 else 0
        features['mode_hour'] = grp['hour'].agg(mode_hour)
        
        return features


def build_feature_pipeline():
    pipeline = Pipeline([
        ('aggregator', CustomerFeatureAggregator()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return pipeline


def get_feature_names():
    return [
        'total_debit', 'total_credit', 'net_amount', 'avg_debit', 
        'txn_count', 'std_debit', 'recency_days', 'avg_amount_abs',
        'fraud_count', 'fraud_rate', 'mode_hour'
    ]


if __name__ == "__main__":
    df = pd.read_csv('data/raw/data.csv')
    pipeline = build_feature_pipeline()
    features = pipeline.fit_transform(df)
    print("Feature matrix shape:", features.shape)
    print("Sample:\n", features[:5])