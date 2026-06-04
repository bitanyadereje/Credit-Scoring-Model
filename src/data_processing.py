"""
src/data_processing.py
End-to-end feature engineering pipeline for credit risk.
Includes: customer-level aggregation, datetime features, categorical encoding,
imputation, scaling, WoE/IV analysis, and a scikit-learn Pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# -------------------------------------------------------------------
# Custom transformers
# -------------------------------------------------------------------

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Aggregate transaction-level data to customer-level features."""
    
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
        
        # Debit (positive) and credit (negative)
        X['debit'] = X[self.amount_col].apply(lambda x: x if x > 0 else 0)
        X['credit'] = X[self.amount_col].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Hour of transaction (for mode later)
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
        
        # Mode hour
        def mode_hour(series):
            mode_vals = series.mode()
            return mode_vals.iloc[0] if len(mode_vals) > 0 else 0
        features['mode_hour'] = grp['hour'].agg(mode_hour)
        
        # ---------- Categorical encoding: most frequent product category ----------
        # We need to keep ProductCategory in the original data for this
        if 'ProductCategory' in X.columns:
            # Get most frequent category per customer
            def most_frequent_cat(series):
                return series.mode().iloc[0] if not series.mode().empty else 'unknown'
            features['top_category'] = grp['ProductCategory'].agg(most_frequent_cat)
            # One-hot encode later in pipeline (after aggregation)
        else:
            features['top_category'] = 'unknown'
        
        return features


class CategoryOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode the 'top_category' column after aggregation."""
    def __init__(self, column='top_category'):
        self.column = column
        self.categories_ = None
    
    def fit(self, X, y=None):
        # X is a DataFrame with a 'top_category' column
        self.categories_ = X[self.column].unique()
        return self
    
    def transform(self, X):
        X = X.copy()
        dummies = pd.get_dummies(X[self.column], prefix='cat', dtype=float)
        # Drop the original column
        X = X.drop(columns=[self.column])
        # Concatenate dummies
        X = pd.concat([X, dummies], axis=1)
        return X


# -------------------------------------------------------------------
# WoE/IV analysis function (for documentation and feature selection)
# -------------------------------------------------------------------

def compute_woe_iv(df, target_col, feature_cols=None, n_bins=10):
    """
    Compute Information Value (IV) for each feature using custom binning.
    This does not require xverse.
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    
    results = {}
    for col in feature_cols:
        # Skip if column is constant
        if df[col].nunique() <= 1:
            results[col] = 0.0
            continue
        
        # For numeric columns, bin into n_bins groups
        if df[col].dtype in ['int64', 'float64']:
            # Use quantile binning for even distribution
            try:
                df_temp = df[[col, target_col]].copy()
                df_temp['bin'] = pd.qcut(df_temp[col], q=n_bins, duplicates='drop')
            except ValueError:
                # If quantile fails, use equal-width bins
                df_temp['bin'] = pd.cut(df_temp[col], bins=n_bins, duplicates='drop')
        else:
            # Categorical: use categories directly
            df_temp = df[[col, target_col]].copy()
            df_temp['bin'] = df_temp[col]
        
        # Calculate WoE and IV for each bin
        grouped = df_temp.groupby('bin')[target_col].agg(['count', 'sum'])
        grouped.columns = ['total', 'bad']
        grouped['good'] = grouped['total'] - grouped['bad']
        
        total_bad = grouped['bad'].sum()
        total_good = grouped['good'].sum()
        
        if total_bad == 0 or total_good == 0:
            results[col] = 0.0
            continue
        
        grouped['bad_pct'] = grouped['bad'] / total_bad
        grouped['good_pct'] = grouped['good'] / total_good
        grouped['woe'] = np.log(grouped['bad_pct'] / grouped['good_pct'])
        grouped['iv_contrib'] = (grouped['bad_pct'] - grouped['good_pct']) * grouped['woe']
        iv = grouped['iv_contrib'].sum()
        results[col] = iv
    
    return pd.Series(results).sort_values(ascending=False)
# -------------------------------------------------------------------
# Main pipeline builder
# -------------------------------------------------------------------

def build_feature_pipeline():
    """
    Returns a complete scikit-learn Pipeline that:
    1. Aggregates transactions to customer level (with categorical encoding).
    2. Imputes missing values (median).
    3. Scales numerical features to zero mean, unit variance.
    """
    pipeline = Pipeline([
        ('aggregator', CustomerAggregator()),
        ('cat_encoder', CategoryOneHotEncoder()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return pipeline


def get_feature_names():
    """Return list of feature names after aggregation and encoding."""
    # This is a static list; the actual names depend on categories present.
    # For clarity, we return the base names (excluding one-hot expanded).
    return [
        'total_debit', 'total_credit', 'net_amount', 'avg_debit',
        'txn_count', 'std_debit', 'recency_days', 'avg_amount_abs',
        'fraud_count', 'fraud_rate', 'mode_hour'
    ] + ['cat_' + cat for cat in ['airtime', 'financial_services', 'utility_bill', 'unknown']]


# -------------------------------------------------------------------
# Quick test when run directly
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv('data/raw/data.csv')
    print("Raw data shape:", df.shape)
    
    # Build and apply pipeline
    pipeline = build_feature_pipeline()
    features = pipeline.fit_transform(df)
    print("Feature matrix shape after pipeline:", features.shape)
    print("First 5 rows:\n", features[:5])
    
    # To compute IV, we would need the target variable. Since we don't have it yet,
    # we show a dummy call after we create the target (in create_target.py).
    # This function is provided for use in the training script or notebook.
    print("\nWoE/IV function is available as compute_woe_iv().")