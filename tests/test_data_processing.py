"""
tests/test_data_processing.py
Unit tests for data processing transformers.
"""

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from src.data_processing import CustomerAggregator, CategoryOneHotEncoder, compute_woe_iv

def test_customer_aggregator_output_shape():
    """Test that aggregator returns correct number of customers and features."""
    df = pd.DataFrame({
        'CustomerId': ['A', 'A', 'B', 'B', 'C'],
        'Amount': [100, -20, 200, 50, -10],
        'TransactionStartTime': [
            '2024-01-01 10:00', '2024-01-02 14:00',
            '2024-01-01 09:00', '2024-01-05 16:00',
            '2024-01-03 12:00'
        ],
        'FraudResult': [0, 0, 1, 0, 0],
        'ProductCategory': ['airtime', 'airtime', 'utility', 'utility', 'financial']
    })
    
    agg = CustomerAggregator()
    agg.fit(df)
    result = agg.transform(df)
    
    assert result.shape[0] == 3
    expected_cols = ['total_debit', 'total_credit', 'txn_count', 'recency_days']
    for col in expected_cols:
        assert col in result.columns

def test_category_one_hot_encoder():
    """Test that one-hot encoding produces correct number of columns."""
    df = pd.DataFrame({
        'top_category': ['airtime', 'utility', 'airtime', 'financial', 'unknown']
    })
    encoder = CategoryOneHotEncoder()
    encoder.fit(df)
    transformed = encoder.transform(df)
    
    assert 'top_category' not in transformed.columns
    assert 'cat_airtime' in transformed.columns
    assert 'cat_utility' in transformed.columns
    assert 'cat_financial' in transformed.columns
    assert 'cat_unknown' in transformed.columns

def test_compute_woe_iv_output():
    """Test that IV function returns a Series with non-negative values."""
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.choice(['A', 'B'], 100),
        'target': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    })
    iv = compute_woe_iv(df, 'target')
    assert isinstance(iv, pd.Series)
    assert all(iv >= 0)