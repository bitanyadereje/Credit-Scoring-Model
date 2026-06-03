import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def aggregate_customers(df):
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    snapshot_date = df['TransactionStartTime'].max()
    df['debit'] = df['Amount'].apply(lambda x: x if x > 0 else 0)
    df['credit'] = df['Amount'].apply(lambda x: abs(x) if x < 0 else 0)
    df['hour'] = df['TransactionStartTime'].dt.hour
    grp = df.groupby('CustomerId')
    features = pd.DataFrame()
    features['total_debit'] = grp['debit'].sum()
    features['total_credit'] = grp['credit'].sum()
    features['net_amount'] = grp['Amount'].sum()
    features['avg_debit'] = grp['debit'].mean()
    features['txn_count'] = grp['TransactionStartTime'].count()
    features['std_debit'] = grp['debit'].std().fillna(0)
    features['recency_days'] = (snapshot_date - grp['TransactionStartTime'].max()).dt.days
    features['avg_amount_abs'] = grp['Amount'].apply(lambda x: x.abs().mean())
    features['fraud_count'] = grp['FraudResult'].sum()
    features['fraud_rate'] = grp['FraudResult'].mean()
    def mode_hour(series):
        mode_vals = series.mode()
        return mode_vals.iloc[0] if len(mode_vals) > 0 else 0
    features['mode_hour'] = grp['hour'].agg(mode_hour)
    return features

def create_rfm_features(customer_df):
    rfm = customer_df[['recency_days', 'txn_count', 'total_debit']].copy()
    rfm.columns = ['recency', 'frequency', 'monetary']
    return rfm

def assign_high_risk_cluster(rfm, n_clusters=3, random_state=42):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    cluster_means = rfm.groupby(clusters)['monetary'].mean()
    high_risk_cluster = cluster_means.idxmin()
    high_risk = (clusters == high_risk_cluster).astype(int)
    return high_risk

if __name__ == "__main__":
    df = pd.read_csv('data/raw/data.csv')
    customer_features = aggregate_customers(df)
    rfm = create_rfm_features(customer_features)
    high_risk = assign_high_risk_cluster(rfm)
    customer_features['is_high_risk'] = high_risk
    print("Target distribution:")
    print(customer_features['is_high_risk'].value_counts(normalize=True))
    os.makedirs('data/processed', exist_ok=True)
    customer_features.to_csv('data/processed/customer_features_with_target.csv')
    print("Saved to data/processed/customer_features_with_target.csv")
