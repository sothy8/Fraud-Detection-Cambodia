#!/usr/bin/env python3
"""
Simple model training on custom datasets
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from datetime import datetime

def engineer_features(df):
    """Create features for fraud detection"""
    df = df.copy()
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Amount features
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    
    # Transaction type encoding
    df['txn_type_encoded'] = df['txn_type'].map({'p2p': 0, 'merchant': 1, 'topup': 2})
    
    # Time-based features
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    # Device and recipient features
    df['is_new_device'] = df.groupby('user_id')['device_id'].transform(lambda x: (x != x.shift()).astype(int))
    df['is_new_recipient'] = df.groupby('user_id')['recipient_id'].transform(lambda x: (x != x.shift()).astype(int))
    
    # Select features for modeling
    feature_cols = [
        'amount_log', 'amount_zscore', 'hour', 'day_of_week', 'is_weekend',
        'txn_type_encoded', 'is_night', 'is_business_hours', 'is_new_device', 'is_new_recipient'
    ]
    
    # Fill NaN values
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df[feature_cols + ['label']]

def train_model_on_dataset(dataset_path, dataset_name):
    """Train XGBoost model on a dataset"""
    print(f"\n{'='*50}")
    print(f"🔍 Processing: {dataset_name}")
    print(f"{'='*50}")
    
    # Load data
    df = pd.read_csv(dataset_path)
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🚨 Fraud rate: {df['label'].mean()*100:.2f}%")
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Prepare data
    X = df_features.drop('label', axis=1)
    y = df_features['label']
    
    print(f"🔧 Features: {list(X.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost model
    print("🤖 Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1,
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Evaluate
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print("📈 Results:")
    print(f"  AUC Score: {auc_score:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save model
    model_filename = f"models/{dataset_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_model.pkl"
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_filename)
    print(f"💾 Model saved: {model_filename}")
    
    return {
        'dataset': dataset_name,
        'auc_score': auc_score,
        'model_path': model_filename,
        'fraud_rate': df['label'].mean() * 100,
        'total_transactions': len(df)
    }

def main():
    """Main training pipeline"""
    print("🇰🇭 Cambodia Fraud Detection - Model Training")
    print("=" * 60)
    
    # Define datasets
    datasets = {
        'Small_5K': 'data/datasets/small_5k.csv',
        'Realistic_75K': 'data/datasets/realistic_75k.csv',
        'High_Volume_150K': 'data/datasets/high_volume_150k.csv',
        'High_Fraud_Test': 'data/datasets/high_fraud_test.csv',
        'Low_Fraud_Realistic': 'data/datasets/low_fraud_realistic.csv',
        'Sparse_Users': 'data/datasets/sparse_users.csv'
    }
    
    # Train models on each dataset
    results = []
    for dataset_name, dataset_path in datasets.items():
        if os.path.exists(dataset_path):
            try:
                result = train_model_on_dataset(dataset_path, dataset_name)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {dataset_name}: {e}")
        else:
            print(f"⚠️ Dataset not found: {dataset_path}")
    
    # Summary report
    print(f"\n{'='*60}")
    print("📊 TRAINING SUMMARY REPORT")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save summary
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/training_summary.csv', index=False)
    
    # Best model recommendation
    best_model = results_df.loc[results_df['auc_score'].idxmax()]
    print(f"\n🏆 BEST PERFORMING MODEL:")
    print(f"  Dataset: {best_model['dataset']}")
    print(f"  AUC Score: {best_model['auc_score']:.3f}")
    print(f"  Model Path: {best_model['model_path']}")
    
    print(f"\n✅ Training completed! All models saved in 'models/' directory")
    print(f"📊 Summary saved to: results/training_summary.csv")

if __name__ == '__main__':
    main()
