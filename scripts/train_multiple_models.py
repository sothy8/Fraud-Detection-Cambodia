#!/usr/bin/env python3
"""
Train and compare models on different datasets
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Ensure we're in the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

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
    
    # User behavior features
    user_stats = df.groupby('user_id').agg({
        'amount': ['mean', 'std', 'count'],
        'timestamp': 'min'
    }).reset_index()
    user_stats.columns = ['user_id', 'user_avg_amount', 'user_std_amount', 'user_txn_count', 'user_first_txn']
    df = df.merge(user_stats, on='user_id', how='left')
    
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
        'user_avg_amount', 'user_txn_count', 'txn_type_encoded',
        'is_night', 'is_business_hours', 'is_new_device', 'is_new_recipient'
    ]
    
    # Fill NaN values
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df[feature_cols + ['label']]

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name, model):
    """Train a model and return evaluation metrics"""
    print(f"\n🤖 Training {model_name}...")
    
    model.fit(X_train, y_train)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        auc_score = roc_auc_score(y_test, y_pred_proba)
    else:
        y_pred = model.predict(X_test)
        auc_score = None
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba if hasattr(model, 'predict_proba') else None,
        'auc_score': auc_score,
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score']
    }

def train_models_on_dataset(dataset_path, dataset_name):
    """Train multiple models on a single dataset"""
    print(f"\n{'='*60}")
    print(f"🔍 Processing Dataset: {dataset_name}")
    print(f"📁 Path: {dataset_path}")
    print(f"{'='*60}")
    
    # Load and prepare data
    df = pd.read_csv(dataset_path)
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🚨 Fraud rate: {df['label'].mean()*100:.2f}%")
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Prepare data for modeling
    X = df_features.drop('label', axis=1)
    y = df_features['label']
    
    print(f"🔧 Features: {list(X.columns)}")
    print(f"📈 Feature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000
        )
    }
    
    # Train and evaluate models
    results = {}
    for model_name, model in models.items():
        if model_name == 'Logistic Regression':
            # Use scaled features for logistic regression
            result = train_and_evaluate_model(
                X_train_scaled, X_test_scaled, y_train, y_test, model_name, model
            )
        else:
            result = train_and_evaluate_model(
                X_train, X_test, y_train, y_test, model_name, model
            )
        results[model_name] = result
    
    return results, df_features, X_test, y_test

def main():
    """Main training pipeline"""
    print("🇰🇭 Cambodia Fraud Detection - Multi-Model Training")
    print("="*60)
    
    # Create output directories
    os.makedirs('models/comparative', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Define datasets to process
    datasets = {
        'Small (5K)': 'data/datasets/small_5k.csv',
        'Realistic (75K)': 'data/datasets/realistic_75k.csv',
        'High Volume (150K)': 'data/datasets/high_volume_150k.csv',
        'High Fraud Test': 'data/datasets/high_fraud_test.csv',
        'Low Fraud Realistic': 'data/datasets/low_fraud_realistic.csv',
        'Sparse Users': 'data/datasets/sparse_users.csv'
    }
    
    # Store all results for comparison
    all_results = {}
    
    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        if os.path.exists(dataset_path):
            try:
                results, df_features, X_test, y_test = train_models_on_dataset(dataset_path, dataset_name)
                all_results[dataset_name] = results
                
                # Save best model for this dataset
                best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'] or 0)
                best_model = results[best_model_name]['model']
                
                model_filename = f"models/comparative/{dataset_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_best_model.pkl"
                joblib.dump(best_model, model_filename)
                print(f"💾 Best model ({best_model_name}) saved: {model_filename}")
                
            except Exception as e:
                print(f"❌ Error processing {dataset_name}: {e}")
        else:
            print(f"⚠️ Dataset not found: {dataset_path}")
    
    # Create comprehensive comparison report
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE MODEL COMPARISON REPORT")
    print(f"{'='*80}")
    
    # Create comparison DataFrame
    comparison_data = []
    for dataset_name, dataset_results in all_results.items():
        for model_name, metrics in dataset_results.items():
            comparison_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'AUC Score': metrics['auc_score'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print summary table
    print("\n📈 Performance Summary:")
    print(comparison_df.to_string(index=False, float_format='%.3f'))
    
    # Save detailed results
    comparison_df.to_csv('results/model_comparison_results.csv', index=False)
    print(f"\n💾 Detailed results saved to: results/model_comparison_results.csv")
    
    # Find best performing combinations
    print(f"\n🏆 TOP PERFORMERS:")
    best_auc = comparison_df.loc[comparison_df['AUC Score'].idxmax()]
    print(f"  🥇 Best AUC Score: {best_auc['Model']} on {best_auc['Dataset']} ({best_auc['AUC Score']:.3f})")
    
    best_f1 = comparison_df.loc[comparison_df['F1 Score'].idxmax()]
    print(f"  🥈 Best F1 Score: {best_f1['Model']} on {best_f1['Dataset']} ({best_f1['F1 Score']:.3f})")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"  • For production: Use {best_auc['Model']} trained on {best_auc['Dataset']}")
    print(f"  • For balanced performance: Use {best_f1['Model']} trained on {best_f1['Dataset']}")
    print(f"  • For high-volume scenarios: Test with 'High Volume (150K)' dataset")
    print(f"  • For algorithm testing: Use 'High Fraud Test' dataset")
    
    print(f"\n🎉 Training completed! Check the dashboard at http://localhost:8501")

if __name__ == '__main__':
    main()
