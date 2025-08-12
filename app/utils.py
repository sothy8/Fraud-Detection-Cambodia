import pandas as pd, joblib
import os
import subprocess
import sys
from pathlib import Path

MODEL_PATH = os.path.join('models', 'fraud_model.pkl')

def engineer_features(df):
    """Enhanced feature engineering with better error handling"""
    try:
        df = df.sort_values(['user_id', 'timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # User behavior features
        df['user_amount_mean'] = df.groupby('user_id')['amount'].transform('mean')
        df['user_amount_std'] = df.groupby('user_id')['amount'].transform('std').replace(0, 1)
        df['amount_zscore'] = (df['amount'] - df['user_amount_mean']) / df['user_amount_std']
        
        # Temporal features
        df['time_since_last'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Behavioral anomaly features
        df['is_new_recipient'] = (df.groupby('user_id')['recipient_id'].shift() != df['recipient_id']).astype(int)
        df['is_new_device'] = (df.groupby('user_id')['device_id'].shift() != df['device_id']).astype(int)
        
        # Transaction frequency features
        df['user_txn_count'] = df.groupby('user_id').cumcount() + 1
        df['amount_rank'] = df.groupby('user_id')['amount'].rank(pct=True)
        
        return df
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return df

def load_model():
    """Load the trained fraud detection model"""
    try:
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def check_model_exists():
    """Check if trained model exists"""
    return os.path.exists(MODEL_PATH)

def score(df, model):
    """Score transactions for fraud probability"""
    try:
        feats = engineer_features(df.copy())
        
        # Select features for prediction
        feature_cols = ['amount_zscore', 'time_since_last', 'is_new_recipient', 'is_new_device', 'hour']
        available_cols = [col for col in feature_cols if col in feats.columns]
        
        if len(available_cols) < len(feature_cols):
            print(f"Warning: Some features missing. Using: {available_cols}")
        
        X = feats[available_cols]
        
        # Handle missing values
        X = X.fillna(0)
        
        # Get predictions
        if model is not None:
            probs = model.predict_proba(X)[:, 1]
            feats['fraud_score'] = probs
            feats['is_fraud_pred'] = (probs > 0.5).astype(int)
        else:
            # Fallback if model is not available
            feats['fraud_score'] = 0.0
            feats['is_fraud_pred'] = 0
            
        return feats
    except Exception as e:
        print(f"Error in scoring: {e}")
        # Return original dataframe with default scores
        df['fraud_score'] = 0.0
        df['is_fraud_pred'] = 0
        return df

def generate_sample_data():
    """Generate sample data and train model pipeline"""
    try:
        print("🎲 Generating sample transactions...")
        
        # Run simulation script
        result = subprocess.run([
            sys.executable, 'scripts/simulate_transactions.py'
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"Error generating data: {result.stderr}")
            return False
            
        print("✅ Sample data generated")
        
        print("🔧 Engineering features...")
        # Run feature engineering
        result = subprocess.run([
            sys.executable, 'scripts/feature_engineering.py'
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"Error in feature engineering: {result.stderr}")
            return False
            
        print("✅ Features engineered")
        
        print("🤖 Training model...")
        # Train model
        result = subprocess.run([
            sys.executable, 'scripts/train_model.py'
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            print(f"Error training model: {result.stderr}")
            return False
            
        print("✅ Model trained successfully")
        return True
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return False

def get_model_info():
    """Get information about the current model"""
    if not check_model_exists():
        return None
        
    try:
        model = load_model()
        info = {
            'model_type': type(model).__name__,
            'file_size': f"{os.path.getsize(MODEL_PATH) / 1024:.1f} KB",
            'last_modified': pd.Timestamp.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime('%Y-%m-%d %H:%M:%S')
        }
        return info
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None

def validate_transaction_data(df):
    """Validate transaction data format"""
    required_columns = ['txn_id', 'user_id', 'recipient_id', 'amount', 'timestamp', 'device_id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check data types
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['amount'] = pd.to_numeric(df['amount'])
    except Exception as e:
        raise ValueError(f"Data type conversion error: {e}")
    
    return True
