import pandas as pd, joblib
import os

MODEL_PATH = os.path.join('models', 'fraud_model.pkl')

def engineer_features(df):
    df = df.sort_values(['user_id', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['user_amount_mean'] = df.groupby('user_id')['amount'].transform('mean')
    df['user_amount_std'] = df.groupby('user_id')['amount'].transform('std').replace(0, 1)
    df['amount_zscore'] = (df['amount'] - df['user_amount_mean']) / df['user_amount_std']
    df['time_since_last'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
    df['is_new_recipient'] = (df.groupby('user_id')['recipient_id'].shift() != df['recipient_id']).astype(int)
    df['is_new_device'] = (df.groupby('user_id')['device_id'].shift() != df['device_id']).astype(int)
    df['hour'] = df['timestamp'].dt.hour
    return df

def load_model():
    return joblib.load(MODEL_PATH)

def score(df, model):
    feats = engineer_features(df.copy())
    X = feats[['amount_zscore','time_since_last','is_new_recipient','is_new_device','hour']]
    probs = model.predict_proba(X)[:,1]
    feats['fraud_score'] = probs
    feats['is_fraud_pred'] = (probs > 0.5).astype(int)
    return feats
