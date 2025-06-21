import pandas as pd
import numpy as np
import os

def engineer(df):
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

def main():
    inp = 'data/raw/transactions.csv'
    outp = 'data/processed/features.csv'
    os.makedirs('data/processed', exist_ok=True)
    df = pd.read_csv(inp)
    df_feat = engineer(df)
    df_feat.to_csv(outp, index=False)
    print('✅ Features saved:', df_feat.shape, '→', outp)

if __name__ == '__main__':
    main()
