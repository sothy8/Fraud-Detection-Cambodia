import pandas as pd
import numpy as np
from faker import Faker
import random, os
from datetime import datetime, timedelta

def generate_transactions(n_users=500, n_txns=20000, fraud_ratio=0.01, seed=42):
    fake = Faker()
    random.seed(seed)
    np.random.seed(seed)
    users = [f'u{str(i).zfill(4)}' for i in range(n_users)]
    device_pool = [fake.uuid4() for _ in range(n_users*2)]
    rows = []
    base_time = datetime.now() - timedelta(days=30)

    for i in range(n_txns):
        sender = random.choice(users)
        recipient = random.choice(users)
        amount = round(np.random.gamma(2, 15000), 2)
        tx_type = random.choice(['p2p', 'merchant', 'topup'])
        timestamp = base_time + timedelta(seconds=random.randint(0, 30*24*60*60))
        device_id = random.choice(device_pool)
        lat = round(random.uniform(10.4, 13.6), 6)
        lon = round(random.uniform(102.5, 106.0), 6)
        fraud = np.random.rand() < fraud_ratio

        if fraud:
            amount *= random.uniform(5, 20)
            device_id = fake.uuid4()

        rows.append({
            'txn_id': f't{i}',
            'user_id': sender,
            'recipient_id': recipient,
            'amount': amount,
            'txn_type': tx_type,
            'timestamp': timestamp.isoformat(),
            'device_id': device_id,
            'location_latlon': f'{lat},{lon}',
            'label': int(fraud)
        })
    return pd.DataFrame(rows)

def main():
    df = generate_transactions()
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/transactions.csv', index=False)
    print('✅ Generated dataset:', df.shape, '→ data/raw/transactions.csv')

if __name__ == '__main__':
    main()
