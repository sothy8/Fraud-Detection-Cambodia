#!/usr/bin/env python3
"""
Enhanced data generation script with configurable parameters
"""

import pandas as pd
import numpy as np
from faker import Faker
import random, os, argparse
from datetime import datetime, timedelta

def generate_transactions(n_users=500, n_txns=50000, fraud_ratio=0.01, seed=42, days_back=30):
    """
    Generate synthetic transaction data
    
    Parameters:
    - n_users: Number of unique users (default: 500)
    - n_txns: Number of transactions (default: 50,000)
    - fraud_ratio: Percentage of fraudulent transactions (default: 1%)
    - seed: Random seed for reproducibility (default: 42)
    - days_back: How many days back to generate data (default: 30)
    """
    fake = Faker()
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate user pool
    users = [f'u{str(i).zfill(4)}' for i in range(n_users)]
    
    # Generate device pool (2x users to allow for multiple devices)
    device_pool = [fake.uuid4() for _ in range(n_users*2)]
    
    # Generate merchant pool
    merchants = [f'm{str(i).zfill(3)}' for i in range(100)]
    
    rows = []
    base_time = datetime.now() - timedelta(days=days_back)
    
    print(f"Generating {n_txns:,} transactions for {n_users} users over {days_back} days...")
    print(f"Expected fraud transactions: {int(n_txns * fraud_ratio):,} ({fraud_ratio*100:.1f}%)")
    
    for i in range(n_txns):
        # Show progress
        if i % 5000 == 0:
            print(f"Progress: {i:,}/{n_txns:,} transactions generated...")
        
        sender = random.choice(users)
        tx_type = random.choice(['p2p', 'merchant', 'topup'])
        
        # Different recipient logic based on transaction type
        if tx_type == 'p2p':
            recipient = random.choice([u for u in users if u != sender])
        elif tx_type == 'merchant':
            recipient = random.choice(merchants)
        else:  # topup
            recipient = f'topup_gateway_{random.randint(1,5)}'
        
        # Generate realistic amounts based on transaction type
        if tx_type == 'p2p':
            amount = round(np.random.gamma(2, 25000), 2)  # Personal transfers
        elif tx_type == 'merchant':
            amount = round(np.random.gamma(1.5, 35000), 2)  # Merchant payments
        else:  # topup
            amount = round(np.random.choice([50000, 100000, 200000, 500000]), 2)  # Standard topup amounts
        
        # Time generation with realistic patterns
        # Business hours are more active
        hour_weights = [0.3, 0.2, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0,
                       1.0, 0.9, 0.8, 0.9, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.5, 0.4]
        hour = np.random.choice(range(24), p=np.array(hour_weights)/sum(hour_weights))
        
        timestamp = base_time + timedelta(
            seconds=random.randint(0, days_back*24*60*60),
            hours=int(hour-base_time.hour)
        )
        
        device_id = random.choice(device_pool)
        
        # Cambodia coordinates (more realistic distribution)
        # Major cities have higher probability
        city_coords = [
            (11.5564, 104.9282),  # Phnom Penh
            (13.3671, 103.8448),  # Siem Reap
            (13.0957, 103.1954),  # Battambang
            (10.6104, 103.5236),  # Sihanoukville
        ]
        
        if random.random() < 0.6:  # 60% chance for major cities
            lat, lon = random.choice(city_coords)
            lat += random.uniform(-0.05, 0.05)  # Small variation
            lon += random.uniform(-0.05, 0.05)
        else:  # 40% chance for rural areas
            lat = round(random.uniform(10.4, 13.6), 6)
            lon = round(random.uniform(102.5, 106.0), 6)
        
        # Determine if transaction is fraudulent
        fraud = np.random.rand() < fraud_ratio
        
        if fraud:
            # Make fraudulent transactions more obvious
            amount *= random.uniform(3, 15)  # Much higher amounts
            device_id = fake.uuid4()  # New device (account takeover)
            
            # Sometimes use unusual timing
            if random.random() < 0.3:
                timestamp = timestamp.replace(hour=random.randint(0, 5))  # Late night
        
        rows.append({
            'txn_id': f't{i:06d}',  # Better formatting: t000001, t000002, etc.
            'user_id': sender,
            'recipient_id': recipient,
            'amount': amount,
            'txn_type': tx_type,
            'timestamp': timestamp.isoformat(),
            'device_id': device_id,
            'location_latlon': f'{lat:.6f},{lon:.6f}',
            'label': int(fraud)
        })
    
    df = pd.DataFrame(rows)
    print(f"\n✅ Generated {len(df):,} transactions")
    print(f"   - Fraudulent: {df['label'].sum():,} ({df['label'].mean()*100:.2f}%)")
    print(f"   - Legitimate: {len(df) - df['label'].sum():,}")
    print(f"   - Transaction types: {dict(df['txn_type'].value_counts())}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic transaction data')
    parser.add_argument('--users', type=int, default=500, help='Number of users (default: 500)')
    parser.add_argument('--transactions', type=int, default=20000, help='Number of transactions (default: 20,000)')
    parser.add_argument('--fraud-ratio', type=float, default=0.01, help='Fraud ratio (default: 0.01 = 1%)')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data (default: 30)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='data/raw/transactions.csv', help='Output file path')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate data
    df = generate_transactions(
        n_users=args.users,
        n_txns=args.transactions,
        fraud_ratio=args.fraud_ratio,
        seed=args.seed,
        days_back=args.days
    )
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f'✅ Dataset saved to: {args.output}')
    
    # Print sample
    print(f'\n📊 Sample data:')
    print(df.head())
    
    return df

if __name__ == '__main__':
    main()
