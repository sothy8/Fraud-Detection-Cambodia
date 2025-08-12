"""
Programmatic data generation examples
"""

import sys
import os

# Add the project root and scripts directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = os.path.join(project_root, 'scripts')
sys.path.insert(0, project_root)
sys.path.insert(0, scripts_dir)

from generate_large_dataset import generate_transactions
import pandas as pd

def generate_custom_datasets():
    """Generate various custom datasets"""
    
    print("🎲 Generating custom datasets...")
    
    # Dataset 1: Small and quick
    print("\n📊 Generating small dataset...")
    df_small = generate_transactions(
        n_users=300,
        n_txns=5000,
        fraud_ratio=0.02,
        seed=42,
        days_back=15
    )
    df_small.to_csv('data/datasets/small_5k.csv', index=False)
    
    # Dataset 2: Realistic size
    print("\n📊 Generating realistic dataset...")
    df_medium = generate_transactions(
        n_users=2000,
        n_txns=75000,
        fraud_ratio=0.018,  # 1.8% fraud rate
        seed=123,
        days_back=60
    )
    df_medium.to_csv('data/datasets/realistic_75k.csv', index=False)
    
    # Dataset 3: High volume
    print("\n📊 Generating high volume dataset...")
    df_large = generate_transactions(
        n_users=5000,
        n_txns=150000,
        fraud_ratio=0.012,  # 1.2% fraud rate
        seed=456,
        days_back=120
    )
    df_large.to_csv('data/datasets/high_volume_150k.csv', index=False)
    
    print("\n✅ All custom datasets generated!")
    
    # Analysis
    datasets = {
        'Small (5K)': df_small,
        'Realistic (75K)': df_medium,
        'High Volume (150K)': df_large
    }
    
    print("\n📈 Dataset Analysis:")
    for name, df in datasets.items():
        fraud_count = df['label'].sum()
        fraud_rate = df['label'].mean() * 100
        print(f"  {name}:")
        print(f"    - Total transactions: {len(df):,}")
        print(f"    - Fraudulent: {fraud_count:,} ({fraud_rate:.2f}%)")
        print(f"    - Unique users: {df['user_id'].nunique():,}")
        print(f"    - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"    - Transaction types: {dict(df['txn_type'].value_counts())}")
        print()

def generate_specialized_datasets():
    """Generate datasets for specific testing scenarios"""
    
    print("🔬 Generating specialized test datasets...")
    
    # High fraud rate for algorithm testing
    df_high_fraud = generate_transactions(
        n_users=500,
        n_txns=10000,
        fraud_ratio=0.1,  # 10% fraud rate
        seed=789
    )
    df_high_fraud.to_csv('data/datasets/high_fraud_test.csv', index=False)
    
    # Low fraud rate (realistic production)
    df_low_fraud = generate_transactions(
        n_users=1000,
        n_txns=30000,
        fraud_ratio=0.005,  # 0.5% fraud rate
        seed=101
    )
    df_low_fraud.to_csv('data/datasets/low_fraud_realistic.csv', index=False)
    
    # Many users, fewer transactions each (sparse data)
    df_sparse = generate_transactions(
        n_users=3000,
        n_txns=15000,  # 5 transactions per user on average
        fraud_ratio=0.02,
        seed=202
    )
    df_sparse.to_csv('data/datasets/sparse_users.csv', index=False)
    
    print("✅ Specialized datasets generated!")

if __name__ == '__main__':
    # Create datasets directory
    os.makedirs('data/datasets', exist_ok=True)
    
    # Generate datasets
    generate_custom_datasets()
    generate_specialized_datasets()
    
    # List all generated files
    print("\n📁 All generated datasets:")
    dataset_files = os.listdir('data/datasets')
    for file in sorted(dataset_files):
        if file.endswith('.csv'):
            filepath = f'data/datasets/{file}'
            size = os.path.getsize(filepath) / 1024 / 1024  # MB
            print(f"  - {file}: {size:.1f} MB")
