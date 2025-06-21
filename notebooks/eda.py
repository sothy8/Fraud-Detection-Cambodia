import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('data/raw/transactions.csv', parse_dates=['timestamp'])

sns.histplot(df['amount'], bins=60)
plt.title('Transaction Amount Distribution')
plt.savefig('data/amount_distribution.png')

df['timestamp'].dt.hour.value_counts().sort_index().plot(kind='bar')
plt.title('Transactions by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.savefig('data/transactions_by_hour.png')

print('✅ EDA plots saved in data/')
