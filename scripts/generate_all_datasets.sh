#!/bin/bash

# Generate multiple datasets with different characteristics
echo "🎲 Generating multiple datasets for fraud detection..."

PYTHON_PATH="/Users/vandethsothy/Documents/Important_Projects/Fraud-Detection-Cambodia/venv/bin/python"
SCRIPT="scripts/generate_large_dataset.py"

# Create datasets directory
mkdir -p data/datasets

echo "📊 1. Small dataset (for quick testing)"
$PYTHON_PATH $SCRIPT --transactions 10000 --users 500 --output data/datasets/small_dataset.csv

echo "📊 2. Medium dataset (for development)"
$PYTHON_PATH $SCRIPT --transactions 50000 --users 1500 --fraud-ratio 0.02 --output data/datasets/medium_dataset.csv

echo "📊 3. Large dataset (for training)"
$PYTHON_PATH $SCRIPT --transactions 100000 --users 3000 --fraud-ratio 0.015 --days 90 --output data/datasets/large_dataset.csv

echo "📊 4. High fraud dataset (for testing edge cases)"
$PYTHON_PATH $SCRIPT --transactions 25000 --users 800 --fraud-ratio 0.05 --output data/datasets/high_fraud_dataset.csv

echo "📊 5. Historical dataset (6 months)"
$PYTHON_PATH $SCRIPT --transactions 75000 --users 2000 --days 180 --output data/datasets/historical_dataset.csv

echo "✅ All datasets generated successfully!"
echo "📁 Available datasets:"
ls -lh data/datasets/

echo ""
echo "🎯 How to use:"
echo "1. Copy any dataset to data/raw/transactions.csv"
echo "2. Run: ./manage.sh train"
echo "3. Run: ./manage.sh dashboard"
