#!/bin/bash

# Fraud Detection Cambodia - Quick Start Script
echo "🇰🇭 Fraud Detection Cambodia - Quick Start"
echo "=========================================="

# Change to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "🔧 Setting up project..."
    
    # Create virtual environment
    python3 -m venv venv
    
    # Activate and install requirements
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Setup project
    python setup.py
    
    echo "✅ Project setup completed!"
fi

# Run the dashboard
echo "🚀 Starting Fraud Detection Dashboard..."
python app/main.py
