#!/bin/bash

# Fraud Detection Cambodia - Quick Start Script
# This script provides easy commands to manage the project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
PYTHON_PATH="$VENV_DIR/bin/python"

# Helper functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
check_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Virtual environment not found at $VENV_DIR"
        print_status "Please create a virtual environment first:"
        print_status "  python -m venv venv"
        print_status "  source venv/bin/activate"
        print_status "  pip install -r requirements.txt"
        exit 1
    fi
}

# Activate virtual environment
activate_venv() {
    check_venv
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
}

# Setup project
setup() {
    print_status "Setting up Fraud Detection Cambodia project..."
    activate_venv
    
    # Install requirements
    print_status "Installing requirements..."
    "$PYTHON_PATH" -m pip install -r requirements.txt
    
    # Run setup script
    print_status "Running project setup..."
    "$PYTHON_PATH" setup.py
    
    print_success "Project setup completed!"
}

# Generate data
generate_data() {
    print_status "Generating sample transaction data..."
    activate_venv
    "$PYTHON_PATH" scripts/simulate_transactions.py
    print_success "Sample data generated!"
}

# Train model
train() {
    print_status "Training fraud detection model..."
    activate_venv
    
    # Check if data exists
    if [ ! -f "data/raw/transactions.csv" ]; then
        print_warning "No raw data found. Generating sample data first..."
        generate_data
    fi
    
    # Feature engineering
    print_status "Engineering features..."
    "$PYTHON_PATH" scripts/feature_engineering.py
    
    # Train model
    print_status "Training model..."
    "$PYTHON_PATH" scripts/train_model.py
    
    print_success "Model training completed!"
}

# Run dashboard
dashboard() {
    print_status "Starting Fraud Detection Dashboard..."
    activate_venv
    
    # Check if model exists
    if [ ! -f "models/fraud_model.pkl" ]; then
        print_warning "No trained model found. Training model first..."
        train
    fi
    
    print_status "Launching Streamlit dashboard at http://localhost:8501"
    "$PYTHON_PATH" -m streamlit run app/main.py
}

# Run tests
test() {
    print_status "Running system tests..."
    activate_venv
    "$PYTHON_PATH" setup.py --test-only
}

# Clean project
clean() {
    print_status "Cleaning project files..."
    
    # Remove generated data
    if [ -d "data" ]; then
        rm -rf data/raw/* data/processed/*
        print_status "Cleaned data directories"
    fi
    
    # Remove models
    if [ -d "models" ]; then
        rm -rf models/*
        print_status "Cleaned models directory"
    fi
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    print_success "Project cleaned!"
}

# Show project status
status() {
    print_status "Fraud Detection Cambodia - Project Status"
    echo ""
    
    # Check virtual environment
    if [ -d "$VENV_DIR" ]; then
        print_success "✓ Virtual environment: Found"
    else
        print_error "✗ Virtual environment: Missing"
    fi
    
    # Check data
    if [ -f "data/raw/transactions.csv" ]; then
        rows=$(wc -l < data/raw/transactions.csv)
        print_success "✓ Sample data: $rows transactions"
    else
        print_warning "⚠ Sample data: Missing"
    fi
    
    # Check processed features
    if [ -f "data/processed/features.csv" ]; then
        print_success "✓ Processed features: Ready"
    else
        print_warning "⚠ Processed features: Missing"
    fi
    
    # Check model
    if [ -f "models/fraud_model.pkl" ]; then
        model_size=$(ls -lh models/fraud_model.pkl | awk '{print $5}')
        print_success "✓ Trained model: $model_size"
    else
        print_warning "⚠ Trained model: Missing"
    fi
    
    echo ""
    print_status "Use './manage.sh help' for available commands"
}

# Show help
help() {
    echo "Fraud Detection Cambodia - Management Script"
    echo ""
    echo "Usage: ./manage.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup      - Complete project setup (install deps, generate data, train model)"
    echo "  data       - Generate sample transaction data"
    echo "  train      - Train the fraud detection model"
    echo "  dashboard  - Start the Streamlit dashboard"
    echo "  run        - Quick start the dashboard (alias for dashboard)"
    echo "  test       - Run system tests"
    echo "  clean      - Clean generated files and cache"
    echo "  status     - Show project status"
    echo "  help       - Show this help message"
    echo ""
    echo "Quick Start Options:"
    echo "  python run_dashboard.py     - Start with launcher"
    echo "  python app/main.py          - Start directly"
    echo "  ./start.sh                  - Auto-setup and start"
    echo ""
    echo "Examples:"
    echo "  ./manage.sh setup      # First time setup"
    echo "  ./manage.sh dashboard  # Start the web interface"
    echo "  ./manage.sh train      # Retrain the model"
    echo ""
}

# Main script logic
case "${1:-help}" in
    setup)
        setup
        ;;
    data)
        generate_data
        ;;
    train)
        train
        ;;
    dashboard|dash|run)
        dashboard
        ;;
    test)
        test
        ;;
    clean)
        clean
        ;;
    status)
        status
        ;;
    help|--help|-h)
        help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        help
        exit 1
        ;;
esac
