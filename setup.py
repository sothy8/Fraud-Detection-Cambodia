#!/usr/bin/env python3
"""
Complete setup script for Fraud Detection Cambodia project
This script sets up the entire project pipeline
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_environment():
    """Check if we're in the right Python environment"""
    python_path = sys.executable
    if 'venv' in python_path:
        print(f"✅ Using virtual environment: {python_path}")
        return True
    else:
        print(f"⚠️  Not using virtual environment: {python_path}")
        return False

def install_requirements():
    """Install required packages"""
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        return run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                          "Installing requirements")
    else:
        print("⚠️  requirements.txt not found, installing core packages...")
        packages = [
            "pandas", "numpy", "scikit-learn", "xgboost", "joblib",
            "streamlit", "plotly", "faker", "matplotlib", "seaborn"
        ]
        return run_command(f"{sys.executable} -m pip install {' '.join(packages)}", 
                          "Installing core packages")

def setup_project():
    """Main setup function"""
    print("🚀 Setting up Fraud Detection Cambodia project...")
    
    # Check environment
    check_python_environment()
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create directories
    directories = ["data/raw", "data/processed", "models", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Generate sample data
    if not run_command(f"{sys.executable} scripts/simulate_transactions.py", 
                      "Generating sample transaction data"):
        return False
    
    # Engineer features
    if not run_command(f"{sys.executable} scripts/feature_engineering.py", 
                      "Engineering features"):
        return False
    
    # Train model
    if not run_command(f"{sys.executable} scripts/train_model.py", 
                      "Training fraud detection model"):
        return False
    
    print("\n🎉 Project setup completed successfully!")
    print("🔗 You can now run the dashboard with:")
    print(f"   {sys.executable} -m streamlit run app/main.py")
    
    return True

def quick_test():
    """Run a quick test of the system"""
    print("\n🧪 Running quick system test...")
    
    try:
        # Test model loading
        from app.utils import load_model, check_model_exists
        
        if check_model_exists():
            model = load_model()
            print("✅ Model loads successfully")
            
            # Test with sample data
            import pandas as pd
            if os.path.exists("data/raw/transactions.csv"):
                df = pd.read_csv("data/raw/transactions.csv").head(10)
                from app.utils import score
                result = score(df, model)
                print(f"✅ Scoring works: {len(result)} transactions processed")
            else:
                print("⚠️  No sample data found for testing")
        else:
            print("❌ Model not found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print("✅ All tests passed!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Fraud Detection Cambodia project")
    parser.add_argument("--test-only", action="store_true", help="Run only the test")
    parser.add_argument("--no-test", action="store_true", help="Skip the test after setup")
    
    args = parser.parse_args()
    
    if args.test_only:
        quick_test()
    else:
        success = setup_project()
        if success and not args.no_test:
            quick_test()
