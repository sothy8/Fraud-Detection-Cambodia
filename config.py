# Configuration file for Fraud Detection System

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    fraud_threshold: float = 0.5
    random_state: int = 42

@dataclass
class DataConfig:
    """Data configuration parameters"""
    n_users: int = 500
    n_transactions: int = 20000
    fraud_ratio: float = 0.01
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class PathConfig:
    """File paths configuration"""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    models_dir: str = "models"
    
    raw_transactions: str = "data/raw/transactions.csv"
    processed_features: str = "data/processed/features.csv"
    model_file: str = "models/fraud_model.pkl"

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    title: str = "Fraud Detection Dashboard 🇰🇭"
    layout: str = "wide"
    auto_refresh_interval: int = 30  # seconds
    max_display_transactions: int = 1000

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.paths = PathConfig()
        self.dashboard = DashboardConfig()
        
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.paths.raw_data_dir,
            self.paths.processed_data_dir,
            self.paths.models_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'paths': self.paths.__dict__,
            'dashboard': self.dashboard.__dict__
        }

# Global configuration instance
config = Config()

# Environment-specific overrides
if os.getenv('ENVIRONMENT') == 'production':
    config.data.n_transactions = 100000
    config.model.n_estimators = 500
elif os.getenv('ENVIRONMENT') == 'development':
    config.data.n_transactions = 5000
    config.model.n_estimators = 50

# Feature engineering configuration
FEATURE_COLUMNS = [
    'amount_zscore',
    'time_since_last', 
    'is_new_recipient',
    'is_new_device',
    'hour'
]

# Alert thresholds
ALERT_THRESHOLDS = {
    'high_risk_score': 0.8,
    'high_value_amount': 1000000,  # 1M KHR
    'suspicious_frequency': 10,     # transactions per hour
    'new_device_threshold': 0.7     # fraud score for new devices
}

# Cambodian mobile wallet specific settings
CAMBODIAN_WALLETS = {
    'wing': {'prefix': 'W', 'max_daily_limit': 5000000},
    'pi_pay': {'prefix': 'P', 'max_daily_limit': 3000000},
    'aba_pay': {'prefix': 'A', 'max_daily_limit': 10000000},
    'true_money': {'prefix': 'T', 'max_daily_limit': 2000000}
}

# Transaction type configurations
TRANSACTION_TYPES = {
    'p2p': {'description': 'Peer to Peer Transfer', 'risk_weight': 1.0},
    'merchant': {'description': 'Merchant Payment', 'risk_weight': 0.8},
    'topup': {'description': 'Account Top-up', 'risk_weight': 0.6},
    'withdrawal': {'description': 'Cash Withdrawal', 'risk_weight': 1.2}
}
