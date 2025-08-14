"""
Mamba for Trading - Chapter 126

A complete implementation of Mamba architecture for financial time series
prediction and trading signal generation.

Modules:
    - mamba_model: Core Mamba architecture implementation
    - data_loader: Data fetching from Yahoo Finance and Bybit
    - features: Feature engineering for trading
    - backtest: Backtesting framework
    - train: Training utilities
"""

from .mamba_model import MambaBlock, MambaLayer, MambaTradingModel
from .data_loader import YahooFinanceLoader, BybitDataLoader
from .features import FeatureEngineer
from .backtest import MambaBacktest
from .train import MambaTrainer

__version__ = "1.0.0"
__author__ = "Machine Learning for Trading"

__all__ = [
    "MambaBlock",
    "MambaLayer",
    "MambaTradingModel",
    "YahooFinanceLoader",
    "BybitDataLoader",
    "FeatureEngineer",
    "MambaBacktest",
    "MambaTrainer",
]
