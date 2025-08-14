"""
Feature Engineering Module for Mamba Trading

This module provides feature engineering utilities for creating
trading-relevant features from OHLCV data.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Feature engineering for trading models.

    Creates technical indicators and derived features from OHLCV data.

    Example:
        >>> fe = FeatureEngineer()
        >>> features = fe.compute_all_features(df)
        >>> print(features.columns.tolist())
    """

    def __init__(self, include_volume: bool = True):
        """
        Initialize feature engineer.

        Args:
            include_volume: Whether to include volume-based features
        """
        self.include_volume = include_volume

    def compute_all_features(
        self,
        df: pd.DataFrame,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """
        Compute all available features.

        Args:
            df: DataFrame with OHLCV data
            dropna: Whether to drop NaN values

        Returns:
            DataFrame with all computed features
        """
        features = df.copy()

        # Price-based features
        features = self._add_returns(features)
        features = self._add_moving_averages(features)
        features = self._add_volatility(features)
        features = self._add_momentum(features)
        features = self._add_price_patterns(features)

        # Volume-based features
        if self.include_volume and "volume" in df.columns:
            features = self._add_volume_features(features)

        if dropna:
            features = features.dropna()

        return features

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        # Simple returns
        df["returns"] = df["close"].pct_change()
        df["returns_2"] = df["close"].pct_change(2)
        df["returns_5"] = df["close"].pct_change(5)
        df["returns_10"] = df["close"].pct_change(10)
        df["returns_20"] = df["close"].pct_change(20)

        # Log returns
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Cumulative returns
        df["cum_returns_5"] = df["returns"].rolling(5).sum()
        df["cum_returns_10"] = df["returns"].rolling(10).sum()

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features."""
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            df[f"close_sma_{period}_ratio"] = df["close"] / df[f"sma_{period}"]

        # Exponential Moving Averages
        for period in [12, 26]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # MACD
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        # Historical volatility
        for period in [5, 10, 20]:
            df[f"volatility_{period}"] = df["returns"].rolling(period).std()

        # Average True Range (ATR)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()

        # Bollinger Bands
        sma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features."""
        # RSI (Relative Strength Index)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Stochastic Oscillator
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = (
                (df["close"] - df["close"].shift(period))
                / df["close"].shift(period)
                * 100
            )

        # Momentum
        df["momentum_10"] = df["close"] - df["close"].shift(10)

        return df

    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features."""
        # Candle patterns
        df["body"] = df["close"] - df["open"]
        df["body_pct"] = df["body"] / df["open"]
        df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

        # Doji pattern (small body relative to range)
        range_size = df["high"] - df["low"]
        df["is_doji"] = (df["body"].abs() / range_size) < 0.1

        # Higher highs / Lower lows
        df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
        df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)

        # Gap features
        df["gap"] = df["open"] - df["close"].shift(1)
        df["gap_pct"] = df["gap"] / df["close"].shift(1)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving averages
        for period in [5, 10, 20]:
            df[f"volume_sma_{period}"] = df["volume"].rolling(period).mean()
            df[f"volume_ratio_{period}"] = df["volume"] / df[f"volume_sma_{period}"]

        # Volume rate of change
        df["volume_roc"] = df["volume"].pct_change()

        # On-Balance Volume (OBV)
        df["obv"] = (
            np.sign(df["close"].diff()) * df["volume"]
        ).cumsum()
        df["obv_sma_10"] = df["obv"].rolling(10).mean()

        # Volume-Weighted Average Price (VWAP)
        df["vwap"] = (
            (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        )

        # Money Flow Index (MFI)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_mf = positive_flow.rolling(14).sum()
        negative_mf = negative_flow.rolling(14).sum()
        mf_ratio = positive_mf / negative_mf
        df["mfi_14"] = 100 - (100 / (1 + mf_ratio))

        return df


def create_labels(
    df: pd.DataFrame,
    method: str = "returns",
    threshold: float = 0.02,
    forward_periods: int = 1,
) -> pd.Series:
    """
    Create labels for supervised learning.

    Args:
        df: DataFrame with OHLCV data
        method: Labeling method ('returns', 'direction', 'triple_barrier')
        threshold: Threshold for classification
        forward_periods: Number of periods to look forward

    Returns:
        Series with labels
    """
    if method == "returns":
        # Raw returns as regression target
        labels = df["close"].pct_change(forward_periods).shift(-forward_periods)

    elif method == "direction":
        # Binary classification: up (1) or down (0)
        returns = df["close"].pct_change(forward_periods).shift(-forward_periods)
        labels = (returns > 0).astype(int)

    elif method == "triple_barrier":
        # Three classes: sell (0), hold (1), buy (2)
        returns = df["close"].pct_change(forward_periods).shift(-forward_periods)
        labels = pd.Series(1, index=df.index)  # Default to hold
        labels[returns > threshold] = 2  # Buy
        labels[returns < -threshold] = 0  # Sell

    else:
        raise ValueError(f"Unknown labeling method: {method}")

    return labels


def prepare_sequences(
    features: pd.DataFrame,
    labels: pd.Series,
    lookback: int = 100,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for model training.

    Args:
        features: DataFrame with computed features
        labels: Series with labels
        lookback: Sequence length
        feature_columns: Specific columns to use (None = all)

    Returns:
        Tuple of (X, y) arrays
    """
    # Select feature columns
    if feature_columns is None:
        # Exclude non-numeric and original OHLCV
        exclude = ["open", "high", "low", "close", "volume", "adj_close", "turnover"]
        feature_columns = [
            col for col in features.columns
            if col not in exclude and features[col].dtype in [np.float64, np.int64]
        ]

    X_data = features[feature_columns].values
    y_data = labels.values

    # Handle NaN values
    mask = ~np.isnan(X_data).any(axis=1) & ~np.isnan(y_data)
    X_data = X_data[mask]
    y_data = y_data[mask]

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(X_data)):
        X.append(X_data[i - lookback:i])
        y.append(y_data[i])

    return np.array(X), np.array(y)


def normalize_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    method: str = "zscore",
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """
    Normalize features using training set statistics.

    Args:
        X_train: Training data of shape (n_samples, seq_len, n_features)
        X_test: Optional test data to normalize
        method: Normalization method ('zscore', 'minmax')

    Returns:
        Tuple of (X_train_normalized, X_test_normalized, stats)
    """
    # Compute statistics on training data (flatten to 2D)
    n_samples, seq_len, n_features = X_train.shape
    X_flat = X_train.reshape(-1, n_features)

    if method == "zscore":
        mean = X_flat.mean(axis=0)
        std = X_flat.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        stats = {"mean": mean, "std": std}

        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std if X_test is not None else None

    elif method == "minmax":
        min_val = X_flat.min(axis=0)
        max_val = X_flat.max(axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        stats = {"min": min_val, "max": max_val, "range": range_val}

        X_train_norm = (X_train - min_val) / range_val
        X_test_norm = (X_test - min_val) / range_val if X_test is not None else None

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_train_norm, X_test_norm, stats
