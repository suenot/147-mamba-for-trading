"""
Data Loader Module for Mamba Trading

This module provides data loading utilities for both traditional stock markets
(via Yahoo Finance) and cryptocurrency markets (via Bybit API).

All data fetching and processing is contained within this chapter's folder.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests


class YahooFinanceLoader:
    """
    Load historical stock data from Yahoo Finance.

    Example:
        >>> loader = YahooFinanceLoader()
        >>> df = loader.fetch("AAPL", period="1y")
        >>> print(df.head())
    """

    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def fetch(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical data for a symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1d', '1wk', '1mo')

        Returns:
            DataFrame with columns: open, high, low, close, volume, adj_close
        """
        params = {
            "period1": 0,
            "period2": int(time.time()),
            "interval": interval,
            "range": period,
        }

        url = f"{self.base_url}/{symbol}"

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]
            adj_close = result["indicators"]["adjclose"][0]["adjclose"]

            df = pd.DataFrame({
                "timestamp": pd.to_datetime(timestamps, unit="s"),
                "open": quotes["open"],
                "high": quotes["high"],
                "low": quotes["low"],
                "close": quotes["close"],
                "volume": quotes["volume"],
                "adj_close": adj_close,
            })

            df.set_index("timestamp", inplace=True)
            df.dropna(inplace=True)

            return df

        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {e}")

    def fetch_multiple(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            period: Data period
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.fetch(symbol, period, interval)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"Warning: Could not fetch {symbol}: {e}")

        return data


class BybitDataLoader:
    """
    Load cryptocurrency data from Bybit exchange.

    Supports both spot and linear perpetual contracts.

    Example:
        >>> loader = BybitDataLoader()
        >>> df = loader.fetch_klines("BTCUSDT", interval="60", limit=1000)
        >>> print(df.head())
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 1000,
        category: str = "linear",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Bybit.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Kline interval in minutes ('1', '5', '15', '60', '240', 'D', 'W')
            limit: Number of klines to fetch (max 1000)
            category: Market category ('linear', 'inverse', 'spot')
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, turnover
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"

        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            if data["retCode"] != 0:
                raise ValueError(f"Bybit API error: {data['retMsg']}")

            records = data["result"]["list"]

            df = pd.DataFrame(records, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = df[col].astype(float)

            # Sort by time (Bybit returns descending)
            df = df.sort_values("timestamp").reset_index(drop=True)
            df.set_index("timestamp", inplace=True)

            return df

        except Exception as e:
            raise ValueError(f"Failed to fetch Bybit data for {symbol}: {e}")

    def fetch_extended(
        self,
        symbol: str,
        interval: str = "60",
        days: int = 30,
        category: str = "linear",
    ) -> pd.DataFrame:
        """
        Fetch extended historical data by making multiple API calls.

        Args:
            symbol: Trading pair
            interval: Kline interval
            days: Number of days of data to fetch
            category: Market category

        Returns:
            DataFrame with extended historical data
        """
        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)

        # Calculate interval in milliseconds
        interval_ms = self._interval_to_ms(interval)
        records_per_call = 1000
        ms_per_call = interval_ms * records_per_call

        # Calculate number of calls needed
        total_ms = days * 24 * 60 * 60 * 1000
        n_calls = int(np.ceil(total_ms / ms_per_call))

        for i in range(n_calls):
            try:
                df = self.fetch_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=1000,
                    category=category,
                    end_time=end_time,
                )

                if df.empty:
                    break

                all_data.append(df)
                end_time = int(df.index.min().timestamp() * 1000) - 1
                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                print(f"Warning: Error fetching batch {i}: {e}")
                break

        if not all_data:
            raise ValueError(f"No data fetched for {symbol}")

        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep="first")]
        result = result.sort_index()

        return result

    def _interval_to_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds."""
        interval_map = {
            "1": 60 * 1000,
            "3": 3 * 60 * 1000,
            "5": 5 * 60 * 1000,
            "15": 15 * 60 * 1000,
            "30": 30 * 60 * 1000,
            "60": 60 * 60 * 1000,
            "120": 2 * 60 * 60 * 1000,
            "240": 4 * 60 * 60 * 1000,
            "360": 6 * 60 * 60 * 1000,
            "720": 12 * 60 * 60 * 1000,
            "D": 24 * 60 * 60 * 1000,
            "W": 7 * 24 * 60 * 60 * 1000,
            "M": 30 * 24 * 60 * 60 * 1000,
        }
        return interval_map.get(interval, 60 * 60 * 1000)

    def get_tickers(self, category: str = "linear") -> pd.DataFrame:
        """
        Get all available tickers.

        Args:
            category: Market category

        Returns:
            DataFrame with ticker information
        """
        endpoint = f"{self.BASE_URL}/v5/market/tickers"
        params = {"category": category}

        response = self.session.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        if data["retCode"] != 0:
            raise ValueError(f"Bybit API error: {data['retMsg']}")

        return pd.DataFrame(data["result"]["list"])


class DataManager:
    """
    Unified data manager for multiple data sources.

    Provides a consistent interface for fetching and caching data
    from different sources.

    Example:
        >>> manager = DataManager()
        >>> stock_data = manager.get_stock_data("AAPL", period="1y")
        >>> crypto_data = manager.get_crypto_data("BTCUSDT", days=30)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.yahoo_loader = YahooFinanceLoader()
        self.bybit_loader = BybitDataLoader()
        self.cache_dir = cache_dir
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_stock_data(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get stock data with caching.

        Args:
            symbol: Stock ticker symbol
            period: Data period
            interval: Data interval
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"stock_{symbol}_{period}_{interval}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        df = self.yahoo_loader.fetch(symbol, period, interval)

        if use_cache:
            self._cache[cache_key] = df

        return df

    def get_crypto_data(
        self,
        symbol: str,
        interval: str = "60",
        days: int = 30,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get cryptocurrency data with caching.

        Args:
            symbol: Trading pair
            interval: Kline interval
            days: Number of days
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"crypto_{symbol}_{interval}_{days}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if days <= 1:
            df = self.bybit_loader.fetch_klines(symbol, interval)
        else:
            df = self.bybit_loader.fetch_extended(symbol, interval, days)

        if use_cache:
            self._cache[cache_key] = df

        return df

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()


def prepare_dataset(
    df: pd.DataFrame,
    lookback: int = 100,
    target_column: str = "close",
    normalize: bool = True,
) -> tuple:
    """
    Prepare dataset for Mamba model training.

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of timesteps to look back
        target_column: Column to predict
        normalize: Whether to normalize the data

    Returns:
        Tuple of (X, y) arrays ready for training
    """
    # Ensure we have the required columns
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create feature matrix
    features = df[required].values

    if normalize:
        # Normalize each feature independently
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        features = (features - mean) / std

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback:i])
        # Target: next period's return
        if i < len(df):
            current_close = df[target_column].iloc[i - 1]
            next_close = df[target_column].iloc[i]
            ret = (next_close - current_close) / current_close
            y.append(ret)

    return np.array(X), np.array(y)
