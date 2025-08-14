"""
Backtesting Framework for Mamba Trading

This module provides a backtesting framework for evaluating
Mamba-based trading strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class Trade:
    """Represents a single trade."""

    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # 'long' or 'short'
    size: float
    confidence: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    trades: List[Trade]
    equity_curve: List[float]
    timestamps: List[pd.Timestamp]
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    avg_holding_period: float
    metrics: Dict = field(default_factory=dict)


class MambaBacktest:
    """
    Backtesting engine for Mamba trading strategies.

    Supports:
    - Long and short positions
    - Transaction costs
    - Position sizing
    - Stop-loss and take-profit orders

    Example:
        >>> backtest = MambaBacktest(model, initial_capital=100000)
        >>> result = backtest.run(df, features, confidence_threshold=0.6)
        >>> print(f"Total Return: {result.total_return:.2f}%")
    """

    def __init__(
        self,
        model: torch.nn.Module,
        initial_capital: float = 100000,
        position_size: float = 1.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        allow_short: bool = False,
    ):
        """
        Initialize backtester.

        Args:
            model: Trained Mamba model
            initial_capital: Starting capital
            position_size: Fraction of capital per trade (0-1)
            transaction_cost: Transaction cost as fraction
            slippage: Slippage as fraction
            allow_short: Whether to allow short selling
        """
        self.model = model
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.allow_short = allow_short

        # State
        self.capital = initial_capital
        self.position = 0.0
        self.position_direction = None
        self.entry_price = 0.0
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[pd.Timestamp] = []

    def reset(self):
        """Reset backtest state."""
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_direction = None
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = []
        self.timestamps = []

    def run(
        self,
        df: pd.DataFrame,
        features: torch.Tensor,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.6,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            df: DataFrame with OHLCV data
            features: Model input features (batch, seq_len, n_features)
            buy_threshold: Confidence threshold for buy signals
            sell_threshold: Confidence threshold for sell signals
            stop_loss: Stop-loss percentage (e.g., 0.05 for 5%)
            take_profit: Take-profit percentage

        Returns:
            BacktestResult with all metrics
        """
        self.reset()

        # Get model predictions
        self.model.eval()
        signals, probs = self.model.generate_signals(
            features,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )

        # Run simulation
        for i, (signal, confidence) in enumerate(signals):
            if i >= len(df):
                break

            timestamp = df.index[i]
            price = df["close"].iloc[i]

            # Check stop-loss / take-profit
            if self.position != 0:
                pnl_pct = self._calculate_unrealized_pnl_pct(price)

                if stop_loss and pnl_pct <= -stop_loss:
                    self._close_position(timestamp, price, "stop_loss")
                elif take_profit and pnl_pct >= take_profit:
                    self._close_position(timestamp, price, "take_profit")

            # Process signal
            if signal == "BUY" and self.position <= 0:
                if self.position < 0:  # Close short first
                    self._close_position(timestamp, price, "signal")
                self._open_long(timestamp, price, confidence)

            elif signal == "SELL":
                if self.position > 0:  # Close long
                    self._close_position(timestamp, price, "signal")
                elif self.position == 0 and self.allow_short:
                    self._open_short(timestamp, price, confidence)

            # Record equity
            equity = self._calculate_equity(price)
            self.equity_curve.append(equity)
            self.timestamps.append(timestamp)

        # Close any remaining position
        if self.position != 0:
            final_price = df["close"].iloc[-1]
            self._close_position(df.index[-1], final_price, "end_of_backtest")

        return self._calculate_results(df)

    def _open_long(self, timestamp: pd.Timestamp, price: float, confidence: float):
        """Open a long position."""
        trade_capital = self.capital * self.position_size
        cost = trade_capital * (self.transaction_cost + self.slippage)
        entry_price = price * (1 + self.slippage)

        shares = (trade_capital - cost) / entry_price

        self.position = shares
        self.position_direction = "long"
        self.entry_price = entry_price
        self.capital -= trade_capital

        self.trades.append(Trade(
            entry_time=timestamp,
            entry_price=entry_price,
            direction="long",
            size=shares,
            confidence=confidence,
        ))

    def _open_short(self, timestamp: pd.Timestamp, price: float, confidence: float):
        """Open a short position."""
        trade_capital = self.capital * self.position_size
        cost = trade_capital * (self.transaction_cost + self.slippage)
        entry_price = price * (1 - self.slippage)

        shares = (trade_capital - cost) / entry_price

        self.position = -shares
        self.position_direction = "short"
        self.entry_price = entry_price

        self.trades.append(Trade(
            entry_time=timestamp,
            entry_price=entry_price,
            direction="short",
            size=shares,
            confidence=confidence,
        ))

    def _close_position(self, timestamp: pd.Timestamp, price: float, reason: str):
        """Close current position."""
        if self.position == 0:
            return

        if self.position_direction == "long":
            exit_price = price * (1 - self.slippage)
            proceeds = abs(self.position) * exit_price
            cost = proceeds * self.transaction_cost
            pnl = proceeds - cost - (abs(self.position) * self.entry_price)
        else:  # short
            exit_price = price * (1 + self.slippage)
            proceeds = abs(self.position) * self.entry_price
            cost = (abs(self.position) * exit_price) * self.transaction_cost
            pnl = proceeds - (abs(self.position) * exit_price) - cost

        pnl_pct = pnl / (abs(self.position) * self.entry_price)

        # Update last trade
        if self.trades:
            self.trades[-1].exit_time = timestamp
            self.trades[-1].exit_price = exit_price
            self.trades[-1].pnl = pnl
            self.trades[-1].pnl_pct = pnl_pct

        # Update capital
        if self.position_direction == "long":
            self.capital += proceeds - cost
        else:
            self.capital += pnl

        self.position = 0
        self.position_direction = None
        self.entry_price = 0

    def _calculate_unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.position == 0:
            return 0.0

        if self.position_direction == "long":
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity."""
        if self.position == 0:
            return self.capital

        if self.position_direction == "long":
            position_value = abs(self.position) * current_price
        else:
            # Short position value
            position_value = (
                abs(self.position) * self.entry_price
                + abs(self.position) * (self.entry_price - current_price)
            )

        return self.capital + position_value

    def _calculate_results(self, df: pd.DataFrame) -> BacktestResult:
        """Calculate all backtest metrics."""
        equity = np.array(self.equity_curve)

        # Returns
        returns = np.diff(equity) / equity[:-1]
        total_return = (equity[-1] / self.initial_capital - 1) * 100

        # Annualized return (assuming daily data)
        n_days = len(equity)
        annual_return = ((equity[-1] / self.initial_capital) ** (252 / n_days) - 1) * 100

        # Sharpe Ratio
        excess_returns = returns - 0.02 / 252  # Risk-free rate
        sharpe = np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-10)

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino = (
            np.sqrt(252) * returns.mean() / (downside_returns.std() + 1e-10)
            if len(downside_returns) > 0
            else 0
        )

        # Max Drawdown
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            max_dd = max(max_dd, dd)

        # Trade statistics
        completed_trades = [t for t in self.trades if t.pnl is not None]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl < 0]

        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        total_profits = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profits / total_losses if total_losses > 0 else float("inf")

        avg_pnl = np.mean([t.pnl for t in completed_trades]) if completed_trades else 0

        # Average holding period
        holding_periods = []
        for t in completed_trades:
            if t.exit_time and t.entry_time:
                holding_periods.append((t.exit_time - t.entry_time).total_seconds() / 3600)
        avg_holding = np.mean(holding_periods) if holding_periods else 0

        return BacktestResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            timestamps=self.timestamps,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd * 100,
            win_rate=win_rate * 100,
            profit_factor=profit_factor,
            total_trades=len(completed_trades),
            avg_trade_pnl=avg_pnl,
            avg_holding_period=avg_holding,
        )


def plot_backtest_results(result: BacktestResult, save_path: Optional[str] = None):
    """
    Plot backtest results.

    Args:
        result: BacktestResult object
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Equity curve
    ax1 = axes[0]
    ax1.plot(result.timestamps, result.equity_curve, "b-", linewidth=1)
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    equity = np.array(result.equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100
    ax2.fill_between(result.timestamps, drawdown, 0, color="red", alpha=0.3)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    # Trade P&L distribution
    ax3 = axes[2]
    pnls = [t.pnl for t in result.trades if t.pnl is not None]
    if pnls:
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax3.set_title("Trade P&L")
        ax3.set_xlabel("Trade #")
        ax3.set_ylabel("P&L ($)")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def print_backtest_summary(result: BacktestResult):
    """Print a formatted summary of backtest results."""
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Return:         {result.total_return:>10.2f}%")
    print(f"Annual Return:        {result.annual_return:>10.2f}%")
    print(f"Sharpe Ratio:         {result.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:        {result.sortino_ratio:>10.2f}")
    print(f"Max Drawdown:         {result.max_drawdown:>10.2f}%")
    print("-" * 50)
    print(f"Total Trades:         {result.total_trades:>10d}")
    print(f"Win Rate:             {result.win_rate:>10.2f}%")
    print(f"Profit Factor:        {result.profit_factor:>10.2f}")
    print(f"Avg Trade P&L:        ${result.avg_trade_pnl:>9.2f}")
    print(f"Avg Holding (hours):  {result.avg_holding_period:>10.2f}")
    print("=" * 50 + "\n")
