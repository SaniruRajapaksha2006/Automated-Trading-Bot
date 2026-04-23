"""
Backtesting Engine
Tests trading strategies on historical data
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.strategies.trading_strategies import get_trading_decision


class BacktestEngine:
    """
    Backtest trading strategies on historical data.

    - Testing strategy on PAST data
    - Simulates trades as if they happened in the past
    - Shows how strategy would have performed

    Why it matters:
    - Validate strategy before using real money
    - Understand risk and drawdowns
    - Optimize parameters
    """

    def __init__(self, initial_capital: float = 10000):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting account balance
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []  # Open positions
        self.trades = []  # Closed trades
        self.equity_curve = []

    def run(self, df: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data.

        Args:
            df: DataFrame with OHLCV and indicators

        Returns:
            Dictionary with backtest results
        """
        print(f"Running backtest with ${self.initial_capital:,.2f} initial capital...")

        for i in range(50, len(df)):  # Start after indicators are calculated
            current_df = df.iloc[:i + 1]
            current_price = current_df['Close'].iloc[-1]

            # Get trading signal
            decision = get_trading_decision(current_df)
            signal = decision['signal']
            confidence = decision['confidence']

            # Only trade if confidence > 50%
            if confidence < 30:
                continue

            # Execute trades based on signal
            if signal == "BUY" and len(self.positions) == 0:
                # Buy
                position = {
                    'entry_price': current_price,
                    'entry_date': current_df.index[-1],
                    'shares': self.capital * 0.95 / current_price  # Use 95% of capital
                }
                self.positions.append(position)
                self.capital -= position['shares'] * current_price

            elif signal == "SELL" and len(self.positions) > 0:
                # Sell
                for pos in self.positions:
                    pnl = pos['shares'] * (current_price - pos['entry_price'])
                    self.capital += pos['shares'] * current_price

                    self.trades.append({
                        'entry_date': pos['entry_date'],
                        'exit_date': current_df.index[-1],
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'pnl_pct': (pnl / (pos['shares'] * pos['entry_price'])) * 100
                    })

                self.positions = []

            # Track equity
            equity = self.capital
            for pos in self.positions:
                equity += pos['shares'] * current_price
            self.equity_curve.append(equity)

        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics.

        Returns:
            Dictionary with metrics:
            - total_return: Overall profit/loss %
            - total_trades: Number of trades
            - win_rate: % of profitable trades
            - avg_win: Average winning trade
            - avg_loss: Average losing trade
            - max_drawdown: Largest peak-to-trough decline
            - sharpe_ratio: Risk-adjusted return
        """
        if not self.trades:
            return {"error": "No trades executed"}

        trades_df = pd.DataFrame(self.trades)

        total_return = ((self.equity_curve[-1] - self.initial_capital) / self.initial_capital) * 100

        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]

        win_rate = (len(winning_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0

        # Calculate drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        return {
            "total_return": total_return,
            "final_equity": self.equity_curve[-1],
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_win": winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            "avg_loss": losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            "max_drawdown": max_drawdown,
            "profit_factor": abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and
                                                                                              losing_trades[
                                                                                                  'pnl'].sum() != 0 else 0
        }


if __name__ == "__main__":
    # Test backtest with sample data
    import yfinance as yf

    print("Testing Backtest Engine...")

    # Download sample data
    df = yf.download("NVDA", start="2024-01-01", end="2024-12-31")
    df = df.reset_index()

    # Add indicators
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from src.indicators.technical import TechnicalIndicators

    indicators = TechnicalIndicators()
    df = indicators.add_all_indicators(df)

    # Run backtest
    backtest = BacktestEngine(initial_capital=10000)
    results = backtest.run(df)

    print("\nBacktest Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"   {key}: ${value:,.2f}" if 'equity' in key or 'capital' in key else f"   {key}: {value:.2f}%")
        else:
            print(f"   {key}: {value}")

    print("\nBacktest Engine Ready!")