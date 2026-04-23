"""
Trading Strategies Module
Defines rules for when to buy and sell
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from enum import Enum


class Signal(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class TradingStrategies:
    """
    Collection of trading strategies.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize strategy with data that has indicators.

        Args:
            df: DataFrame with price and indicator columns
        """
        self.df = df

    def _get_latest_value(self, column: str) -> float:
        """
        Safely get the latest value from a column as a scalar.
        """
        val = self.df[column].iloc[-1]
        # If it's a Series, extract the scalar value
        if isinstance(val, (pd.Series, np.ndarray)):
            val = val.iloc[0] if hasattr(val, 'iloc') else val[0]
        return float(val)

    # ================================================================
    # STRATEGY 1: RSI Mean Reversion
    # ================================================================

    def rsi_mean_reversion(self) -> Tuple[Signal, float, str]:
        """
        RSI Mean Reversion Strategy

        Logic:
        - RSI < 30 = Oversold → BUY
        - RSI > 70 = Overbought → SELL
        """
        if 'RSI' not in self.df.columns:
            return Signal.HOLD, 0, "RSI not available"

        current_rsi = self._get_latest_value('RSI')

        if current_rsi < 30:
            oversold_amount = 30 - current_rsi
            confidence = min(100, 50 + oversold_amount * 2)
            return Signal.BUY, confidence, f"RSI oversold at {current_rsi:.1f}"

        elif current_rsi > 70:
            overbought_amount = current_rsi - 70
            confidence = min(100, 50 + overbought_amount * 2)
            return Signal.SELL, confidence, f"RSI overbought at {current_rsi:.1f}"

        else:
            return Signal.HOLD, 0, f"RSI neutral at {current_rsi:.1f}"

    # ================================================================
    # STRATEGY 2: MACD Crossover
    # ================================================================

    def macd_crossover(self) -> Tuple[Signal, float, str]:
        """
        MACD Crossover Strategy
        """
        required = ['MACD', 'MACD_Signal']
        if not all(col in self.df.columns for col in required):
            return Signal.HOLD, 0, "MACD not available"

        current_macd = self._get_latest_value('MACD')
        current_signal = self._get_latest_value('MACD_Signal')
        prev_macd = float(self.df['MACD'].iloc[-2])
        prev_signal = float(self.df['MACD_Signal'].iloc[-2])

        # Check for crossover
        if prev_macd <= prev_signal and current_macd > current_signal:
            return Signal.BUY, 70, "MACD bullish crossover"

        elif prev_macd >= prev_signal and current_macd < current_signal:
            return Signal.SELL, 70, "MACD bearish crossover"

        else:
            return Signal.HOLD, 0, "No MACD crossover"

    # ================================================================
    # STRATEGY 3: Moving Average Crossover
    # ================================================================

    def ma_crossover(self, fast: int = 20, slow: int = 50) -> Tuple[Signal, float, str]:
        """
        Moving Average Crossover Strategy
        """
        fast_col = f'SMA_{fast}'
        slow_col = f'SMA_{slow}'

        if fast_col not in self.df.columns or slow_col not in self.df.columns:
            return Signal.HOLD, 0, f"MA {fast}/{slow} not available"

        current_fast = self._get_latest_value(fast_col)
        current_slow = self._get_latest_value(slow_col)
        prev_fast = float(self.df[fast_col].iloc[-2])
        prev_slow = float(self.df[slow_col].iloc[-2])

        # Check for crossover
        if prev_fast <= prev_slow and current_fast > current_slow:
            return Signal.BUY, 75, f"Golden Cross: SMA{fast} crossed above SMA{slow}"

        elif prev_fast >= prev_slow and current_fast < current_slow:
            return Signal.SELL, 75, f"Death Cross: SMA{fast} crossed below SMA{slow}"

        else:
            if current_fast > current_slow:
                return Signal.HOLD, 30, f"Uptrend"
            else:
                return Signal.HOLD, 30, f"Downtrend"

    # ================================================================
    # STRATEGY 4: Bollinger Band Reversion
    # ================================================================

    def bollinger_reversion(self) -> Tuple[Signal, float, str]:
        """
        Bollinger Band Reversion Strategy
        """
        if not all(col in self.df.columns for col in ['BB_Upper', 'BB_Lower']):
            return Signal.HOLD, 0, "Bollinger Bands not available"

        current_price = self._get_latest_value('Close')
        upper_band = self._get_latest_value('BB_Upper')
        lower_band = self._get_latest_value('BB_Lower')

        if current_price <= lower_band:
            confidence = 65
            return Signal.BUY, confidence, f"Price touched lower Bollinger Band"

        elif current_price >= upper_band:
            confidence = 65
            return Signal.SELL, confidence, f"Price touched upper Bollinger Band"

        else:
            return Signal.HOLD, 0, "Price within Bollinger Bands"

    # ================================================================
    # STRATEGY 5: Combined Strategy
    # ================================================================

    def combined_strategy(self, aggressive: bool = True) -> Tuple[Signal, float, Dict[str, Any]]:
        """
        Combined Strategy - Aggregates signals from all strategies.

        Args:
            aggressive: If True, acts on any non-HOLD signal
        """
        signals = []
        confidences = []
        reasons = []

        # RSI Strategy
        s, c, r = self.rsi_mean_reversion()
        signals.append(s)
        confidences.append(c)
        reasons.append(r)

        # MACD Strategy
        s, c, r = self.macd_crossover()
        signals.append(s)
        confidences.append(c)
        reasons.append(r)

        # MA Crossover Strategy
        s, c, r = self.ma_crossover()
        signals.append(s)
        confidences.append(c)
        reasons.append(r)

        # Bollinger Strategy
        s, c, r = self.bollinger_reversion()
        signals.append(s)
        confidences.append(c)
        reasons.append(r)

        # Count votes
        buy_votes = signals.count(Signal.BUY)
        sell_votes = signals.count(Signal.SELL)
        hold_votes = signals.count(Signal.HOLD)

        if aggressive:
            # AGGRESSIVE MODE: Act on ANY non-HOLD signal
            if buy_votes > 0 and buy_votes >= sell_votes:
                final_signal = Signal.BUY
                avg_confidence = max(confidences) if confidences else 0
                avg_confidence = min(100, avg_confidence + (buy_votes * 10))
            elif sell_votes > 0 and sell_votes > buy_votes:
                final_signal = Signal.SELL
                avg_confidence = max(confidences) if confidences else 0
                avg_confidence = min(100, avg_confidence + (sell_votes * 10))
            else:
                final_signal = Signal.HOLD
                avg_confidence = 0
        else:
            # CONSERVATIVE MODE: Require majority
            if buy_votes > sell_votes and buy_votes > hold_votes:
                final_signal = Signal.BUY
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                avg_confidence = min(100, avg_confidence + (buy_votes * 10))
            elif sell_votes > buy_votes and sell_votes > hold_votes:
                final_signal = Signal.SELL
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                avg_confidence = min(100, avg_confidence + (sell_votes * 10))
            else:
                final_signal = Signal.HOLD
                avg_confidence = 0

        return final_signal, avg_confidence, {
            "buy_votes": buy_votes,
            "sell_votes": sell_votes,
            "hold_votes": hold_votes,
            "individual_signals": list(zip(reasons, signals, confidences))
        }


def get_trading_decision(df: pd.DataFrame, aggressive: bool = True) -> Dict[str, Any]:
    """
    Get trading decision based on all strategies.

    Args:
        df: DataFrame with indicators
        aggressive: If True, trades on any signal (more active)
    """
    strategies = TradingStrategies(df)
    signal, confidence, details = strategies.combined_strategy(aggressive=aggressive)

    return {
        "signal": signal.value,
        "confidence": confidence,
        "action": signal.value,
        "details": details
    }


if __name__ == "__main__":
    # Test with sample data
    print("Testing Trading Strategies...")

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
    })

    # Add indicators
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from src.indicators.technical import TechnicalIndicators

    indicators = TechnicalIndicators()
    df = indicators.add_all_indicators(df)

    # Get trading decision
    decision = get_trading_decision(df)

    print(f"\n📊 Trading Decision:")
    print(f"   Signal: {decision['signal']}")
    print(f"   Confidence: {decision['confidence']:.1f}%")
    print(f"   Details: Buy:{decision['details']['buy_votes']} | Sell:{decision['details']['sell_votes']} | Hold:{decision['details']['hold_votes']}")

    print("\nTrading Strategies Module Ready!")