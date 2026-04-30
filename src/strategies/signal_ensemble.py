"""
Signal Ensemble Module
Combines multiple strategies with intelligent weighting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
from enum import Enum

# Now local imports work
from src.strategies.strategies import CompleteStrategies, Signal
from src.indicators.technical import TechnicalIndicators
from src.indicators.market_regime import MarketRegime


class SignalEnsemble:
    """
    Advanced signal combiner with:
    - Dynamic strategy weights
    - Market regime adjustment
    - Confidence scoring
    - Risk adjustment
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with market data.

        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df
        self.strategies = CompleteStrategies(df)
        self.regime = MarketRegime(df)

        # Strategy weights (can be adjusted)
        self.base_weights = {
            # Trend Following (higher weight in trending markets)
            "MA_Crossover": 1.0,
            "EMA_Trend": 0.8,
            "ADX_Trend": 1.2,

            # Momentum (medium weight)
            "RSI_Extreme": 0.7,
            "MACD_Histogram": 0.9,
            "CCI": 0.6,

            # Mean Reversion (higher weight in ranging markets)
            "Bollinger": 0.5,
            "Z_Score": 0.5,
            "Volume_Breakout": 0.6,

            # Volume (confirmation)
            "MFI_Extreme": 0.6,
            "Chaikin": 0.5,

            # Breakout strategies
            "Keltner_Breakout": 0.7,
            "Donchian": 0.7,
            "PSAR": 0.6,
        }

        # Store signals history
        self.signal_history = []

    def get_regime_multiplier(self) -> float:
        """
        Get weight multiplier based on market regime.

        Returns:
            Multiplier for strategy weights
        """
        regime_result = self.regime.detect_regime()
        regime = regime_result['regime']

        if regime == "BULLISH":
            return 1.3  # Increase confidence in trending strategies
        elif regime == "BEARISH":
            return 0.7  # Decrease confidence
        else:
            return 1.0  # Neutral

    def get_all_signals(self) -> List[Dict]:
        """
        Get all strategy signals with weights.

        Returns:
            List of signal dictionaries
        """
        all_signals = []

        # Get signals from all strategies
        for strategy_name, weight in self.base_weights.items():
            # Get signal using reflection
            method_name = f"strategy_{strategy_name.lower()}"
            method = getattr(self.strategies, method_name, None)

            if method:
                try:
                    signal, confidence, reason = method()
                    all_signals.append({
                        "strategy": strategy_name,
                        "signal": signal.value,
                        "confidence": confidence,
                        "weight": weight,
                        "reason": reason
                    })
                except Exception as e:
                    # Skip strategies that fail
                    pass

        return all_signals

    def calculate_ensemble_signal(self) -> Dict:
        """
        Calculate weighted ensemble signal.

        Returns:
            Dictionary with final signal and details
        """
        all_signals = self.get_all_signals()
        regime_multiplier = self.get_regime_multiplier()

        # Calculate weighted scores
        buy_score = 0
        sell_score = 0
        total_weight = 0

        active_strategies = []

        for s in all_signals:
            if s['confidence'] > 0:  # Only consider active signals
                adjusted_weight = s['weight'] * regime_multiplier
                total_weight += adjusted_weight

                if s['signal'] == "BUY":
                    buy_score += adjusted_weight * (s['confidence'] / 100)
                    active_strategies.append((s['strategy'], "BUY", s['confidence']))
                elif s['signal'] == "SELL":
                    sell_score += adjusted_weight * (s['confidence'] / 100)
                    active_strategies.append((s['strategy'], "SELL", s['confidence']))

        # Determine final signal
        if total_weight > 0:
            buy_ratio = buy_score / total_weight
            sell_ratio = sell_score / total_weight

            if buy_ratio > 0.6:
                final_signal = "BUY"
                confidence = min(95, int(buy_ratio * 100))
            elif sell_ratio > 0.6:
                final_signal = "SELL"
                confidence = min(95, int(sell_ratio * 100))
            else:
                final_signal = "HOLD"
                confidence = 0
        else:
            final_signal = "HOLD"
            confidence = 0
            buy_ratio = 0
            sell_ratio = 0

        # Store in history
        self.signal_history.append({
            "timestamp": datetime.now(),
            "signal": final_signal,
            "confidence": confidence,
            "buy_ratio": buy_ratio,
            "sell_ratio": sell_ratio,
            "active_strategies": active_strategies,
            "regime": self.regime.detect_regime()['regime']
        })

        return {
            "signal": final_signal,
            "confidence": confidence,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "buy_ratio": buy_ratio,
            "sell_ratio": sell_ratio,
            "total_weight": total_weight,
            "active_strategies": active_strategies,
            "regime": self.regime.detect_regime()['regime']
        }

    def get_recent_signals(self, lookback: int = 10) -> List[Dict]:
        """
        Get recent signal history.

        Args:
            lookback: Number of signals to return

        Returns:
            List of recent signals
        """
        return self.signal_history[-lookback:]

    def get_trading_recommendation(self) -> Dict:
        """
        Get complete trading recommendation.

        Returns:
            Dictionary with recommendation details
        """
        ensemble = self.calculate_ensemble_signal()

        # Calculate position size recommendation
        if ensemble['signal'] == "BUY":
            position_size = min(0.95, ensemble['confidence'] / 100)
            action = "BUY"
        elif ensemble['signal'] == "SELL":
            position_size = min(0.95, ensemble['confidence'] / 100)
            action = "SELL"
        else:
            position_size = 0
            action = "HOLD"

        return {
            "action": action,
            "confidence": ensemble['confidence'],
            "position_size": position_size,
            "regime": ensemble['regime'],
            "active_strategies": ensemble['active_strategies'],
            "buy_ratio": ensemble['buy_ratio'],
            "sell_ratio": ensemble['sell_ratio'],
            "reason": f"{action} signal with {ensemble['confidence']}% confidence from {len(ensemble['active_strategies'])} active strategies"
        }


if __name__ == "__main__":
    import yfinance as yf
    print("Testing Signal Ensemble...")

    # Download latest data
    df = yf.download("NVDA", period="3mo", progress=False)
    df = df.reset_index()

    # Add indicators
    indicators = TechnicalIndicators()
    df = indicators.add_all_indicators(df)

    # Get ensemble signal
    ensemble = SignalEnsemble(df)
    recommendation = ensemble.get_trading_recommendation()

    print(f"\n MARKET REGIME: {recommendation['regime']}")
    print(f"\n TRADING RECOMMENDATION:")
    print(f"   Action: {recommendation['action']}")
    print(f"   Confidence: {recommendation['confidence']}%")
    print(f"   Position Size: {recommendation['position_size']:.1%}")
    print(f"   Buy Ratio: {recommendation['buy_ratio']:.1%}")
    print(f"   Sell Ratio: {recommendation['sell_ratio']:.1%}")

    print(f"\n Active Strategies ({len(recommendation['active_strategies'])}):")
    for strategy, signal, conf in recommendation['active_strategies']:
        print(f"   {strategy:20}: {signal:4} ({conf:.0f}%)")

    print(f"\n Signal Ensemble Ready!")