"""
Market Regime Detection Module
Identifies whether the market is Bullish, Bearish, or Sideways
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import ta


class MarketRegime:
    """
    Detect current market regime using multiple methods.

    Methods:
    1. Trend Slope (Moving Average angle)
    2. ADX (Trend Strength)
    3. Price vs Moving Averages
    4. Higher Highs/Higher Lows
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV data.

        Args:
            df: DataFrame with 'High', 'Low', 'Close' columns
        """
        self.df = df.copy()

        # Ensure we have 1D Series (not DataFrames with 1 column)
        self.close = df['Close'].squeeze()
        self.high = df['High'].squeeze()
        self.low = df['Low'].squeeze()

        # Add ADX using ta library (pass 1D Series)
        self.df['ADX'] = ta.trend.ADXIndicator(
            high=self.high,
            low=self.low,
            close=self.close,
            window=14
        ).adx()

        # Add Moving Averages
        self.df['SMA_50'] = self.close.rolling(50).mean()
        self.df['SMA_200'] = self.close.rolling(200).mean()

    def regime_by_slope(self) -> Tuple[str, float]:
        """
        Detect regime based on moving average slope.

        Returns:
            (bullish/bearish/sideways, confidence %)
        """
        sma_50 = self.df['SMA_50'].dropna()

        if len(sma_50) < 50:
            return ("SIDEWAYS", 0)

        # Slope of SMA_50 over last 5 days
        slope_50 = (sma_50.iloc[-1] - sma_50.iloc[-5]) / sma_50.iloc[-5] * 100

        # Position of price vs moving averages
        current_price = self.close.iloc[-1]
        price_vs_50 = (current_price - sma_50.iloc[-1]) / sma_50.iloc[-1] * 100

        # Determine regime
        if slope_50 > 0.3 and price_vs_50 > 0:
            confidence = min(90, 50 + slope_50 * 10)
            return ("BULLISH", confidence)

        elif slope_50 < -0.3 and price_vs_50 < 0:
            confidence = min(90, 50 + abs(slope_50) * 10)
            return ("BEARISH", confidence)

        else:
            return ("SIDEWAYS", 40)

    def regime_by_adx(self) -> Tuple[str, float]:
        """
        Detect regime using ADX (Trend Strength).

        ADX > 25 = Trending market
        ADX < 20 = Sideways/Ranging market

        Returns:
            (bullish/bearish/sideways, confidence %)
        """
        current_adx = self.df['ADX'].iloc[-1]

        if pd.isna(current_adx):
            current_adx = 20

        if current_adx > 25:
            # Trending, now check direction
            current_price = self.close.iloc[-1]
            price_50_ago = self.close.iloc[-50] if len(self.close) > 50 else self.close.iloc[0]

            if current_price > price_50_ago:
                confidence = min(80, 50 + (current_adx - 25))
                return ("BULLISH", confidence)
            else:
                confidence = min(80, 50 + (current_adx - 25))
                return ("BEARISH", confidence)
        else:
            # Ranging/Sideways
            confidence = min(70, 40 + (20 - current_adx))
            return ("SIDEWAYS", confidence)

    def regime_by_hh_hl(self) -> Tuple[str, float]:
        """
        Detect regime using Higher Highs/Higher Lows.

        Returns:
            (bullish/bearish/sideways, confidence %)
        """
        lookback = 20

        if len(self.high) < lookback:
            return ("SIDEWAYS", 0)

        recent_highs = self.high.tail(lookback)
        recent_lows = self.low.tail(lookback)

        # Count higher highs and higher lows
        higher_high_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs.iloc[i] > recent_highs.iloc[i-1])
        higher_low_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows.iloc[i] > recent_lows.iloc[i-1])

        # Count lower highs and lower lows
        lower_high_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs.iloc[i] < recent_highs.iloc[i-1])
        lower_low_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows.iloc[i] < recent_lows.iloc[i-1])

        # Check for bullish pattern
        if higher_high_count > 12 and higher_low_count > 12:
            return ("BULLISH", 65)
        elif lower_high_count > 12 and lower_low_count > 12:
            return ("BEARISH", 65)
        else:
            return ("SIDEWAYS", 50)

    def detect_regime(self) -> Dict:
        """
        Combine all methods to detect overall market regime.

        Returns:
            Dictionary with regime, confidence, and method details
        """
        # Get signals from each method
        slope_regime, slope_conf = self.regime_by_slope()
        adx_regime, adx_conf = self.regime_by_adx()
        hhhl_regime, hhhl_conf = self.regime_by_hh_hl()

        # Count votes
        regimes = [slope_regime, adx_regime, hhhl_regime]
        bullish_votes = regimes.count("BULLISH")
        bearish_votes = regimes.count("BEARISH")
        sideways_votes = regimes.count("SIDEWAYS")

        # Calculate average confidence
        avg_confidence = (slope_conf + adx_conf + hhhl_conf) / 3

        # Determine overall regime
        if bullish_votes >= 2:
            overall = "BULLISH"
            final_confidence = avg_confidence + 10
        elif bearish_votes >= 2:
            overall = "BEARISH"
            final_confidence = avg_confidence + 10
        else:
            overall = "SIDEWAYS"
            final_confidence = avg_confidence

        return {
            "regime": overall,
            "confidence": round(min(final_confidence, 95), 1),
            "bullish_votes": bullish_votes,
            "bearish_votes": bearish_votes,
            "sideways_votes": sideways_votes,
            "details": {
                "slope_method": {"regime": slope_regime, "confidence": slope_conf},
                "adx_method": {"regime": adx_regime, "confidence": adx_conf},
                "hh_hl_method": {"regime": hhhl_regime, "confidence": hhhl_conf}
            }
        }

    def get_trading_bias(self) -> str:
        """
        Get trading bias based on market regime.

        Returns:
            "BUY_BIAS" - Favor long positions
            "SELL_BIAS" - Favor short positions
            "NEUTRAL" - Reduce position size
        """
        regime_info = self.detect_regime()
        regime = regime_info["regime"]
        confidence = regime_info["confidence"]

        if regime == "BULLISH" and confidence > 50:
            return "BUY_BIAS"
        elif regime == "BEARISH" and confidence > 50:
            return "SELL_BIAS"
        else:
            return "NEUTRAL"


if __name__ == "__main__":
    import yfinance as yf

    print("Testing Market Regime Detection...")

    # Download data
    df = yf.download("NVDA", period="6mo")
    df = df.reset_index()

    print(f"Data shape: {df.shape}\n")

    # Detect regime
    regime = MarketRegime(df)
    result = regime.detect_regime()

    print(" Market Regime Analysis")
    print(f"Overall Regime: {result['regime']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"\nVotes: Bullish={result['bullish_votes']}, Bearish={result['bearish_votes']}, Sideways={result['sideways_votes']}")

    print("\n Method Details:")
    for method, details in result['details'].items():
        print(f"   {method:15}: {details['regime']:8} (Confidence: {details['confidence']:.0f}%)")

    print(f"\n Trading Bias: {regime.get_trading_bias()}")

    print("\n Market Regime Detection Ready!")