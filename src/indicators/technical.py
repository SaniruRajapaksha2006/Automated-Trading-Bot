"""
Technical Indicators Module
Calculates various trading indicators for market analysis

Tools that help read the market:
- RSI tells if a stock is overbought or oversold
- MACD shows momentum direction
- Bollinger Bands show volatility
- Moving Averages show trend direction
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


class TechnicalIndicators:
    """
    Calculate technical indicators from price data.
    """

    def __init__(self):
        """Initialize the indicator calculator"""
        pass

    def _get_close_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Safely extract Close price as a Series.
        This fixes the DataFrame vs Series issue.
        """
        close = df['Close']
        # If it's a DataFrame with one column, convert to Series
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close

    # ================================================================
    # MOVING AVERAGES (Trend Following)
    # ================================================================

    def add_sma(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Simple Moving Average - Smooths price data to identify trend.
        """
        close = self._get_close_series(df)
        df[f'SMA_{period}'] = close.rolling(window=period).mean()
        return df

    def add_ema(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Exponential Moving Average - Gives more weight to recent prices.
        """
        close = self._get_close_series(df)
        df[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
        return df

    # ================================================================
    # RSI - Relative Strength Index (Momentum)
    # ================================================================

    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Relative Strength Index - Measures speed and change of price movements.
        """
        close = self._get_close_series(df)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    # ================================================================
    # MACD - Moving Average Convergence Divergence
    # ================================================================

    def add_macd(self, df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
        """
        MACD - Shows relationship between two moving averages.
        """
        close = self._get_close_series(df)

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        return df

    # ================================================================
    # BOLLINGER BANDS (Volatility)
    # ================================================================

    def add_bollinger_bands(self, df: pd.DataFrame, period=20, num_std=2) -> pd.DataFrame:
        """
        Bollinger Bands - Shows volatility and price extremes.
        """
        close = self._get_close_series(df)

        # Middle Band = SMA
        df['BB_Middle'] = close.rolling(window=period).mean()

        # Standard Deviation
        bb_std = close.rolling(window=period).std()

        # Upper and Lower Bands
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * num_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * num_std)

        return df

    # ================================================================
    # ATR - Average True Range (Volatility for Stop Losses)
    # ================================================================

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average True Range - Measures market volatility.
        """
        # Extract as Series
        high = df['High']
        low = df['Low']
        close = self._get_close_series(df)

        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]

        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=period).mean()

        return df

    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average Directional Index - Trend strength indicator.

        ADX > 25 = Strong trend
        ADX < 20 = Weak trend/Sideways
        """
        high = df['High']
        low = df['Low']
        close = df['Close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm < 0, 0).abs()

        # Smooth with Wilder's method
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(period).mean()
        df['PLUS_DI'] = plus_di
        df['MINUS_DI'] = minus_di

        return df

    def add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Commodity Channel Index - Measures deviation from mean.

        CCI > 100 = Overbought
        CCI < -100 = Oversold
        """
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: abs(x - x.mean()).mean())
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)
        return df

    def add_roc(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """
        Rate of Change - Price momentum.

        ROC > 0 = Positive momentum
        ROC < 0 = Negative momentum
        """
        df['ROC'] = (df['Close'] / df['Close'].shift(period) - 1) * 100
        return df

    def add_aroon(self, df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
        """
        Aroon Indicator - Trend strength and reversal.

        Aroon Up > 70 = Strong uptrend
        Aroon Down > 70 = Strong downtrend
        """
        aroon_up = 100 * df['High'].rolling(period + 1).apply(lambda x: x.argmax()) / period
        aroon_down = 100 * df['Low'].rolling(period + 1).apply(lambda x: x.argmin()) / period
        df['AROON_UP'] = aroon_up
        df['AROON_DOWN'] = aroon_down
        df['AROON_OSC'] = aroon_up - aroon_down
        return df

    def add_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Williams %R - Momentum indicator similar to Stochastic.

        Williams %R < -80 = Oversold
        Williams %R > -20 = Overbought
        """
        highest_high = df['High'].rolling(period).max()
        lowest_low = df['Low'].rolling(period).min()
        df['WILLIAMS_R'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
        return df

    # ================================================================
    # ADD ALL INDICATORS AT ONCE
    # ================================================================

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ALL technical indicators to DataFrame."""
        df = self.add_sma(df, 20)
        df = self.add_sma(df, 50)
        df = self.add_ema(df, 12)
        df = self.add_ema(df, 20)
        df = self.add_ema(df, 26)
        df = self.add_rsi(df, 14)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df, 14)
        #df = self.add_adx(df, 14)  # NEW
        #df = self.add_cci(df, 20)  # NEW
        #df = self.add_roc(df, 10)  # NEW
        #df = self.add_aroon(df, 25)  # NEW
        #df = self.add_williams_r(df, 14)  # NEW
        return df


# Simple test
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    })

    # Calculate indicators
    indicators = TechnicalIndicators()
    df = indicators.add_all_indicators(df)

    print("✅ Technical Indicators Module Ready!")
    print(f"\nSample data with indicators:")
    print(df[['Close', 'SMA_20', 'RSI', 'ATR']].tail())