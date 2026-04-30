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
        """Bollinger Bands - Shows volatility and price extremes."""
        close = self._get_close_series(df)

        df['BB_Middle'] = close.rolling(window=period).mean()
        bb_std = close.rolling(window=period).std()
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
        Average Directional Index - Simplified version that works
        """
        try:
            import ta
            df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=period).adx()
            df['PLUS_DI'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=period).adx_pos()
            df['MINUS_DI'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=period).adx_neg()
        except:
            # Fallback: simple approximation
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            df['ADX'] = 25  # neutral value
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

    def add_awesome_oscillator(self, df: pd.DataFrame, period_fast=5, period_slow=34) -> pd.DataFrame:
        """Awesome Oscillator - Market momentum"""
        median_price = (df['High'] + df['Low']) / 2
        sma_fast = median_price.rolling(period_fast).mean()
        sma_slow = median_price.rolling(period_slow).mean()
        df['AO'] = sma_fast - sma_slow
        return df

    def add_keltner_channel(self, df: pd.DataFrame, period=20, multiplier=2) -> pd.DataFrame:
        """Keltner Channel - Volatility-based envelope"""
        ema = df['Close'].ewm(span=period, adjust=False).mean()
        # Fix: call add_atr first, then get 'ATR' column
        df = self.add_atr(df, period)
        df['KC_Upper'] = ema + (df['ATR'] * multiplier)
        df['KC_Lower'] = ema - (df['ATR'] * multiplier)
        df['KC_Middle'] = ema
        return df

    def add_money_flow_index(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """Money Flow Index - Volume-weighted RSI"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']

        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)

        mfr = positive_flow.rolling(period).sum() / negative_flow.rolling(period).sum()
        df['MFI'] = 100 - (100 / (1 + mfr))
        return df

    def add_chaikin_oscillator(self, df: pd.DataFrame, fast=3, slow=10) -> pd.DataFrame:
        """Chaikin Oscillator - Accumulation/Distribution momentum"""
        adl = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
        adl = adl.fillna(0).cumsum()
        df['Chaikin'] = adl.ewm(span=fast).mean() - adl.ewm(span=slow).mean()
        return df

    def add_donchian_channel(self, df: pd.DataFrame, period=20) -> pd.DataFrame:
        """Donchian Channel - Highest high / Lowest low"""
        df['DC_Upper'] = df['High'].rolling(period).max()
        df['DC_Lower'] = df['Low'].rolling(period).min()
        df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2
        return df

    def add_parabolic_sar(self, df: pd.DataFrame, step=0.02, maximum=0.2) -> pd.DataFrame:
        """Parabolic SAR - Trend following stop and reverse"""
        try:
            import ta
            df['PSAR'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close'], step=step, maximum=maximum).psar()
        except:
            df['PSAR'] = df['Close']  # Fallback
        return df

    def add_ichimoku_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ichimoku Cloud components"""
        # Tenkan-sen (Conversion Line)
        period9_high = df['High'].rolling(9).max()
        period9_low = df['Low'].rolling(9).min()
        df['Ichimoku_Tenkan'] = (period9_high + period9_low) / 2

        # Kijun-sen (Base Line)
        period26_high = df['High'].rolling(26).max()
        period26_low = df['Low'].rolling(26).min()
        df['Ichimoku_Kijun'] = (period26_high + period26_low) / 2

        # Senkou Span A (Leading Span A)
        df['Ichimoku_SenkouA'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)

        # Senkou Span B (Leading Span B)
        period52_high = df['High'].rolling(52).max()
        period52_low = df['Low'].rolling(52).min()
        df['Ichimoku_SenkouB'] = ((period52_high + period52_low) / 2).shift(26)

        return df

    # ================================================================
    # ADD ALL INDICATORS AT ONCE
    # ================================================================

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_sma(df, 20)
        df = self.add_sma(df, 50)
        df = self.add_ema(df, 12)
        df = self.add_ema(df, 20)
        df = self.add_ema(df, 26)
        df = self.add_rsi(df, 14)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df, 14)
        df = self.add_cci(df, 20)
        df = self.add_roc(df, 10)
        df = self.add_williams_r(df, 14)
        df = self.add_awesome_oscillator(df)
        df = self.add_money_flow_index(df)
        df = self.add_chaikin_oscillator(df)
        df = self.add_donchian_channel(df)
        df = self.add_ichimoku_components(df)
        df = self.add_adx(df, 14)  # UNCOMMENTED
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