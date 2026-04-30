"""
Advanced Technical Indicators Module
Indicators using TA-Lib library
"""

import talib as ta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class AdvancedIndicators:
    """
    Comprehensive technical indicators using TA-Lib.
    """

    def __init__(self):
        """Initialize Advanced Indicators"""
        pass

    def _to_array(self, series) -> np.ndarray:
        """Convert pandas Series to 1D numpy array"""
        if series is None:
            return None

        # If it's a DataFrame with MultiIndex, extract as 1D
        if isinstance(series, pd.DataFrame):
            # Get first column if multiple
            if len(series.columns) > 0:
                series = series.iloc[:, 0]
            else:
                return None

        # Convert to numpy array and ensure 1D
        arr = np.asarray(series.values.astype(float), dtype=float)

        # Flatten if multi-dimensional
        if arr.ndim > 1:
            arr = arr.flatten()

        return arr

    # ================================================================
    # OVERLAP STUDIES (Moving Averages)
    # ================================================================

    def add_all_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 10+ different moving averages"""
        close = self._to_array(df['Close'])

        # Simple Moving Average
        df['SMA_10'] = ta.SMA(close, timeperiod=10)
        df['SMA_20'] = ta.SMA(close, timeperiod=20)
        df['SMA_50'] = ta.SMA(close, timeperiod=50)
        df['SMA_200'] = ta.SMA(close, timeperiod=200)

        # Exponential Moving Average
        df['EMA_9'] = ta.EMA(close, timeperiod=9)
        df['EMA_12'] = ta.EMA(close, timeperiod=12)
        df['EMA_20'] = ta.EMA(close, timeperiod=20)
        df['EMA_26'] = ta.EMA(close, timeperiod=26)
        df['EMA_50'] = ta.EMA(close, timeperiod=50)

        # Weighted Moving Average
        df['WMA_20'] = ta.WMA(close, timeperiod=20)

        # Double EMA
        df['DEMA_20'] = ta.DEMA(close, timeperiod=20)

        # Triple EMA
        df['TEMA_20'] = ta.TEMA(close, timeperiod=20)

        # Triangular MA
        df['TRIMA_20'] = ta.TRIMA(close, timeperiod=20)

        # Kaufman Adaptive MA
        df['KAMA_20'] = ta.KAMA(close, timeperiod=20)

        return df

    # ================================================================
    # MOMENTUM INDICATORS
    # ================================================================

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 20+ momentum indicators"""
        close = self._to_array(df['Close'])
        high = self._to_array(df['High'])
        low = self._to_array(df['Low'])
        volume = self._to_array(df['Volume']) if 'Volume' in df else None

        # RSI (multiple periods)
        df['RSI_7'] = ta.RSI(close, timeperiod=7)
        df['RSI_14'] = ta.RSI(close, timeperiod=14)
        df['RSI_21'] = ta.RSI(close, timeperiod=21)

        # MACD
        macd, signal, hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = hist

        # Stochastic
        slowk, slowd = ta.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['Stochastic_K'] = slowk
        df['Stochastic_D'] = slowd

        # Stochastic RSI
        stochrsi_k, stochrsi_d = ta.STOCHRSI(close, timeperiod=14, fastk_period=3, fastd_period=3)
        df['StochRSI_K'] = stochrsi_k
        df['StochRSI_D'] = stochrsi_d

        # Williams %R
        df['Williams_R'] = ta.WILLR(high, low, close, timeperiod=14)

        # ADX (Trend Strength)
        df['ADX'] = ta.ADX(high, low, close, timeperiod=14)
        df['ADX_Pos'] = ta.PLUS_DI(high, low, close, timeperiod=14)
        df['ADX_Neg'] = ta.MINUS_DI(high, low, close, timeperiod=14)

        # CCI (Commodity Channel Index)
        df['CCI'] = ta.CCI(high, low, close, timeperiod=14)

        # Aroon
        aroon_down, aroon_up = ta.AROON(high, low, timeperiod=25)
        df['Aroon_Up'] = aroon_up
        df['Aroon_Down'] = aroon_down
        df['Aroon_Osc'] = ta.AROONOSC(high, low, timeperiod=25)

        # Money Flow Index
        if volume is not None:
            df['MFI'] = ta.MFI(high, low, close, volume, timeperiod=14)

        # Ultimate Oscillator
        df['Ultimate_Osc'] = ta.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        # Rate of Change
        df['ROC_10'] = ta.ROC(close, timeperiod=10)
        df['ROC_20'] = ta.ROC(close, timeperiod=20)

        # Momentum
        df['MOM_10'] = ta.MOM(close, timeperiod=10)

        return df

    # ================================================================
    # VOLUME INDICATORS
    # ================================================================

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        close = self._to_array(df['Close'])
        high = self._to_array(df['High'])
        low = self._to_array(df['Low'])
        volume = self._to_array(df['Volume']) if 'Volume' in df else None

        if volume is not None:
            # On Balance Volume
            df['OBV'] = ta.OBV(close, volume)

            # Accumulation/Distribution Line
            df['AD'] = ta.AD(high, low, close, volume)

            # AD Oscillator
            df['ADOSC'] = ta.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

            # Chaikin Money Flow
            mfv = ((close - low) - (high - close)) / (high - low) * volume
            df['CMF'] = pd.Series(mfv).rolling(window=20).mean().values

        return df

    # ================================================================
    # VOLATILITY INDICATORS
    # ================================================================

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        high = self._to_array(df['High'])
        low = self._to_array(df['Low'])
        close = self._to_array(df['Close'])

        # ATR (multiple periods)
        df['ATR_7'] = ta.ATR(high, low, close, timeperiod=7)
        df['ATR_14'] = ta.ATR(high, low, close, timeperiod=14)
        df['ATR_21'] = ta.ATR(high, low, close, timeperiod=21)

        # Normalized ATR
        df['NATR_14'] = ta.NATR(high, low, close, timeperiod=14)

        # True Range
        df['TRANGE'] = ta.TRANGE(high, low, close)

        # Bollinger Bands (20 period)
        upper, middle, lower = ta.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_Upper_20'] = upper
        df['BB_Middle_20'] = middle
        df['BB_Lower_20'] = lower
        df['BB_Width_20'] = (upper - middle) / middle
        df['BB_Position_20'] = (close - lower) / (upper - lower)

        # Bollinger Bands (50 period)
        upper, middle, lower = ta.BBANDS(close, timeperiod=50, nbdevup=2, nbdevdn=2)
        df['BB_Upper_50'] = upper
        df['BB_Middle_50'] = middle
        df['BB_Lower_50'] = lower

        return df

    # ================================================================
    # PATTERN RECOGNITION
    # ================================================================

    def add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognitions"""

        # Convert to arrays
        open_arr = self._to_array(df['Open']) if 'Open' in df else None
        high = self._to_array(df['High'])
        low = self._to_array(df['Low'])
        close = self._to_array(df['Close'])

        if open_arr is None:
            return df

        # Bullish Patterns
        df['CDL_HAMMER'] = ta.CDLHAMMER(open_arr, high, low, close)
        df['CDL_INVERTED_HAMMER'] = ta.CDLINVERTEDHAMMER(open_arr, high, low, close)
        df['CDL_ENGULFING'] = ta.CDLENGULFING(open_arr, high, low, close)
        df['CDL_PIERCING'] = ta.CDLPIERCING(open_arr, high, low, close)
        df['CDL_MORNING_STAR'] = ta.CDLMORNINGSTAR(open_arr, high, low, close)
        df['CDL_DRAGONFLY_DOJI'] = ta.CDLDRAGONFLYDOJI(open_arr, high, low, close)

        # Bearish Patterns
        df['CDL_SHOOTING_STAR'] = ta.CDLSHOOTINGSTAR(open_arr, high, low, close)
        df['CDL_HANGING_MAN'] = ta.CDLHANGINGMAN(open_arr, high, low, close)
        df['CDL_DARK_CLOUD'] = ta.CDLDARKCLOUDCOVER(open_arr, high, low, close)
        df['CDL_EVENING_STAR'] = ta.CDLEVENINGSTAR(open_arr, high, low, close)
        df['CDL_BEARISH_HARAMI'] = ta.CDLHARAMI(open_arr, high, low, close)

        # Reversal Patterns
        df['CDL_DOJI'] = ta.CDLDOJI(open_arr, high, low, close)
        df['CDL_LONG_LEGGED_DOJI'] = ta.CDLLONGLEGGEDDOJI(open_arr, high, low, close)
        df['CDL_HARAMI_CROSS'] = ta.CDLHARAMICROSS(open_arr, high, low, close)

        # Continuation Patterns
        df['CDL_3_WHITE_SOLDIERS'] = ta.CDL3WHITESOLDIERS(open_arr, high, low, close)
        df['CDL_3_BLACK_CROWS'] = ta.CDL3BLACKCROWS(open_arr, high, low, close)

        return df

    # ================================================================
    # ADD ALL INDICATORS
    # ================================================================

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ALL indicators to DataFrame"""
        df = self.add_all_moving_averages(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volume_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_candlestick_patterns(df)

        return df


if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf

    print("Testing Advanced Indicators...")

    # Download data
    ticker = yf.Ticker("NVDA")
    df = ticker.history(period="3mo")

    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # Reset index to get Date as column (yfinance returns DataFrame with no MultiIndex)
    df = df.reset_index()

    print(f"\nAfter reset: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Add all indicators
    adv = AdvancedIndicators()
    df = adv.add_all_indicators(df)

    # Count indicators
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    indicator_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\n Added {len(indicator_cols)} advanced indicators")
    print(f"   Sample indicators: {indicator_cols[:20]}")

    # Show latest values
    print("\n Latest Indicator Values:")
    latest = df.iloc[-1]
    for col in indicator_cols[:15]:
        val = latest[col]
        if pd.notna(val):
            print(f"   {col:25}: {val:8.2f}")

    print("\n Advanced Indicators Module Ready!")