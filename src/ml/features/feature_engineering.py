"""
ML Feature Engineering Module
Prepares data for machine learning models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import numpy as np
from typing import Tuple, List


class MLFeatureEngineer:
    """
    Prepare features for ML models (LSTM, XGBoost).

    Creates:
    - Technical indicator features
    - Lag features (past prices)
    - Target labels (future price direction)
    - Train/test splits
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV data.

        Args:
            df: DataFrame with OHLCV columns
        """
        self.df = df.copy()
        self.features = None
        self.target = None

    def add_lag_features(self, periods: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Add lagged price features.

        Why? Past prices help predict future prices.

        Args:
            periods: Number of periods to lag

        Returns:
            DataFrame with lag features
        """
        df = self.df.copy()

        for period in periods:
            df[f'LAG_{period}'] = df['Close'].shift(period)
            df[f'RETURN_{period}'] = df['Close'].pct_change(period) * 100

        return df

    def add_rolling_features(self, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Add rolling statistics features.

        Why? Captures recent price behavior.

        Args:
            windows: Rolling window sizes

        Returns:
            DataFrame with rolling features
        """
        df = self.df.copy()

        for window in windows:
            df[f'MEAN_{window}'] = df['Close'].rolling(window).mean()
            df[f'STD_{window}'] = df['Close'].rolling(window).std()
            df[f'MIN_{window}'] = df['Close'].rolling(window).min()
            df[f'MAX_{window}'] = df['Close'].rolling(window).max()

        return df

    def create_target(self, forward_period: int = 1) -> pd.DataFrame:
        """
        Create target labels for supervised learning.

        Target: 1 if price goes up, 0 if price goes down

        Args:
            forward_period: How many days ahead to predict

        Returns:
            DataFrame with target column
        """
        df = self.df.copy()
        future_price = df['Close'].shift(-forward_period)
        df['TARGET'] = (future_price > df['Close']).astype(int)
        return df

    def prepare_ml_data(self, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray, list]:
        df = self.add_lag_features()
        df = self.add_rolling_features()
        df = self.create_target()
        df = df.dropna()

        # Add technical indicators if available
        indicator_cols = ['RSI', 'MACD', 'ADX', 'CCI', 'ROC']
        for col in indicator_cols:
            if col in df.columns:
                df[f'IND_{col}'] = df[col]

        # Remove non-numeric columns
        exclude = ['Date', 'TARGET', 'Open', 'High', 'Low', 'Close', 'Volume']
        exclude += list(df.select_dtypes(include=['object', 'datetime64']).columns)

        feature_cols = [col for col in df.columns if col not in exclude]

        # Ensure all features are numeric
        X_df = df[feature_cols].copy()
        X_df = X_df.apply(pd.to_numeric, errors='coerce')
        X_df = X_df.dropna(axis=1, how='all')
        feature_cols = list(X_df.columns)

        X, y = [], []
        for i in range(lookback, len(X_df)):
            X.append(X_df.iloc[i - lookback:i].values)
            y.append(df['TARGET'].iloc[i])

        return np.array(X), np.array(y), feature_cols

    def get_feature_importance_ready(self):
        df = self.add_lag_features()
        df = self.add_rolling_features()
        df = self.create_target()
        df = df.dropna()

        # Add technical indicators if available
        indicator_cols = ['RSI', 'MACD', 'ADX', 'CCI', 'ROC']
        for col in indicator_cols:
            if col in df.columns:
                df[f'IND_{col}'] = df[col]

        # Remove datetime columns and non-numeric columns
        exclude = ['Date', 'TARGET', 'Open', 'High', 'Low', 'Close', 'Volume']
        # Also remove any object/string columns
        exclude += list(df.select_dtypes(include=['object', 'datetime64']).columns)

        feature_cols = [col for col in df.columns if col not in exclude]

        # Ensure all features are numeric
        X = df[feature_cols].copy()
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.dropna(axis=1, how='all')

        return X, df['TARGET']


if __name__ == "__main__":
    import yfinance as yf
    from src.indicators.technical import TechnicalIndicators

    print("Testing ML Feature Engineering...")

    # Download data
    df = yf.download("NVDA", period="1y")
    df = df.reset_index()

    # Add indicators
    indicators = TechnicalIndicators()
    df = indicators.add_all_indicators(df)

    # Feature engineering
    mfe = MLFeatureEngineer(df)
    X, y, features = mfe.prepare_ml_data(lookback=60)

    print(f"\n ML Data Shape:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Features: {len(features)}")
    print(f"   Features list: {features[:10]}...")

    # Feature importance ready
    X_tab, y_tab = mfe.get_feature_importance_ready()
    print(f"\n XGBoost Ready:")
    print(f"   X shape: {X_tab.shape}")
    print(f"   y shape: {y_tab.shape}")

    print("\n ML Feature Engineering Ready!")