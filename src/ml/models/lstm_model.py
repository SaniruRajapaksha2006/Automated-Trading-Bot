"""
LSTM Price Prediction Model
Deep learning for time series forecasting
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler


class LSTMPredictor:
    """
    LSTM model for price direction prediction.

    Features:
    - Sequential memory (remembers past patterns)
    - Multi-step prediction
    - Confidence scoring
    """

    def __init__(self, sequence_length: int = 60, n_features: int = 10):
        """
        Initialize LSTM model.

        Args:
            sequence_length: Number of past days to look back
            n_features: Number of input features
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()

    def build_model(self) -> Sequential:
        """
        Build LSTM architecture.

        Architecture:
        - LSTM layer 1: 128 units (returns sequences)
        - Dropout: 20%
        - LSTM layer 2: 64 units
        - Dropout: 20%
        - Dense: 32 units
        - Output: 1 unit (sigmoid for binary classification)
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(64, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),

            Dense(32, activation='relu'),
            Dropout(0.1),

            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, validation_split: float = 0.2) -> Dict:
        """
        Train LSTM model.

        Args:
            X: Feature array (samples, sequence_length, features)
            y: Target array (samples,)
            epochs: Training epochs
            validation_split: Portion for validation

        Returns:
            Training history
        """
        # Scale data
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(original_shape)

        # Build model
        self.model = self.build_model()

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]

        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature array

        Returns:
            Prediction probabilities
        """
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X = X_scaled.reshape(original_shape)

        return self.model.predict(X).flatten()

    def predict_direction(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict direction (UP/DOWN).

        Args:
            X: Feature array
            threshold: Classification threshold

        Returns:
            1 for UP, 0 for DOWN
        """
        probs = self.predict(X)
        return (probs >= threshold).astype(int)


if __name__ == "__main__":
    from src.ml.features.feature_engineering import MLFeatureEngineer
    import yfinance as yf
    from src.indicators.technical import TechnicalIndicators

    print("Testing LSTM Model...")
    print("=" * 60)

    # Download data
    df = yf.download("NVDA", period="2y")
    df = df.reset_index()

    # Add indicators
    indicators = TechnicalIndicators()
    df = indicators.add_all_indicators(df)

    # Prepare data
    mfe = MLFeatureEngineer(df)
    X, y, features = mfe.prepare_ml_data(lookback=60)

    print(f"Data shape: X={X.shape}, y={y.shape}")

    if len(X) > 100:
        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Create and train model
        lstm = LSTMPredictor(sequence_length=60, n_features=X.shape[2])
        print("\nTraining LSTM...")
        history = lstm.train(X_train, y_train, epochs=30, validation_split=0.2)

        # Test
        predictions = lstm.predict_direction(X_test)
        accuracy = (predictions == y_test).mean()

        print(f"\n✅ LSTM Test Accuracy: {accuracy:.2%}")
    else:
        print("Not enough data for LSTM training (need more than 100 samples)")

    print("\n✅ LSTM Model Ready!")