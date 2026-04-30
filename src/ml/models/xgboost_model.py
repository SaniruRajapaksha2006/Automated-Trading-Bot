"""
XGBoost Signal Enhancer
Gradient boosting for trade signal improvement
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class XGBoostSignalEnhancer:
    """
    XGBoost model to enhance trading signals.

    Features:
    - Fast training
    - Feature importance
    - Handles complex interactions
    """

    def __init__(self):
        """Initialize XGBoost model."""
        self.model = None
        self.feature_importance = None

    def build_model(self, params: Dict = None) -> xgb.XGBClassifier:
        """
        Build XGBoost classifier.

        Args:
            params: Model parameters (uses defaults if None)

        Returns:
            XGBClassifier instance
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }

        return xgb.XGBClassifier(**params)

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train XGBoost model.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Training metrics
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Build and train
        self.model = self.build_model()
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Validation accuracy
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        return {
            'accuracy': accuracy,
            'feature_importance': self.feature_importance.head(10)
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Prediction probabilities
        """
        return self.model.predict_proba(X)[:, 1]

    def predict_direction(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict direction.

        Args:
            X: Feature DataFrame
            threshold: Classification threshold

        Returns:
            1 for UP, 0 for DOWN
        """
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

    def enhance_signal(self, signal: np.ndarray, ml_prediction: np.ndarray, weight: float = 0.3) -> np.ndarray:
        """
        Combine traditional signal with ML prediction.

        Args:
            signal: Traditional strategy signal (-1, 0, 1)
            ml_prediction: ML probability (0-1)
            weight: ML weight (0-1)

        Returns:
            Enhanced signal
        """
        # Convert ML to -1, 0, 1
        ml_signal = (ml_prediction - 0.5) * 2

        # Weighted combination
        enhanced = (1 - weight) * signal + weight * ml_signal

        return np.clip(enhanced, -1, 1)


if __name__ == "__main__":
    from src.ml.features.feature_engineering import MLFeatureEngineer
    import yfinance as yf
    from src.indicators.technical import TechnicalIndicators

    print("Testing XGBoost Signal Enhancer...")
    print("=" * 60)

    # Download data
    df = yf.download("NVDA", period="2y")
    df = df.reset_index()

    # Add indicators
    indicators = TechnicalIndicators()
    df = indicators.add_all_indicators(df)

    # Prepare data
    mfe = MLFeatureEngineer(df)
    X, y = mfe.get_feature_importance_ready()

    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Train XGBoost
    xgb_model = XGBoostSignalEnhancer()
    results = xgb_model.train(X, y)

    print(f"\n XGBoost Validation Accuracy: {results['accuracy']:.2%}")
    print("\n Top 10 Features:")
    for _, row in results['feature_importance'].iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

    print("\n XGBoost Model Ready!")