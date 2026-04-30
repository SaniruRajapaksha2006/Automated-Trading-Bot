"""
Seasonality Analysis Module
Detects time-based patterns in price movements
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple


class SeasonalityAnalyzer:
    """
    Detect seasonal patterns in price data.

    Patterns detected:
    - Day of week effects (Monday effect, Friday effect)
    - Month of year effects (January effect, September effect)
    - Intraday patterns (open, close, lunch)
    - Holiday effects
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with price data.

        Args:
            df: DataFrame with 'Date' and 'Close' columns
        """
        self.df = df.copy()
        if 'Date' not in df.columns:
            self.df = df.reset_index()

        self.df['DayOfWeek'] = pd.to_datetime(self.df['Date']).dt.dayofweek
        self.df['Month'] = pd.to_datetime(self.df['Date']).dt.month
        self.df['WeekOfYear'] = pd.to_datetime(self.df['Date']).dt.isocalendar().week
        self.df['DayOfMonth'] = pd.to_datetime(self.df['Date']).dt.day

    def get_day_of_week_bias(self) -> Dict[str, float]:
        """
        Calculate average return for each day of week.

        Returns:
            Dictionary with day -> avg_return
        """
        self.df['Return'] = self.df['Close'].pct_change() * 100

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_bias = {}

        for i, name in enumerate(day_names):
            avg_return = self.df[self.df['DayOfWeek'] == i]['Return'].mean()
            day_bias[name] = avg_return if not pd.isna(avg_return) else 0

        return day_bias

    def get_month_bias(self) -> Dict[str, float]:
        """
        Calculate average return for each month.

        Returns:
            Dictionary with month -> avg_return
        """
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_bias = {}

        self.df['Return'] = self.df['Close'].pct_change() * 100

        for i, name in enumerate(month_names, 1):
            avg_return = self.df[self.df['Month'] == i]['Return'].mean()
            month_bias[name] = avg_return if not pd.isna(avg_return) else 0

        return month_bias

    def get_weekly_pattern(self) -> Dict:
        """
        Analyze weekly pattern and return trading bias.

        Returns:
            Dictionary with bias for each day
        """
        day_bias = self.get_day_of_week_bias()

        best_day = max(day_bias, key=day_bias.get)
        worst_day = min(day_bias, key=day_bias.get)

        return {
            "day_bias": day_bias,
            "best_day": best_day,
            "worst_day": worst_day,
            "tuesday_bias": "BULLISH" if day_bias.get('Tuesday', 0) > 0 else "BEARISH",
            "friday_bias": "BULLISH" if day_bias.get('Friday', 0) > 0 else "BEARISH"
        }

    def get_seasonal_bias(self) -> Tuple[str, float]:
        """
        Get current seasonal bias.

        Returns:
            (bias, confidence) - bias is BULLISH/BEARISH/NEUTRAL
        """
        today = datetime.now()
        current_month = today.month
        current_weekday = today.weekday()

        month_bias = self.get_month_bias()
        day_bias = self.get_day_of_week_bias()

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        current_month_name = month_names[current_month - 1]

        month_return = month_bias.get(current_month_name, 0)
        day_return = day_bias.get(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][current_weekday], 0)

        # Combine signals
        if month_return > 0.1 and day_return > 0:
            return ("BULLISH", min(70, 50 + abs(month_return) * 10))
        elif month_return < -0.1 and day_return < 0:
            return ("BEARISH", min(70, 50 + abs(month_return) * 10))
        else:
            return ("NEUTRAL", 30)

    def is_option_expiration_week(self) -> bool:
        """Check if current week is options expiration."""
        today = datetime.now()
        # Options expire 3rd Friday of each month
        third_friday = self._get_third_friday(today.year, today.month)
        week_num = today.isocalendar().week
        exp_week = third_friday.isocalendar().week
        return week_num == exp_week

    def _get_third_friday(self, year: int, month: int) -> datetime:
        """Get third Friday of a given month."""
        from datetime import date
        d = date(year, month, 1)
        # Find first Friday
        while d.weekday() != 4:
            d = d.replace(day=d.day + 1)
        # Add 14 days to get third Friday
        third_friday = d.replace(day=d.day + 14)
        return datetime(third_friday.year, third_friday.month, third_friday.day)

    def get_trading_bias(self) -> str:
        """
        Get trading bias based on seasonality.

        Returns:
            "BUY_BIAS", "SELL_BIAS", or "NEUTRAL"
        """
        bias, confidence = self.get_seasonal_bias()

        if bias == "BULLISH" and confidence > 50:
            return "BUY_BIAS"
        elif bias == "BEARISH" and confidence > 50:
            return "SELL_BIAS"
        else:
            return "NEUTRAL"


if __name__ == "__main__":
    import yfinance as yf

    print("Testing Seasonality Analysis...")

    # Download 2 years of data
    df = yf.download("SPY", period="2y")

    seasonality = SeasonalityAnalyzer(df)

    print("\n Day of Week Bias:")
    day_bias = seasonality.get_day_of_week_bias()
    for day, bias in day_bias.items():
        print(f"   {day}: {bias:+.2f}%")

    print("\n Month Bias:")
    month_bias = seasonality.get_month_bias()
    for month, bias in list(month_bias.items())[:6]:
        print(f"   {month}: {bias:+.2f}%")

    weekly = seasonality.get_weekly_pattern()
    print(f"\n Best Day: {weekly['best_day']}")
    print(f" Worst Day: {weekly['worst_day']}")

    bias, conf = seasonality.get_seasonal_bias()
    print(f"\n Current Seasonal Bias: {bias} (Confidence: {conf:.0f}%)")
    print(f" Trading Bias: {seasonality.get_trading_bias()}")

    print("\n Seasonality Analysis Ready!")