"""
Pattern Recognition Module
Detects candlestick patterns and chart patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class PatternRecognition:
    """
    Detect trading patterns in price data.

    Categories:
    1. Candlestick Patterns (single and multi-candle)
    2. Chart Patterns (Head & Shoulders, Double Top/Bottom)
    3. Support & Resistance Levels
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV data.

        Args:
            df: DataFrame with 'Open', 'High', 'Low', 'Close' columns
        """
        self.df = df.copy()
        self.open = df['Open'].squeeze()
        self.high = df['High'].squeeze()
        self.low = df['Low'].squeeze()
        self.close = df['Close'].squeeze()

    # ================================================================
    # CANDLESTICK PATTERNS
    # ================================================================

    def is_doji(self, row) -> bool:
        """
        Doji pattern - Open and Close are almost equal.

        Indicates indecision in the market.
        """
        try:
            # Handle both Series and dict-like objects
            if hasattr(row, 'iloc'):
                # If it's a Series with multiple values
                open_val = float(row['Open']) if hasattr(row['Open'], 'item') else float(row['Open'])
                close_val = float(row['Close']) if hasattr(row['Close'], 'item') else float(row['Close'])
                high_val = float(row['High']) if hasattr(row['High'], 'item') else float(row['High'])
                low_val = float(row['Low']) if hasattr(row['Low'], 'item') else float(row['Low'])
            else:
                open_val = float(row['Open'])
                close_val = float(row['Close'])
                high_val = float(row['High'])
                low_val = float(row['Low'])

            body = abs(close_val - open_val)
            range_ = high_val - low_val
            threshold = 0.1

            if range_ == 0:
                return False

            return body <= (range_ * threshold)
        except Exception as e:
            return False

    def is_hammer(self, row) -> bool:
        """
        Hammer pattern - Small body at top, long lower wick.

        Bullish reversal signal at bottom of downtrend.
        """
        try:
            if hasattr(row, 'iloc'):
                open_val = float(row['Open']) if hasattr(row['Open'], 'item') else float(row['Open'])
                close_val = float(row['Close']) if hasattr(row['Close'], 'item') else float(row['Close'])
                high_val = float(row['High']) if hasattr(row['High'], 'item') else float(row['High'])
                low_val = float(row['Low']) if hasattr(row['Low'], 'item') else float(row['Low'])
            else:
                open_val = float(row['Open'])
                close_val = float(row['Close'])
                high_val = float(row['High'])
                low_val = float(row['Low'])

            body = abs(close_val - open_val)
            lower_wick = min(open_val, close_val) - low_val
            upper_wick = high_val - max(open_val, close_val)

            if body == 0:
                return False

            return (lower_wick > body * 2 and
                    upper_wick < body * 0.5)
        except Exception:
            return False

    def is_shooting_star(self, row) -> bool:
        """
        Shooting Star pattern - Small body at bottom, long upper wick.

        Bearish reversal signal at top of uptrend.
        """
        try:
            if hasattr(row, 'iloc'):
                open_val = float(row['Open']) if hasattr(row['Open'], 'item') else float(row['Open'])
                close_val = float(row['Close']) if hasattr(row['Close'], 'item') else float(row['Close'])
                high_val = float(row['High']) if hasattr(row['High'], 'item') else float(row['High'])
                low_val = float(row['Low']) if hasattr(row['Low'], 'item') else float(row['Low'])
            else:
                open_val = float(row['Open'])
                close_val = float(row['Close'])
                high_val = float(row['High'])
                low_val = float(row['Low'])

            body = abs(close_val - open_val)
            upper_wick = high_val - max(open_val, close_val)
            lower_wick = min(open_val, close_val) - low_val

            if body == 0:
                return False

            return (upper_wick > body * 2 and
                    lower_wick < body * 0.5)
        except Exception:
            return False

    def is_engulfing(self, row1, row2) -> Tuple[bool, str]:
        """
        Engulfing pattern - Second candle completely engulfs first.

        Returns:
            (is_pattern, direction): direction = "bullish" or "bearish"
        """
        try:
            # Get first row values
            o1 = float(row1['Open']) if hasattr(row1['Open'], 'item') else float(row1['Open'])
            c1 = float(row1['Close']) if hasattr(row1['Close'], 'item') else float(row1['Close'])

            # Get second row values
            o2 = float(row2['Open']) if hasattr(row2['Open'], 'item') else float(row2['Open'])
            c2 = float(row2['Close']) if hasattr(row2['Close'], 'item') else float(row2['Close'])

            # Bullish Engulfing (green candle engulfs red candle)
            if (c2 > o2 and  # Second is bullish
                    c1 < o1 and  # First is bearish
                    c2 > o1 and  # Second closes above first's open
                    o2 < c1):  # Second opens below first's close
                return (True, "bullish")

            # Bearish Engulfing (red candle engulfs green candle)
            elif (c2 < o2 and  # Second is bearish
                  c1 > o1 and  # First is bullish
                  o2 > c1 and  # Second opens above first's close
                  c2 < o1):  # Second closes below first's open
                return (True, "bearish")

            return (False, "")
        except Exception:
            return (False, "")

    def find_candlestick_patterns(self) -> List[Dict]:
        """
        Find all candlestick patterns in the data.
        """
        patterns = []

        for i in range(len(self.df)):
            current = self.df.iloc[i]

            # Get current date
            date = current.name if hasattr(current, 'name') else self.df.index[i]

            # Single candle patterns - simplified
            try:
                open_val = float(current['Open'])
                close_val = float(current['Close'])
                high_val = float(current['High'])
                low_val = float(current['Low'])

                body = abs(close_val - open_val)
                total_range = high_val - low_val

                if total_range > 0:
                    # Doji (very small body)
                    if body < total_range * 0.1:
                        patterns.append({
                            'date': date,
                            'pattern': 'DOJI',
                            'signal': 'NEUTRAL',
                            'description': f'Doji at ${close_val:.2f} - indecision'
                        })

                    # Large bullish candle (body > 2x previous range)
                    if close_val > open_val and body > total_range * 0.7:
                        patterns.append({
                            'date': date,
                            'pattern': 'LONG_BULLISH',
                            'signal': 'BULLISH',
                            'description': f'Strong bullish candle +{((close_val - open_val) / open_val * 100):.1f}%'
                        })

                    # Large bearish candle
                    if close_val < open_val and body > total_range * 0.7:
                        patterns.append({
                            'date': date,
                            'pattern': 'LONG_BEARISH',
                            'signal': 'BEARISH',
                            'description': f'Strong bearish candle -{((open_val - close_val) / open_val * 100):.1f}%'
                        })
            except:
                continue

        return patterns

    # ================================================================
    # CHART PATTERNS
    # ================================================================

    def find_double_top(self, lookback: int = 50, tolerance: float = 0.02) -> List[Dict]:
        """
        Find Double Top patterns (bearish reversal).

        Args:
            lookback: Number of candles to look back
            tolerance: Price tolerance for peaks (2% default)

        Returns:
            List of detected double tops
        """
        patterns = []
        highs = self.high.tail(lookback)

        # Find peaks (local maxima)
        peaks = []
        for i in range(5, len(highs) - 5):
            if (highs.iloc[i] > highs.iloc[i - 5:i].max() and
                    highs.iloc[i] > highs.iloc[i + 1:i + 6].max()):
                peaks.append((i, highs.iloc[i]))

        # Find two peaks at similar price levels
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                price_diff = abs(peaks[i][1] - peaks[j][1]) / peaks[i][1]

                if price_diff < tolerance:
                    # Check if there's a trough between them
                    between_highs = highs.iloc[peaks[i][0]:peaks[j][0] + 1]
                    trough = between_highs.min()
                    trough_ratio = (peaks[i][1] - trough) / peaks[i][1]

                    if trough_ratio > 0.03:  # At least 3% drop between peaks
                        patterns.append({
                            'date': self.df.index[peaks[j][0]],
                            'pattern': 'DOUBLE_TOP',
                            'signal': 'BEARISH',
                            'confidence': round(70 + trough_ratio * 100, 1),
                            'description': f'Double top at ${peaks[i][1]:.2f} and ${peaks[j][1]:.2f}'
                        })
                        break

        return patterns

    def find_double_bottom(self, lookback: int = 50, tolerance: float = 0.02) -> List[Dict]:
        """
        Find Double Bottom patterns (bullish reversal).

        Returns:
            List of detected double bottoms
        """
        patterns = []
        lows = self.low.tail(lookback)

        # Find troughs (local minima)
        troughs = []
        for i in range(5, len(lows) - 5):
            if (lows.iloc[i] < lows.iloc[i - 5:i].min() and
                    lows.iloc[i] < lows.iloc[i + 1:i + 6].min()):
                troughs.append((i, lows.iloc[i]))

        # Find two troughs at similar price levels
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                price_diff = abs(troughs[i][1] - troughs[j][1]) / troughs[i][1]

                if price_diff < tolerance:
                    # Check if there's a peak between them
                    between_lows = lows.iloc[troughs[i][0]:troughs[j][0] + 1]
                    peak = between_lows.max()
                    peak_ratio = (peak - troughs[i][1]) / troughs[i][1]

                    if peak_ratio > 0.03:
                        patterns.append({
                            'date': self.df.index[troughs[j][0]],
                            'pattern': 'DOUBLE_BOTTOM',
                            'signal': 'BULLISH',
                            'confidence': round(70 + peak_ratio * 100, 1),
                            'description': f'Double bottom at ${troughs[i][1]:.2f} and ${troughs[j][1]:.2f}'
                        })
                        break

        return patterns

    # ================================================================
    # SUPPORT & RESISTANCE LEVELS
    # ================================================================

    def find_support_resistance(self, lookback: int = 100, cluster_tolerance: float = 0.01) -> Dict:
        """
        Find key support and resistance levels.

        Args:
            lookback: Number of candles to analyze
            cluster_tolerance: Price tolerance for clustering (1% default)

        Returns:
            Dictionary with support and resistance levels
        """
        highs = self.high.tail(lookback)
        lows = self.low.tail(lookback)

        # Find peaks for resistance
        peaks = []
        for i in range(5, len(highs) - 5):
            if (highs.iloc[i] >= highs.iloc[i - 5:i + 5].max()):
                peaks.append(highs.iloc[i])

        # Find troughs for support
        troughs = []
        for i in range(5, len(lows) - 5):
            if (lows.iloc[i] <= lows.iloc[i - 5:i + 5].min()):
                troughs.append(lows.iloc[i])

        # Cluster nearby price levels
        resistance_levels = self._cluster_levels(peaks, cluster_tolerance)
        support_levels = self._cluster_levels(troughs, cluster_tolerance)

        # Sort by strength (frequency of touches)
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        support_levels.sort(key=lambda x: x['strength'], reverse=True)

        return {
            "resistance": resistance_levels[:5],  # Top 5 resistance levels
            "support": support_levels[:5],  # Top 5 support levels
            "current_price": self.close.iloc[-1],
            "nearest_resistance": resistance_levels[0] if resistance_levels else None,
            "nearest_support": support_levels[0] if support_levels else None
        }

    def _cluster_levels(self, levels: List[float], tolerance: float) -> List[Dict]:
        """
        Cluster nearby price levels together.

        Args:
            levels: List of price levels
            tolerance: Price tolerance for clustering

        Returns:
            Clustered levels with strength (number of touches)
        """
        if not levels:
            return []

        levels = sorted(levels)
        clusters = []

        for level in levels:
            found = False
            for cluster in clusters:
                if abs(level - cluster['price']) / cluster['price'] < tolerance:
                    # Add to existing cluster
                    cluster['price'] = (cluster['price'] * cluster['strength'] + level) / (cluster['strength'] + 1)
                    cluster['strength'] += 1
                    found = True
                    break

            if not found:
                clusters.append({'price': level, 'strength': 1})

        return clusters

    def is_at_support(self, price: float, support_levels: List[Dict], tolerance: float = 0.01) -> Tuple[
        bool, Optional[Dict]]:
        """
        Check if price is near a support level.

        Returns:
            (is_at_support, support_level)
        """
        for level in support_levels:
            if abs(price - level['price']) / level['price'] < tolerance:
                return (True, level)
        return (False, None)

    def is_at_resistance(self, price: float, resistance_levels: List[Dict], tolerance: float = 0.01) -> Tuple[
        bool, Optional[Dict]]:
        """
        Check if price is near a resistance level.

        Returns:
            (is_at_resistance, resistance_level)
        """
        for level in resistance_levels:
            if abs(price - level['price']) / level['price'] < tolerance:
                return (True, level)
        return (False, None)

    # ================================================================
    # GET ALL PATTERNS
    # ================================================================

    def get_all_patterns(self) -> Dict:
        """
        Get all detected patterns and levels.

        Returns:
            Dictionary with all patterns and levels
        """
        return {
            "candlestick_patterns": self.find_candlestick_patterns(),
            "double_tops": self.find_double_top(),
            "double_bottoms": self.find_double_bottom(),
            "support_resistance": self.find_support_resistance()
        }


if __name__ == "__main__":
    import yfinance as yf

    print("Testing Pattern Recognition...")

    # Download data
    df = yf.download("NVDA", period="3mo")
    df = df.reset_index()

    print(f"Data shape: {df.shape}\n")

    # Initialize pattern recognizer
    pattern = PatternRecognition(df)
    results = pattern.get_all_patterns()

    print(" Candlestick Patterns:")
    for p in results['candlestick_patterns'][-10:]:
        print(f"   {p['date'].strftime('%Y-%m-%d')}: {p['pattern']:15} | {p['signal']:8} | {p['description']}")

    print(f"\n Double Tops Found: {len(results['double_tops'])}")
    for dt in results['double_tops'][:3]:
        print(f"   {dt['date'].strftime('%Y-%m-%d')}: {dt['pattern']} (Confidence: {dt['confidence']:.0f}%)")

    print(f"\n Double Bottoms Found: {len(results['double_bottoms'])}")
    for db in results['double_bottoms'][:3]:
        print(f"   {db['date'].strftime('%Y-%m-%d')}: {db['pattern']} (Confidence: {db['confidence']:.0f}%)")

    print("\n Support & Resistance Levels:")
    sr = results['support_resistance']
    print(f"   Current Price: ${sr['current_price']:.2f}")
    print(f"   Nearest Resistance: ${sr['nearest_resistance']['price']:.2f}" if sr[
        'nearest_resistance'] else "   Nearest Resistance: None")
    print(f"   Nearest Support: ${sr['nearest_support']['price']:.2f}" if sr[
        'nearest_support'] else "   Nearest Support: None")

    print("\n   Top Resistance Levels:")
    for r in sr['resistance'][:3]:
        print(f"      ${r['price']:.2f} (touched {r['strength']} times)")

    print("   Top Support Levels:")
    for s in sr['support'][:3]:
        print(f"      ${s['price']:.2f} (touched {s['strength']} times)")

    print("\n Pattern Recognition Ready!")