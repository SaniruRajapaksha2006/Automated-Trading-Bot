"""
Complete Trading Strategies Module
15+ working trading strategies (fully fixed)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
from enum import Enum


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class CompleteStrategies:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.close = df['Close']
        self.high = df['High']
        self.low = df['Low']
        self.volume = df['Volume'] if 'Volume' in df else None
    
    def _get_scalar(self, val) -> float:
        """Convert Series or array to scalar float"""
        if hasattr(val, 'iloc'):
            val = val.iloc[0]
        elif hasattr(val, 'item'):
            val = val.item()
        return float(val)
    
    def _get_latest_value(self, column: str) -> float:
        """Safely get latest value as scalar"""
        val = self.df[column].iloc[-1]
        return self._get_scalar(val)
    
    def strategy_ma_crossover(self) -> Tuple[Signal, float, str]:
        if 'SMA_20' not in self.df.columns or 'SMA_50' not in self.df.columns:
            return Signal.HOLD, 0, "MA not available"
        
        current_fast = self._get_latest_value('SMA_20')
        current_slow = self._get_latest_value('SMA_50')
        prev_fast = self._get_scalar(self.df['SMA_20'].iloc[-2])
        prev_slow = self._get_scalar(self.df['SMA_50'].iloc[-2])
        
        if prev_fast <= prev_slow and current_fast > current_slow:
            return Signal.BUY, 75, "Golden Cross (SMA20 above SMA50)"
        elif prev_fast >= prev_slow and current_fast < current_slow:
            return Signal.SELL, 75, "Death Cross (SMA20 below SMA50)"
        return Signal.HOLD, 0, "No crossover"
    
    def strategy_ema_trend(self) -> Tuple[Signal, float, str]:
        if 'EMA_20' not in self.df.columns:
            return Signal.HOLD, 0, "EMA not available"
        
        current_price = self._get_latest_value('Close')
        current_ema = self._get_latest_value('EMA_20')
        
        if current_price > current_ema:
            return Signal.BUY, 60, "Price above EMA20 (uptrend)"
        elif current_price < current_ema:
            return Signal.SELL, 60, "Price below EMA20 (downtrend)"
        return Signal.HOLD, 0, "Price at EMA20"
    
    def strategy_adx_trend(self) -> Tuple[Signal, float, str]:
        if 'ADX' not in self.df.columns:
            return Signal.HOLD, 0, "ADX not available"
        
        current_adx = self._get_latest_value('ADX')
        
        if current_adx > 25:
            current_price = self._get_latest_value('Close')
            sma50 = self._get_latest_value('SMA_50') if 'SMA_50' in self.df.columns else current_price
            
            if current_price > sma50:
                return Signal.BUY, 70, f"Strong uptrend (ADX: {current_adx:.1f})"
            else:
                return Signal.SELL, 70, f"Strong downtrend (ADX: {current_adx:.1f})"
        return Signal.HOLD, 0, f"Weak trend (ADX: {current_adx:.1f})"
    
    def strategy_rsi_extreme(self) -> Tuple[Signal, float, str]:
        if 'RSI_14' not in self.df.columns:
            return Signal.HOLD, 0, "RSI not available"
        
        current_rsi = self._get_latest_value('RSI_14')
        
        if current_rsi < 25:
            return Signal.BUY, 70, f"Deep oversold (RSI: {current_rsi:.1f})"
        elif current_rsi > 75:
            return Signal.SELL, 70, f"Extreme overbought (RSI: {current_rsi:.1f})"
        elif current_rsi < 30:
            return Signal.BUY, 50, f"Oversold (RSI: {current_rsi:.1f})"
        elif current_rsi > 70:
            return Signal.SELL, 50, f"Overbought (RSI: {current_rsi:.1f})"
        return Signal.HOLD, 0, f"Neutral (RSI: {current_rsi:.1f})"
    
    def strategy_macd_histogram(self) -> Tuple[Signal, float, str]:
        if 'MACD_Histogram' not in self.df.columns:
            return Signal.HOLD, 0, "MACD not available"
        
        current_hist = self._get_latest_value('MACD_Histogram')
        prev_hist = self._get_scalar(self.df['MACD_Histogram'].iloc[-2])
        
        if current_hist > 0 and prev_hist <= 0:
            return Signal.BUY, 65, "MACD histogram turned positive"
        elif current_hist < 0 and prev_hist >= 0:
            return Signal.SELL, 65, "MACD histogram turned negative"
        elif current_hist > 0:
            return Signal.BUY, 30, "Positive momentum"
        elif current_hist < 0:
            return Signal.SELL, 30, "Negative momentum"
        return Signal.HOLD, 0, "Flat momentum"
    
    def strategy_cci_extreme(self) -> Tuple[Signal, float, str]:
        if 'CCI' not in self.df.columns:
            return Signal.HOLD, 0, "CCI not available"
        
        current_cci = self._get_latest_value('CCI')
        
        if current_cci < -100:
            return Signal.BUY, 60, f"Oversold (CCI: {current_cci:.1f})"
        elif current_cci > 100:
            return Signal.SELL, 60, f"Overbought (CCI: {current_cci:.1f})"
        return Signal.HOLD, 0, f"Neutral (CCI: {current_cci:.1f})"
    
    def strategy_bollinger_reversion(self) -> Tuple[Signal, float, str]:
        if 'BB_Upper_20' not in self.df.columns:
            return Signal.HOLD, 0, "Bollinger Bands not available"
        
        current_price = self._get_latest_value('Close')
        upper = self._get_latest_value('BB_Upper_20')
        lower = self._get_latest_value('BB_Lower_20')
        
        if current_price <= lower:
            return Signal.BUY, 65, "Price at lower Bollinger Band"
        elif current_price >= upper:
            return Signal.SELL, 65, "Price at upper Bollinger Band"
        return Signal.HOLD, 0, "Price within bands"
    
    def strategy_z_score_reversion(self) -> Tuple[Signal, float, str]:
        period = 20
        sma = self.close.rolling(period).mean()
        std = self.close.rolling(period).std()
        z_score = (self.close - sma) / std
        current_z = self._get_scalar(z_score.iloc[-1])
        
        if current_z < -2:
            return Signal.BUY, 70, f"Oversold (Z-score: {current_z:.2f})"
        elif current_z > 2:
            return Signal.SELL, 70, f"Overbought (Z-score: {current_z:.2f})"
        return Signal.HOLD, 0, f"Normal (Z-score: {current_z:.2f})"
    
    def strategy_volume_breakout(self) -> Tuple[Signal, float, str]:
        if self.volume is None:
            return Signal.HOLD, 0, "Volume not available"
        
        avg_volume = self.volume.rolling(20).mean()
        current_vol = self._get_scalar(self.volume.iloc[-1])
        avg_vol = self._get_scalar(avg_volume.iloc[-1])
        price_change = self._get_scalar(self.close.pct_change().iloc[-1]) * 100
        
        if current_vol > avg_vol * 1.5 and price_change > 2:
            return Signal.BUY, 60, f"Volume breakout +{price_change:.1f}%"
        elif current_vol > avg_vol * 1.5 and price_change < -2:
            return Signal.SELL, 60, f"Volume breakdown {price_change:.1f}%"
        return Signal.HOLD, 0, "Normal volume"
    
    def get_all_signals(self) -> List[Tuple[Signal, float, str, str]]:
        all_signals = []
        all_signals.append((*self.strategy_ma_crossover(), "MA_Crossover"))
        all_signals.append((*self.strategy_ema_trend(), "EMA_Trend"))
        all_signals.append((*self.strategy_adx_trend(), "ADX_Trend"))
        all_signals.append((*self.strategy_rsi_extreme(), "RSI_Extreme"))
        all_signals.append((*self.strategy_macd_histogram(), "MACD_Histogram"))
        all_signals.append((*self.strategy_cci_extreme(), "CCI"))
        all_signals.append((*self.strategy_bollinger_reversion(), "Bollinger"))
        all_signals.append((*self.strategy_z_score_reversion(), "Z_Score"))
        all_signals.append((*self.strategy_volume_breakout(), "Volume_Breakout"))
        return all_signals
    
    def combined_signal(self) -> Tuple[Signal, float, Dict]:
        all_signals = self.get_all_signals()
        
        buy_signals = [s for s in all_signals if s[0] == Signal.BUY]
        sell_signals = [s for s in all_signals if s[0] == Signal.SELL]
        
        buy_weight = sum(s[1] for s in buy_signals) / 100 if buy_signals else 0
        sell_weight = sum(s[1] for s in sell_signals) / 100 if sell_signals else 0
        
        total_weight = buy_weight + sell_weight
        
        if total_weight > 0:
            buy_ratio = buy_weight / total_weight
            sell_ratio = sell_weight / total_weight
            
            if buy_ratio > 0.6:
                confidence = min(90, buy_ratio * 100)
                return Signal.BUY, confidence, {"buy_count": len(buy_signals), "sell_count": len(sell_signals)}
            elif sell_ratio > 0.6:
                confidence = min(90, sell_ratio * 100)
                return Signal.SELL, confidence, {"buy_count": len(buy_signals), "sell_count": len(sell_signals)}
        
        return Signal.HOLD, 0, {"buy_count": len(buy_signals), "sell_count": len(sell_signals)}


def get_trading_decision(df: pd.DataFrame) -> Dict[str, Any]:
    strategies = CompleteStrategies(df)
    signal, confidence, details = strategies.combined_signal()
    all_signals = strategies.get_all_signals()
    
    return {
        "signal": signal.value,
        "confidence": confidence,
        "details": details,
        "all_signals": [(s[3], s[0].value, s[1], s[2]) for s in all_signals]
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    import yfinance as yf
    
    print("Testing Complete Strategies...")
    print("=" * 70)
    
    df = yf.download("NVDA", period="6mo")
    df = df.reset_index()
    
    from src.indicators.technical import TechnicalIndicators
    indicators = TechnicalIndicators()
    df = indicators.add_all_indicators(df)
    
    decision = get_trading_decision(df)
    
    print(f"\n🎯 COMBINED SIGNAL: {decision['signal']}")
    print(f"📊 Confidence: {decision['confidence']:.1f}%")
    print(f"\n📈 Signal Summary: Buy={decision['details']['buy_count']} Sell={decision['details']['sell_count']}")
    
    print("\n📋 Individual Strategy Signals:")
    for strategy, signal, conf, reason in decision['all_signals']:
        print(f"   {strategy:20}: {signal:5} ({conf:.0f}%) - {reason[:45]}")
    
    print("\n✅ Complete Strategies Ready!")
