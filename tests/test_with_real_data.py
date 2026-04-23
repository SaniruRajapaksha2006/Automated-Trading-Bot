"""
Test strategies with real stock data from yfinance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
from src.indicators.technical import TechnicalIndicators
from src.strategies.trading_strategies import get_trading_decision

print("Testing with REAL stock data...")
print("=" * 50)

# Download real NVDA data
print("\nDownloading NVDA data...")
nvda = yf.download("NVDA", period="3mo", interval="1d")
print(f"   Downloaded {len(nvda)} days of data")

# Add indicators
indicators = TechnicalIndicators()
df = indicators.add_all_indicators(nvda)

# Get trading decision
decision = get_trading_decision(df)

print(f"\nNVDA Trading Decision:")
print(f"   Signal: {decision['signal']}")
print(f"   Confidence: {decision['confidence']:.1f}%")
print(f"   Votes: Buy={decision['details']['buy_votes']} | Sell={decision['details']['sell_votes']} | Hold={decision['details']['hold_votes']}")

# Show individual signals
print(f"\nIndividual Strategy Signals:")
for reason, signal, conf in decision['details']['individual_signals']:
    print(f"   {signal.value:6} | {reason[:60]}")

# Test with AAPL
print("\n" + "=" * 50)
print("\nDownloading AAPL data...")
aapl = yf.download("AAPL", period="3mo", interval="1d")
df = indicators.add_all_indicators(aapl)
decision = get_trading_decision(df)

print(f"\nAAPL Trading Decision:")
print(f"   Signal: {decision['signal']}")
print(f"   Confidence: {decision['confidence']:.1f}%")
print(f"   Votes: Buy={decision['details']['buy_votes']} | Sell={decision['details']['sell_votes']} | Hold={decision['details']['hold_votes']}")

# Test with TSLA
print("\n" + "=" * 50)
print("\nDownloading TSLA data...")
tsla = yf.download("TSLA", period="3mo", interval="1d")
df = indicators.add_all_indicators(tsla)
decision = get_trading_decision(df)

print(f"\nTSLA Trading Decision:")
print(f"   Signal: {decision['signal']}")
print(f"   Confidence: {decision['confidence']:.1f}%")
print(f"   Votes: Buy={decision['details']['buy_votes']} | Sell={decision['details']['sell_votes']} | Hold={decision['details']['hold_votes']}")

print("\n" + "=" * 50)
print("Real data test complete!")