"""
Test with a simple moving average strategy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf

def get_scalar(value):
    """Convert Series or array to scalar float"""
    if hasattr(value, 'iloc'):
        return float(value.iloc[0])
    elif hasattr(value, 'item'):
        return float(value.item())
    else:
        return float(value)

print("Testing Simple Strategy...")
print("=" * 60)

# Download data
print("\n📥 Downloading NVDA data...")
nvda = yf.download("NVDA", period="6mo", interval="1d")
print(f"   {len(nvda)} days")

# Calculate just SMA
nvda['SMA_20'] = nvda['Close'].rolling(20).mean()
nvda['SMA_50'] = nvda['Close'].rolling(50).mean()

# Generate simple signals
nvda['Signal'] = 0
nvda.loc[nvda['SMA_20'] > nvda['SMA_50'], 'Signal'] = 1  # Bullish
nvda.loc[nvda['SMA_20'] < nvda['SMA_50'], 'Signal'] = -1  # Bearish

# Calculate returns
nvda['Returns'] = nvda['Close'].pct_change()
nvda['Strategy_Returns'] = nvda['Signal'].shift(1) * nvda['Returns']

# Calculate performance (extract scalar values)
total_return_series = (1 + nvda['Strategy_Returns']).prod() - 1
total_return = get_scalar(total_return_series)

buy_hold_return_series = (nvda['Close'].iloc[-1] / nvda['Close'].iloc[0]) - 1
buy_hold_return = get_scalar(buy_hold_return_series)

print(f"\n📊 Strategy Results:")
print(f"   Strategy Return: {total_return:.2%}")
print(f"   Buy & Hold Return: {buy_hold_return:.2%}")
print(f"   Outperformance: {(total_return - buy_hold_return):.2%}")

# Count signals (convert to scalar)
buy_signals = int((nvda['Signal'] == 1).sum())
sell_signals = int((nvda['Signal'] == -1).sum())

print(f"\n📈 Signals Generated:")
print(f"   Buy Signals: {buy_signals}")
print(f"   Sell Signals: {sell_signals}")

print("\n✅ Simple strategy test complete!")
