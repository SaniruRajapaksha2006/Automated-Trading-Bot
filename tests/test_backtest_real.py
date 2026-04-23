"""
Test backtest with real stock data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.backtest.engine import BacktestEngine

print("Testing Backtest with REAL stock data...")
print("=" * 60)

# Download real stock data (1 year of NVDA)
print("\n📥 Downloading NVDA data...")
nvda = yf.download("NVDA", period="1y", interval="1d")
print(f"   Downloaded {len(nvda)} days of data")

# Add indicators
print("\n📊 Adding technical indicators...")
indicators = TechnicalIndicators()
df = indicators.add_all_indicators(nvda)
print(f"   Added {len(df.columns)} columns")

# Run backtest
print("\n💰 Running backtest with $10,000...")
backtest = BacktestEngine(initial_capital=10000)
results = backtest.run(df)

print("\n" + "=" * 60)
print("📊 BACKTEST RESULTS")
print("=" * 60)

for key, value in results.items():
    if isinstance(value, float):
        if 'equity' in key or 'capital' in key:
            print(f"   {key:<20}: ${value:>15,.2f}")
        elif 'pnl' in key:
            print(f"   {key:<20}: ${value:>15,.2f}")
        elif 'max_drawdown' in key:
            print(f"   {key:<20}: {value:>15.2f}%")
        else:
            print(f"   {key:<20}: {value:>15.2f}%")
    else:
        print(f"   {key:<20}: {value}")

print("\n✅ Backtest complete!")
