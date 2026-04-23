"""
Check what signals are being generated
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from src.indicators.technical import TechnicalIndicators
from src.strategies.trading_strategies import get_trading_decision

print("Checking Trading Signals...")
print("=" * 60)

# Download data
print("\n📥 Downloading NVDA data...")
nvda = yf.download("NVDA", period="3mo", interval="1d")
print(f"   {len(nvda)} days")

# Add indicators
indicators = TechnicalIndicators()
df = indicators.add_all_indicators(nvda)

# Check signals for last 10 days
print("\n📊 Signal Analysis (Last 10 days):")
print("-" * 60)

for i in range(-10, 0):
    current_df = df.iloc[:i]
    if len(current_df) < 50:
        continue
    
    decision = get_trading_decision(current_df)
    signal = decision['signal']
    confidence = decision['confidence']
    
    # Get individual votes
    buy_votes = decision['details']['buy_votes']
    sell_votes = decision['details']['sell_votes']
    hold_votes = decision['details']['hold_votes']
    
    date = df.index[i]
    print(f"   {date.strftime('%Y-%m-%d')}: {signal:5} | Conf: {confidence:.0f}% | Votes: B={buy_votes} S={sell_votes} H={hold_votes}")

print("\n✅ Signal check complete!")
