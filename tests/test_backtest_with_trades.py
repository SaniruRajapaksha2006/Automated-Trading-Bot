"""
Backtest that shows individual trades with lower threshold
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.strategies.trading_strategies import get_trading_decision

def get_scalar(value):
    """Convert Series or array to scalar float"""
    if hasattr(value, 'iloc'):
        return float(value.iloc[0])
    elif hasattr(value, 'item'):
        return float(value.item())
    else:
        return float(value)

print("Testing Backtest with Trade Logging...")
print("=" * 70)

# Download data
print("\n📥 Downloading NVDA data (1 year)...")
nvda = yf.download("NVDA", period="1y", interval="1d")
print(f"   {len(nvda)} days of data")

# Add indicators
print("📊 Adding technical indicators...")
indicators = TechnicalIndicators()
df = indicators.add_all_indicators(nvda)

# Run simple backtest with LOWER THRESHOLD
initial_capital = 10000
capital = initial_capital
position = 0
trades = []
confidence_threshold = 20  # Lowered from 30 to 20

print(f"\n💰 Running backtest (confidence threshold: {confidence_threshold}%)...")
print("-" * 70)

for i in range(50, len(df)):
    current_df = df.iloc[:i+1]
    
    # Extract scalar values safely
    current_price = get_scalar(df['Close'].iloc[i])
    current_date = df.index[i]
    
    # Get signal
    decision = get_trading_decision(current_df)
    signal = decision['signal']
    confidence = get_scalar(decision['confidence']) if hasattr(decision['confidence'], '__len__') else decision['confidence']
    
    # Get individual signals for debugging
    buy_reasons = [r for r, s, c in decision['details']['individual_signals'] if s.value == "BUY"]
    sell_reasons = [r for r, s, c in decision['details']['individual_signals'] if s.value == "SELL"]
    
    # Execute trades with lower threshold
    if signal == "BUY" and confidence >= confidence_threshold and position == 0:
        # Calculate position size (use 95% of capital)
        position = (capital * 0.95) / current_price
        capital -= position * current_price
        trades.append({
            'date': current_date,
            'type': 'BUY',
            'price': current_price,
            'shares': float(position),
            'confidence': float(confidence),
            'reasons': buy_reasons
        })
        print(f"   📈 BUY  @ ${current_price:.2f} | Shares: {position:.2f} | Conf: {confidence:.0f}% | Reasons: {len(buy_reasons)}")
        
    elif signal == "SELL" and confidence >= confidence_threshold and position > 0:
        # Sell
        capital += position * current_price
        pnl = position * (current_price - trades[-1]['price']) if trades else 0
        trades.append({
            'date': current_date,
            'type': 'SELL',
            'price': current_price,
            'shares': position,
            'pnl': float(pnl),
            'confidence': float(confidence),
            'reasons': sell_reasons
        })
        print(f"   📉 SELL @ ${current_price:.2f} | PnL: ${pnl:+.2f} | Conf: {confidence:.0f}% | Reasons: {len(sell_reasons)}")
        position = 0

# Final value
final_price = get_scalar(df['Close'].iloc[-1])
final_value = capital + (position * final_price if position > 0 else 0)
total_return = ((final_value - initial_capital) / initial_capital) * 100

print("-" * 70)
print("\n📊 RESULTS:")
print(f"   Confidence Threshold: {confidence_threshold}%")
print(f"   Initial Capital: ${initial_capital:,.2f}")
print(f"   Final Value: ${final_value:,.2f}")
print(f"   Total Return: {total_return:+.2f}%")
print(f"   Total Trades: {len(trades) // 2}")

if len(trades) >= 2:
    profits = [t['pnl'] for t in trades if t['type'] == 'SELL' and 'pnl' in t]
    winning_trades = [p for p in profits if p > 0]
    if profits:
        win_rate = (len(winning_trades) / len(profits)) * 100
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Total Profit: ${sum(profits):+.2f}")
        print(f"   Avg Profit per Trade: ${sum(profits)/len(profits):+.2f}")

if trades:
    print("\n📋 TRADE SUMMARY:")
    for i in range(0, len(trades), 2):
        if i+1 < len(trades):
            buy = trades[i]
            sell = trades[i+1]
            print(f"   {buy['date'].strftime('%Y-%m-%d')} BUY @ ${buy['price']:.2f} → {sell['date'].strftime('%Y-%m-%d')} SELL @ ${sell['price']:.2f} | PnL: ${sell['pnl']:+.2f} | Conf: {buy['confidence']:.0f}%")
else:
    print("\n⚠️ No trades executed. Try lowering the confidence threshold further.")

print("\n✅ Backtest complete!")
