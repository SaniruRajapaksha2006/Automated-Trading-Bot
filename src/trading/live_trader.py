"""
Live Trading Integration - With Real Alpaca Orders
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

import yfinance as yf

from src.indicators.technical import TechnicalIndicators
from src.risk.risk_manager import RiskManager

# Add Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from dotenv import load_dotenv

load_dotenv()


def get_scalar(value):
    """Convert any value to a scalar float"""
    if hasattr(value, 'iloc'):
        return float(value.iloc[0])
    elif hasattr(value, 'item'):
        return float(value.item())
    else:
        return float(value)


class SimpleLiveTrader:
    """
    Live Trading System with Real Alpaca Orders
    """

    def __init__(self,
                 symbols: List[str] = None,
                 interval: int = 60,
                 paper_trading: bool = True):

        self.symbols = symbols or ["NVDA", "AAPL", "MSFT", "GOOGL"]
        self.interval = interval
        self.paper_trading = paper_trading

        # Initialize components
        self.indicators = TechnicalIndicators()
        self.risk_manager = RiskManager(initial_capital=10000)
        self.trade_history = []

        # Initialize Alpaca client
        self.trading_client = TradingClient(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'),
            paper=paper_trading
        )

        print(f"🚀 Live Trader Initialized")
        print(f"   Symbols: {self.symbols}")
        print(f"   Interval: {interval}s")
        print(f"   Mode: {'Paper Trading' if paper_trading else 'LIVE'}")

        # Verify connection
        account = self.trading_client.get_account()
        print(f"   ✅ Connected to Alpaca")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")

    def get_signal(self, df: pd.DataFrame) -> Dict:
        """
        Generate trading signal from DataFrame.

        Returns:
            Dictionary with signal and confidence
        """
        try:
            # Get latest values safely
            close = get_scalar(df['Close'].iloc[-1])
            ema20 = get_scalar(df['EMA_20'].iloc[-1]) if 'EMA_20' in df.columns else close
            macd_hist = get_scalar(df['MACD_Histogram'].iloc[-1]) if 'MACD_Histogram' in df.columns else 0

            # Simple signal logic
            buy_signals = 0
            sell_signals = 0

            # EMA Trend
            if close > ema20:
                buy_signals += 60
            else:
                sell_signals += 40

            # MACD Momentum
            if macd_hist > 0:
                buy_signals += 30
            else:
                sell_signals += 20

            total = buy_signals + sell_signals
            if total > 0:
                confidence = int((buy_signals / total) * 100)
            else:
                confidence = 0

            if confidence > 60:
                action = "BUY"
            elif confidence < 40:
                action = "SELL"
            else:
                action = "HOLD"

            return {
                "action": action,
                "confidence": confidence,
                "buy_weight": buy_signals,
                "sell_weight": sell_signals,
                "current_price": close,
                "ema20": ema20,
                "macd_hist": macd_hist
            }
        except Exception as e:
            print(f"   ❌ Signal error: {e}")
            return {"action": "HOLD", "confidence": 0}

    def fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch and process data for symbol"""
        try:
            # Fetch 2 months of data
            df = yf.download(symbol, period="2mo", progress=False)

            if len(df) < 30:
                df = yf.download(symbol, period="3mo", progress=False)

            if len(df) < 20:
                print(f"   ⚠️ Insufficient data for {symbol}: {len(df)} rows")
                return None

            df = df.reset_index()

            # Add basic indicators
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

            # MACD
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

            return df
        except Exception as e:
            print(f"   ❌ Fetch error: {e}")
            return None

    def get_pending_orders(self, symbol: str = None):
        """Get pending (not yet filled) orders"""
        try:
            orders = self.trading_client.get_orders()
            pending = []
            for order in orders:
                if order.status in ['new', 'pending_new', 'accepted', 'partially_filled']:
                    if symbol is None or order.symbol == symbol:
                        pending.append(order)
            return pending
        except:
            return []

    def check_symbol(self, symbol: str):
        """Check a single symbol and execute trades via Alpaca"""
        print(f"\n{'='*50}")
        print(f"📊 Checking {symbol} at {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*50}")

        df = self.fetch_data(symbol)
        if df is None or len(df) < 20:
            print(f"   ⚠️ No data for {symbol}")
            return

        signal = self.get_signal(df)
        current_price = signal['current_price']

        print(f"   Price: ${current_price:.2f}")
        print(f"   Signal: {signal['action']} (Confidence: {signal['confidence']}%)")
        print(f"   EMA20: ${signal['ema20']:.2f}")
        print(f"   MACD Hist: {signal['macd_hist']:.4f}")

        # Check current positions from Alpaca
        try:
            alpaca_positions = self.trading_client.get_all_positions()
            position_symbols = [p.symbol for p in alpaca_positions]
        except:
            position_symbols = []

        # Check for pending orders first
        pending_orders = self.get_pending_orders(symbol)
        if pending_orders:
            print(f"   ⏳ Already have pending order for {symbol}, skipping...")
            return

        # Execute BUY order
        if signal['action'] == "BUY" and signal['confidence'] > 50:
            if symbol not in position_symbols:
                # Get account info for position sizing
                account = self.trading_client.get_account()
                buying_power = float(account.buying_power)

                # Calculate shares (use 10% of buying power)
                shares = int((buying_power * 0.1) / current_price)
                if shares > 0:
                    print(f"   🟢 Placing BUY order for {shares} shares of {symbol} @ ${current_price:.2f}")

                    try:
                        # Send REAL order to Alpaca
                        order_request = MarketOrderRequest(
                            symbol=symbol,
                            qty=shares,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                        order = self.trading_client.submit_order(order_request)
                        print(f"   ✅ ORDER SUBMITTED! Order ID: {order.id}")
                        print(f"   Status: {order.status}")

                        self.trade_history.append({
                            "timestamp": datetime.now(),
                            "symbol": symbol,
                            "action": "BUY",
                            "shares": shares,
                            "price": current_price,
                            "order_id": order.id
                        })
                    except Exception as e:
                        print(f"   ❌ Order failed: {e}")
                else:
                    print(f"   ⚠️ Not enough buying power for {symbol}")
            else:
                print(f"   📊 Already have position in {symbol}")

        # Execute SELL order
        elif signal['action'] == "SELL" and signal['confidence'] > 50:
            if symbol in position_symbols:
                # Find position quantity
                for pos in alpaca_positions:
                    if pos.symbol == symbol:
                        shares = int(float(pos.qty))
                        print(f"   🔴 Placing SELL order for {shares} shares of {symbol} @ ${current_price:.2f}")

                        try:
                            order_request = MarketOrderRequest(
                                symbol=symbol,
                                qty=shares,
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.DAY
                            )
                            order = self.trading_client.submit_order(order_request)
                            print(f"   ✅ SELL ORDER SUBMITTED! Order ID: {order.id}")
                            print(f"   Status: {order.status}")

                            self.trade_history.append({
                                "timestamp": datetime.now(),
                                "symbol": symbol,
                                "action": "SELL",
                                "shares": shares,
                                "price": current_price,
                                "order_id": order.id
                            })
                        except Exception as e:
                            print(f"   ❌ Sell order failed: {e}")
                        break
            else:
                print(f"   📊 No position in {symbol} to sell")
        else:
            print(f"   📊 HOLD - No clear signal")

    def check_positions_for_exits(self):
        """Check open positions for stop loss/take profit triggers"""
        try:
            positions = self.trading_client.get_all_positions()

            for position in positions:
                symbol = position.symbol
                current_price = float(position.current_price)
                entry_price = float(position.avg_entry_price)
                qty = int(float(position.qty))

                # Calculate P&L percentage
                pnl_pct = ((current_price - entry_price) / entry_price) * 100

                # Stop Loss: Sell if down 3%
                if pnl_pct <= -3.0:
                    print(f"\n   🔴 STOP LOSS TRIGGERED for {symbol}")
                    print(f"      Entry: ${entry_price:.2f} | Current: ${current_price:.2f} | Loss: {pnl_pct:.1f}%")

                    order_request = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    order = self.trading_client.submit_order(order_request)
                    print(f"      ✅ Stop loss order submitted! Order ID: {order.id}")

                    self.trade_history.append({
                        "timestamp": datetime.now(),
                        "symbol": symbol,
                        "action": "STOP_LOSS",
                        "shares": qty,
                        "price": current_price,
                        "pnl": qty * (current_price - entry_price)
                    })

                # Take Profit: Sell if up 5%
                elif pnl_pct >= 5.0:
                    print(f"\n   🟢 TAKE PROFIT TRIGGERED for {symbol}")
                    print(f"      Entry: ${entry_price:.2f} | Current: ${current_price:.2f} | Profit: {pnl_pct:.1f}%")

                    order_request = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    order = self.trading_client.submit_order(order_request)
                    print(f"      ✅ Take profit order submitted! Order ID: {order.id}")

                    self.trade_history.append({
                        "timestamp": datetime.now(),
                        "symbol": symbol,
                        "action": "TAKE_PROFIT",
                        "shares": qty,
                        "price": current_price,
                        "pnl": qty * (current_price - entry_price)
                    })

        except Exception as e:
            print(f"Error checking positions: {e}")

    def print_performance(self):
        """Print performance summary"""
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()

            # FIX: Remove 'limit' parameter - use get_orders() without arguments
            orders = self.trading_client.get_orders()

            print("\n" + "=" * 50)
            print("📊 PERFORMANCE SUMMARY")
            print("=" * 50)

            # Account performance
            initial_capital = 100000.0
            current_equity = float(account.equity)
            total_pnl = current_equity - initial_capital
            total_return = (total_pnl / initial_capital) * 100

            print(f"💰 Account:")
            print(f"   Initial Capital: ${initial_capital:,.2f}")
            print(f"   Current Equity: ${current_equity:,.2f}")
            print(f"   Total P&L: ${total_pnl:+,.2f}")
            print(f"   Total Return: {total_return:+.2f}%")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")

            # Open positions
            if positions:
                print(f"\n📈 OPEN POSITIONS ({len(positions)}):")
                for pos in positions:
                    entry = float(pos.avg_entry_price)
                    current = float(pos.current_price)
                    pnl = float(pos.unrealized_pl)
                    pnl_pct = ((current - entry) / entry) * 100
                    print(
                        f"   {pos.symbol}: {pos.qty} shares | Entry: ${entry:.2f} | Current: ${current:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")

            # Recent orders (last 10)
            if orders:
                print(f"\n📋 RECENT ORDERS (last 10):")
                filled_count = 0
                for order in orders[-10:]:  # Get last 10 using list slicing
                    if order.status == 'filled':
                        filled_count += 1
                        print(
                            f"   ✅ {order.symbol}: {order.side} {order.filled_qty} shares @ ${float(order.filled_avg_price):.2f}")
                    elif order.status in ['new', 'pending_new', 'accepted']:
                        print(f"   ⏳ {order.symbol}: {order.side} {order.qty} shares - {order.status} (not yet filled)")
                    else:
                        print(f"   {order.symbol}: {order.side} {order.qty} shares - {order.status}")

                if filled_count == 0 and orders:
                    print(f"   ⚠️ No filled orders found.")

            print("=" * 50)
        except Exception as e:
            print(f"Error getting performance: {e}")

    def print_trade_summary(self):
        """Print detailed summary of all trades"""
        if not self.trade_history:
            print("\n📊 No trades executed yet")
            return

        print("\n" + "=" * 60)
        print("📊 COMPLETE TRADE HISTORY")
        print("=" * 60)

        # Group by action
        buys = [t for t in self.trade_history if t['action'] == 'BUY']
        sells = [t for t in self.trade_history if t['action'] == 'SELL']
        stop_losses = [t for t in self.trade_history if t['action'] == 'STOP_LOSS']
        take_profits = [t for t in self.trade_history if t['action'] == 'TAKE_PROFIT']

        print(f"\n📈 BUYS ({len(buys)}):")
        for t in buys:
            print(
                f"   {t['timestamp'].strftime('%H:%M:%S')} | {t['symbol']} | {t['shares']} shares @ ${t['price']:.2f}")

        print(f"\n📉 SELLS ({len(sells)}):")
        for t in sells:
            pnl = t.get('pnl', 0)
            print(
                f"   {t['timestamp'].strftime('%H:%M:%S')} | {t['symbol']} | {t['shares']} shares @ ${t['price']:.2f} | P&L: ${pnl:+.2f}")

        print(f"\n🔴 STOP LOSSES ({len(stop_losses)}):")
        for t in stop_losses:
            print(f"   {t['timestamp'].strftime('%H:%M:%S')} | {t['symbol']} | Loss: ${t['pnl']:+.2f}")

        print(f"\n🟢 TAKE PROFITS ({len(take_profits)}):")
        for t in take_profits:
            print(f"   {t['timestamp'].strftime('%H:%M:%S')} | {t['symbol']} | Profit: ${t['pnl']:+.2f}")

        # Calculate totals
        total_pnl = sum(t.get('pnl', 0) for t in self.trade_history if 'pnl' in t)
        winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]

        print(f"\n💰 TOTAL P&L: ${total_pnl:+.2f}")
        print(f"📊 WINNING TRADES: {len(winning_trades)}")
        print(f"📉 LOSING TRADES: {len(losing_trades)}")
        if len(winning_trades) + len(losing_trades) > 0:
            win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) * 100
            print(f"🎯 WIN RATE: {win_rate:.1f}%")

        print("=" * 60)

    def run_once(self):
        """Run one trading cycle"""
        print(f"\n{'#' * 60}")
        print(f"🔄 TRADING CYCLE at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#' * 60}")

        # Get account info
        account = self.trading_client.get_account()
        print(f"💰 Buying Power: ${float(account.buying_power):,.2f}")
        print(f"💰 Portfolio Value: ${float(account.portfolio_value):,.2f}")

        # Check existing positions for stop loss/take profit
        self.check_positions_for_exits()

        for symbol in self.symbols:
            try:
                self.check_symbol(symbol)
            except Exception as e:
                print(f"❌ Error checking {symbol}: {e}")

        # Show trade summary (ADD THIS)
        self.print_trade_summary()

        # Show performance
        self.print_performance()

    def run_continuous(self):
        """Run continuously"""
        print("\n🚀 Starting continuous trading...")
        print("   Press Ctrl+C to stop\n")

        try:
            while True:
                self.run_once()
                print(f"\n💤 Sleeping for {self.interval} seconds...")
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping...")
            self.stop()

    def stop(self):
        """Stop the bot"""
        print("\n📊 Final Summary:")
        print(f"   Total Trades: {len(self.trade_history)}")
        print("\n✅ Trading bot stopped")


if __name__ == "__main__":
    print("=" * 70)
    print("LIVE TRADING SYSTEM - WITH ALPACA ORDERS")
    print("=" * 70)

    trader = SimpleLiveTrader(
        symbols=["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"],
        interval=60,
        paper_trading=True
    )

    # Run one test cycle
    print("\n📋 Running one test cycle...")
    trader.run_continuous()

    print("\n✅ Live Trader Ready!")