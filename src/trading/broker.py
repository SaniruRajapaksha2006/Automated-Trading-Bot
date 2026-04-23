"""
Broker Connection Module
Connects to Alpaca API for trading and account management

Think of this as your trading assistant that communicates with the brokerage.
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# Load API keys from .env file
load_dotenv()


class AlpacaBroker:
    """
    A wrapper class for Alpaca's trading API.

    Why a wrapper?
    - Simplifies complex Alpaca code
    - Adds error handling
    - Makes trading easier to understand

    Analogy:
    - Alpaca API = Car's engine (complex, powerful)
    - This wrapper = Steering wheel (simple, intuitive)
    """

    def __init__(self, paper: bool = True):
        """
        Initialize connection to Alpaca.

        Args:
            paper: True = Paper trading (fake money)
                   False = Live trading (real money)

        What happens when you create a broker:
        1. Reads your API keys from .env
        2. Connects to Alpaca servers
        3. Verifies the connection works
        """

        # Get API keys from environment variables
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.api_secret = os.getenv("APCA_API_SECRET_KEY")

        # Validate keys exist (stop if missing)
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "❌ API keys not found!\n"
                "Make sure you have a .env file with:\n"
                "APCA_API_KEY_ID=your_key\n"
                "APCA_API_SECRET_KEY=your_secret"
            )

        # Choose paper or live trading
        if paper:
            # Paper trading URL (fake money)
            self.base_url = "https://paper-api.alpaca.markets"
            print("📝 Paper Trading Mode (using fake money)")
        else:
            # Live trading URL (real money)
            self.base_url = "https://api.alpaca.markets"
            print("💰 LIVE Trading Mode (using REAL money!)")

        # Create the trading client (our connection to Alpaca)
        # This is like dialing the phone - we now have a connection
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=paper
        )

        # Create data client (for fetching market data)
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret
        )

        # Test the connection by getting account info
        try:
            account = self.get_account()
            print(f"✅ Connected to Alpaca successfully!")
            print(f"   Account ID: {account.id}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            raise

    def get_account(self):
        """
        Get account information from Alpaca.

        Returns account details like:
        - buying_power: How much money you can spend
        - cash: Your actual cash balance
        - portfolio_value: Total value (cash + stocks)
        - equity: Net account value
        """
        return self.trading_client.get_account()

    def get_positions(self) -> List[Dict]:
        """
        Get all current stock positions (what you own).

        Returns a list of holdings with:
        - symbol: Stock ticker (e.g., "NVDA")
        - qty: Number of shares owned
        - avg_entry_price: Average purchase price
        - current_price: Latest price
        - unrealized_pl: Profit/loss on open positions
        """
        positions = self.trading_client.get_all_positions()

        # Convert to simpler dictionary format
        result = []
        for pos in positions:
            result.append({
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "avg_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc) * 100
            })
        return result

    def get_quote(self, symbol: str) -> Dict:
        """
        Get the latest price quote for a stock.

        Args:
            symbol: Stock ticker (e.g., "AAPL", "NVDA")

        Returns:
            Dictionary with bid/ask prices
        """
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self.data_client.get_stock_latest_quote(request)

        quote_data = quote[symbol]
        return {
            "symbol": symbol,
            "bid_price": float(quote_data.bid_price),
            "ask_price": float(quote_data.ask_price),
            "bid_size": quote_data.bid_size,
            "ask_size": quote_data.ask_size,
            "timestamp": quote_data.timestamp
        }

    def place_market_order(
            self,
            symbol: str,
            quantity: int,
            side: str
    ) -> Dict:
        """
        Place a market order (buy/sell immediately at current price).

        Args:
            symbol: Stock ticker (e.g., "NVDA")
            quantity: Number of shares to buy/sell
            side: "buy" or "sell"

        Returns:
            Order confirmation details

        What is a market order?
        - You say "Buy 10 shares of NVDA"
        - Alpaca executes immediately at best available price
        - Fast but price might be slightly different than expected

        Think of it as: Walking into a store and buying at the marked price
        """

        # Validate input
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        if side not in ["buy", "sell"]:
            raise ValueError(f"Side must be 'buy' or 'sell', got {side}")

        # Convert string to Alpaca's OrderSide enum
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        # Create the order request
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=order_side,
            time_in_force=TimeInForce.DAY  # Order valid for today only
        )

        # Submit the order to Alpaca
        order = self.trading_client.submit_order(order_request)

        print(f"\n📊 Order placed:")
        print(f"   Symbol: {symbol}")
        print(f"   Side: {side.upper()}")
        print(f"   Quantity: {quantity}")
        print(f"   Order ID: {order.id}")
        print(f"   Status: {order.status}")

        return {
            "order_id": order.id,
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "status": order.status,
            "filled_quantity": float(order.filled_qty) if order.filled_qty else 0,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None
        }

    def get_orders(self, status: str = "all") -> List[Dict]:
        """
        Get all orders.

        Args:
            status: "open", "closed", or "all"

        Returns:
            List of orders with details
        """
        orders = self.trading_client.get_orders(status=status)

        result = []
        for order in orders:
            result.append({
                "order_id": order.id,
                "symbol": order.symbol,
                "quantity": float(order.qty),
                "side": order.side.value,
                "status": order.status,
                "filled_quantity": float(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "created_at": order.created_at
            })
        return result

    def cancel_all_orders(self):
        """Cancel all open orders."""
        self.trading_client.cancel_orders()
        print("✅ Cancelled all open orders")