"""
WebSocket Real-time Data Feed
Live price streaming from Alpaca
"""

import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Callable, Optional
import websocket
import ssl


class AlpacaWebSocket:
    """
    WebSocket connection to Alpaca for real-time market data.
    """
    
    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        """
        Initialize WebSocket connection.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        
        if paper:
            self.ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        else:
            self.ws_url = "wss://stream.data.alpaca.markets/v2"
        
        self.ws = None
        self.subscribed_symbols = set()
        self.callbacks = []
        self.running = False
        self.last_prices = {}
        self.authenticated = False
    
    def add_callback(self, callback: Callable):
        """Add callback function for price updates."""
        self.callbacks.append(callback)

    def on_message(self, ws, message):
        """Handle incoming messages"""
        try:
            data = json.loads(message)

            # Handle different message types
            if isinstance(data, list):
                for item in data:
                    # Success/control messages
                    if item.get('T') == 'success':
                        msg = item.get('msg', '')
                        if msg == 'connected':
                            print("✅ Connected to Alpaca WebSocket")
                        elif msg == 'authenticated':
                            print("✅ Authentication successful!")
                            self.authenticated = True
                            self._subscribe_after_auth()
                        elif msg == 'subscribed':
                            print(f"📡 Subscription confirmed")

                    # Subscription confirmation
                    elif item.get('T') == 'subscription':
                        print(f"📡 Subscribed to trades: {item.get('trades', [])}")

                    # Trade message (uppercase T for trade type)
                    elif item.get('T') == 't':
                        symbol = item.get('S', '')
                        price = item.get('p', 0)
                        volume = item.get('s', 0)
                        timestamp = item.get('t', '')

                        self.last_prices[symbol] = {
                            'price': price,
                            'volume': volume,
                            'timestamp': timestamp,
                            'time': datetime.now()
                        }

                        for callback in self.callbacks:
                            try:
                                callback(symbol, price, volume, timestamp)
                            except Exception as e:
                                print(f"Callback error: {e}")

                    # Quote message
                    elif item.get('T') == 'q':
                        pass

                    # Any other message type
                    else:
                        # Don't print every message to avoid spam
                        if 'msg' not in item:
                            pass

        except Exception as e:
            print(f"Error parsing message: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closing"""
        print("WebSocket connection closed")
        self.running = False
        self.authenticated = False
        
        # Attempt to reconnect after 5 seconds
        if self.running:
            time.sleep(5)
            self.connect()
    
    def on_open(self, ws):
        """Handle WebSocket opening"""
        print("WebSocket connected! Authenticating...")
        
        auth_msg = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret
        }
        ws.send(json.dumps(auth_msg))
    
    def _subscribe_after_auth(self):
        """Subscribe to symbols after authentication"""
        if self.subscribed_symbols and self.ws and self.ws.sock:
            subscribe_msg = {
                "action": "subscribe",
                "trades": list(self.subscribed_symbols),
                "quotes": list(self.subscribed_symbols)
            }
            self.ws.send(json.dumps(subscribe_msg))
            print(f"📡 Subscribed to: {list(self.subscribed_symbols)}")
    
    def connect(self):
        """Establish WebSocket connection"""
        self.running = True
        
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        wst = threading.Thread(target=self.ws.run_forever, kwargs={'sslopt': {"cert_reqs": ssl.CERT_NONE}})
        wst.daemon = True
        wst.start()
    
    def subscribe(self, symbols: List[str]):
        """
        Subscribe to real-time data for symbols.
        
        Args:
            symbols: List of stock symbols
        """
        new_symbols = set(symbols) - self.subscribed_symbols
        
        if new_symbols:
            self.subscribed_symbols.update(new_symbols)
            
            # If already authenticated, subscribe immediately
            if self.authenticated and self.ws and self.ws.sock:
                subscribe_msg = {
                    "action": "subscribe",
                    "trades": list(new_symbols),
                    "quotes": list(new_symbols)
                }
                self.ws.send(json.dumps(subscribe_msg))
                print(f"📡 Subscribed to: {list(new_symbols)}")
            else:
                print(f"⏳ Will subscribe to {list(new_symbols)} after authentication")
    
    def unsubscribe(self, symbols: List[str]):
        """
        Unsubscribe from symbols.
        
        Args:
            symbols: List of stock symbols
        """
        if self.authenticated and self.ws and self.ws.sock:
            unsubscribe_msg = {
                "action": "unsubscribe",
                "trades": symbols,
                "quotes": symbols
            }
            self.ws.send(json.dumps(unsubscribe_msg))
        
        self.subscribed_symbols -= set(symbols)
        print(f"📡 Unsubscribed from: {symbols}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Latest price or None if not available
        """
        if symbol in self.last_prices:
            return self.last_prices[symbol]['price']
        return None
    
    def get_all_prices(self) -> Dict:
        """Get all current prices"""
        return {symbol: data['price'] for symbol, data in self.last_prices.items()}
    
    def stop(self):
        """Close WebSocket connection"""
        self.running = False
        self.authenticated = False
        if self.ws:
            self.ws.close()
        print("WebSocket stopped")


def print_price(symbol: str, price: float, volume: int, timestamp: str):
    """Example callback to print prices"""
    print(f"📊 {symbol}: ${price:.2f} | Volume: {volume:,} | {timestamp}")


def update_dataframe_callback(df_dict: Dict, symbol: str, price: float, volume: int, timestamp: str):
    """Callback to update a DataFrame with live prices"""
    df_dict[symbol] = {'price': price, 'volume': volume, 'timestamp': timestamp, 'update_time': datetime.now()}


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("APCA_API_KEY_ID")
    api_secret = os.getenv("APCA_API_SECRET_KEY")
    
    if not api_key or not api_secret:
        print("❌ Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env")
        exit()
    
    print("=" * 60)
    print("Testing Alpaca WebSocket Real-time Feed")
    print("=" * 60)
    
    # Create WebSocket connection
    ws = AlpacaWebSocket(api_key, api_secret, paper=True)
    
    # Add callback to print prices
    ws.add_callback(print_price)
    
    # Set symbols to subscribe to
    symbols_to_track = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"]
    
    # Add symbols BEFORE connecting
    for symbol in symbols_to_track:
        ws.subscribed_symbols.add(symbol)
    
    print(f"\n🎯 Symbols to track: {symbols_to_track}")
    
    # Connect
    ws.connect()
    
    print("\n🔄 Waiting for connection and authentication...\n")
    print("📡 Live price feed active! (Ctrl+C to stop)\n")
    print("-" * 60)
    
    try:
        # Keep running until interrupted
        while ws.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("\n🛑 Stopping WebSocket connection...")
        ws.stop()
        print("\n✅ WebSocket stopped. Goodbye!")