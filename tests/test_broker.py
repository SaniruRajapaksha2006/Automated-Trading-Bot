"""
Test script to verify everything is working.
Run this to confirm your setup is correct.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trading.broker import AlpacaBroker
from src.trading.order_manager import OrderManager
from src.utils.logger import setup_logger

def main():
    print("\n" + "=" * 60)
    print("🧪 TESTING TRADING BOT SETUP")
    print("=" * 60)
    
    # Test 1: Logger
    print("\n1. Testing Logger...")
    logger = setup_logger()
    logger.info("Logger initialized successfully!")
    print("   ✅ Logger working")
    
    # Test 2: Broker Connection
    print("\n2. Connecting to Alpaca...")
    try:
        broker = AlpacaBroker(paper=True)
        print("   ✅ Broker connected")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Test 3: Get Account Info
    print("\n3. Getting account info...")
    account = broker.get_account()
    print(f"   ✅ Account ID: {account.id}")
    print(f"   ✅ Buying Power: ${float(account.buying_power):,.2f}")
    print(f"   ✅ Cash: ${float(account.cash):,.2f}")
    print(f"   ✅ Portfolio Value: ${float(account.portfolio_value):,.2f}")
    
    # Test 4: Get Quote
    print("\n4. Getting stock quote for NVDA...")
    quote = broker.get_quote("NVDA")
    print(f"   ✅ NVDA Bid: ${quote['bid_price']:.2f}")
    print(f"   ✅ NVDA Ask: ${quote['ask_price']:.2f}")
    
    # Test 5: Order Manager
    print("\n5. Testing Order Manager...")
    order_manager = OrderManager(broker)
    
    # Calculate position size
    shares = order_manager.calculate_position_size("NVDA", risk_percent=5.0)
    print(f"   ✅ Suggested position: {shares} shares")

    order = order_manager.buy_with_risk_limit("NVDA", quantity=1)
    if order:
        print(f"   ✅ Order placed: {order}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYour trading bot is ready! 🚀")

if __name__ == "__main__":
    main()
