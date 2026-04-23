"""
Order Manager Module
Simplifies placing and managing trades

Trading assistant that:
- Validates orders before sending
- Calculates position sizes
- Prevents common mistakes
"""

from typing import Optional
from src.trading.broker import AlpacaBroker


class OrderManager:
    """
    Manages order placement with safety checks.

    Why have a separate order manager?
    - Validates trades before sending
    - Calculates safe position sizes
    - Prevents costly mistakes
    """

    def __init__(self, broker: AlpacaBroker):
        """
        Initialize with a broker connection.

        Args:
            broker: Your AlpacaBroker instance
        """
        self.broker = broker
        self.account = broker.get_account()

    def calculate_position_size(
            self,
            symbol: str,
            risk_percent: float = 2.0
    ) -> int:
        """
        Calculate how many shares to buy based on risk.

        Args:
            symbol: Stock ticker
            risk_percent: % of account to risk (default 2%)

        Returns:
            Number of shares to buy

        Formula:
        position_size = (account_value * risk_percent) / share_price

        Example:
        Account: $10,000
        Risk: 2% = $200
        NVDA price: $177
        Shares = 200 / 177 ≈ 1 share

        Why this matters:
        - Never risk more than you can lose
        - Adjusts position size to volatility
        - Professional risk management
        """
        # Get current account value
        account_value = float(self.account.equity)

        # Calculate dollar amount to risk
        risk_amount = account_value * (risk_percent / 100)

        # Get current price
        quote = self.broker.get_quote(symbol)
        current_price = quote["ask_price"]  # Use ask price for buys

        # Calculate shares
        shares = int(risk_amount / current_price)

        # Minimum 1 share (if account is large enough)
        if shares == 0 and risk_amount >= current_price:
            shares = 1

        print(f"\n📐 Position Size Calculation:")
        print(f"   Account Value: ${account_value:,.2f}")
        print(f"   Risk %: {risk_percent}%")
        print(f"   Risk Amount: ${risk_amount:.2f}")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Suggested Shares: {shares}")

        return shares

    def buy_with_risk_limit(
            self,
            symbol: str,
            quantity: Optional[int] = None,
            risk_percent: float = 2.0
    ):
        """
        Buy a stock with risk management.

        Args:
            symbol: Stock to buy
            quantity: Number of shares (auto-calculated if None)
            risk_percent: % of account to risk
        """
        # Auto-calculate quantity if not provided
        if quantity is None:
            quantity = self.calculate_position_size(symbol, risk_percent)

        if quantity == 0:
            print(f"❌ Cannot buy {symbol}: Calculated quantity is 0")
            return None

        # Check if we have enough buying power
        buying_power = float(self.account.buying_power)
        quote = self.broker.get_quote(symbol)
        estimated_cost = quantity * quote["ask_price"]

        if estimated_cost > buying_power:
            print(f"❌ Insufficient buying power!")
            print(f"   Need: ${estimated_cost:.2f}")
            print(f"   Have: ${buying_power:.2f}")
            return None

        # Place the order
        return self.broker.place_market_order(symbol, quantity, "buy")

    def sell_all(self, symbol: str):
        """
        Sell all shares of a specific stock.

        Args:
            symbol: Stock to sell
        """
        positions = self.broker.get_positions()

        # Find the position
        position = next((p for p in positions if p["symbol"] == symbol), None)

        if not position:
            print(f"❌ No position found for {symbol}")
            return None

        quantity = int(position["qty"])

        if quantity == 0:
            print(f"No shares to sell for {symbol}")
            return None

        return self.broker.place_market_order(symbol, quantity, "sell")