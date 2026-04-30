"""
Risk Management Module
Protects capital with stop losses, position sizing, and drawdown limits
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    current_price: float = 0.0


class RiskManager:
    """
    Professional risk management for trading.

    Features:
    - Position sizing (Kelly, Fixed %, Volatility-based)
    - Stop loss (Fixed, Trailing, ATR-based)
    - Take profit (Risk/Reward ratios)
    - Maximum drawdown protection
    - Portfolio risk limits
    """

    def __init__(self, initial_capital: float = 10000):
        """
        Initialize risk manager.

        Args:
            initial_capital: Starting account balance
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_drawdown = 0
        self.open_positions: Dict[str, Position] = {}
        self.trade_history = []

        # Risk limits
        self.max_position_size_pct = 0.20  # Max 20% per position
        self.max_portfolio_risk_pct = 0.06  # Max 6% total portfolio risk
        self.max_daily_loss_pct = 0.03  # Stop trading after 3% daily loss
        self.daily_pnl = 0

    # ================================================================
    # POSITION SIZING
    # ================================================================

    def calculate_position_size_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Kelly Criterion position sizing.

        Formula: f* = (p * b - q) / b
        where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = win/loss ratio

        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount

        Returns:
            Optimal position size as fraction of capital (0-0.25)
        """
        if avg_loss == 0:
            return 0

        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b

        # Limit to 25% of capital (conservative)
        return min(max(kelly, 0), 0.25)

    def calculate_position_size_volatility(self, symbol: str, atr: float, current_price: float) -> float:
        """
        Volatility-based position sizing (adjusts for risk).

        Formula: Risk Amount / (ATR * ATR_Multiplier)

        Args:
            symbol: Stock symbol
            atr: Average True Range
            current_price: Current price

        Returns:
            Number of shares to buy
        """
        # Risk 1% of capital per trade
        risk_amount = self.current_capital * 0.01

        # ATR-based stop distance (2x ATR)
        stop_distance = atr * 2

        if stop_distance == 0:
            stop_distance = current_price * 0.02  # Fallback to 2%

        shares = risk_amount / stop_distance

        # Cap at 20% of portfolio
        max_shares = (self.current_capital * 0.20) / current_price

        return min(shares, max_shares)

    def calculate_position_size_fixed(self, current_price: float) -> float:
        """
        Fixed percentage position sizing.

        Args:
            current_price: Current price

        Returns:
            Number of shares to buy
        """
        position_value = self.current_capital * self.max_position_size_pct
        return position_value / current_price

    # ================================================================
    # STOP LOSS CALCULATIONS
    # ================================================================

    def calculate_stop_loss_atr(self, entry_price: float, atr: float, multiplier: float = 2.0) -> float:
        """
        ATR-based stop loss (adapts to volatility).

        Args:
            entry_price: Entry price
            atr: Average True Range
            multiplier: ATR multiplier (2 = stop at 2x ATR)

        Returns:
            Stop loss price
        """
        return entry_price - (atr * multiplier)

    def calculate_stop_loss_percent(self, entry_price: float, percent: float = 0.05) -> float:
        """
        Fixed percentage stop loss.

        Args:
            entry_price: Entry price
            percent: Percentage drop to trigger stop (5% default)

        Returns:
            Stop loss price
        """
        return entry_price * (1 - percent)

    def calculate_trailing_stop(self, current_price: float, highest_price: float, trail_percent: float = 0.05) -> float:
        """
        Trailing stop loss (moves up with price).

        Args:
            current_price: Current price
            highest_price: Highest price since entry
            trail_percent: Trail percentage (5% default)

        Returns:
            Adjusted stop loss price
        """
        return highest_price * (1 - trail_percent)

    # ================================================================
    # TAKE PROFIT CALCULATIONS
    # ================================================================

    def calculate_take_profit_risk_reward(self, entry_price: float, stop_loss: float, ratio: float = 2.0) -> float:
        """
        Risk/Reward based take profit.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            ratio: Risk/Reward ratio (2:1 default)

        Returns:
            Take profit price
        """
        risk = entry_price - stop_loss
        return entry_price + (risk * ratio)

    def calculate_take_profit_percent(self, entry_price: float, percent: float = 0.10) -> float:
        """
        Fixed percentage take profit.

        Args:
            entry_price: Entry price
            percent: Percentage gain target (10% default)

        Returns:
            Take profit price
        """
        return entry_price * (1 + percent)

    # ================================================================
    # PORTFOLIO RISK
    # ================================================================

    def check_drawdown_limit(self) -> Tuple[bool, float]:
        """
        Check if maximum drawdown limit is hit.

        Returns:
            (should_stop_trading, current_drawdown_pct)
        """
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # Calculate drawdown
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Stop trading if drawdown > 15%
        should_stop = drawdown > 15.0

        return should_stop, drawdown

    def check_daily_loss_limit(self) -> Tuple[bool, float]:
        """
        Check if daily loss limit is hit.

        Returns:
            (should_stop_trading, daily_loss_pct)
        """
        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl) / self.initial_capital * 100
            should_stop = daily_loss_pct > self.max_daily_loss_pct * 100
            return should_stop, daily_loss_pct

        return False, 0

    def calculate_correlation_risk(self, positions: List[str]) -> float:
        """
        Calculate correlation risk between positions.

        Args:
            positions: List of symbols in portfolio

        Returns:
            Correlation risk factor (higher = more correlated)
        """
        # Simplified: tech stocks are highly correlated
        tech_stocks = {'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA'}

        tech_count = sum(1 for s in positions if s in tech_stocks)

        if tech_count >= 3:
            return 0.8  # High correlation risk
        elif tech_count == 2:
            return 0.5
        else:
            return 0.2

    # ================================================================
    # POSITION MANAGEMENT
    # ================================================================

    def open_position(self, symbol: str, quantity: float, price: float, stop_loss: float, take_profit: float):
        """
        Open a new position.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_price=price
        )

        self.open_positions[symbol] = position

        # Reduce capital
        position_value = quantity * price
        self.current_capital -= position_value

        print(f"\n🟢 OPENED POSITION: {symbol}")
        print(f"   Shares: {quantity:.2f}")
        print(f"   Entry: ${price:.2f}")
        print(f"   Stop Loss: ${stop_loss:.2f}")
        print(f"   Take Profit: ${take_profit:.2f}")
        print(f"   Position Value: ${position_value:,.2f}")

    def close_position(self, symbol: str, price: float, reason: str):
        """
        Close an existing position.

        Args:
            symbol: Stock symbol
            price: Exit price
            reason: Why closed (stop loss, take profit, manual)
        """
        if symbol not in self.open_positions:
            return

        position = self.open_positions[symbol]
        pnl = position.quantity * (price - position.entry_price)
        pnl_pct = (pnl / (position.quantity * position.entry_price)) * 100

        # Update capital
        self.current_capital += position.quantity * price
        self.daily_pnl += pnl

        # Record trade
        self.trade_history.append({
            'symbol': symbol,
            'entry_price': position.entry_price,
            'exit_price': price,
            'quantity': position.quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'reason': reason,
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit
        })

        print(f"\n🔴 CLOSED POSITION: {symbol}")
        print(f"   Reason: {reason}")
        print(f"   Exit Price: ${price:.2f}")
        print(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")

        # Remove position
        del self.open_positions[symbol]

    def update_position_price(self, symbol: str, current_price: float):
        """
        Update current price and check stops/targets.

        Args:
            symbol: Stock symbol
            current_price: Current market price
        """
        if symbol not in self.open_positions:
            return

        position = self.open_positions[symbol]
        position.current_price = current_price

        # Check stop loss
        if current_price <= position.stop_loss:
            self.close_position(symbol, current_price, "STOP_LOSS")

        # Check take profit
        elif current_price >= position.take_profit:
            self.close_position(symbol, current_price, "TAKE_PROFIT")

    def update_trailing_stop(self, symbol: str, current_price: float, highest_price: float):
        """
        Update trailing stop for a position.

        Args:
            symbol: Stock symbol
            current_price: Current price
            highest_price: Highest price since entry
        """
        if symbol not in self.open_positions:
            return

        position = self.open_positions[symbol]
        new_stop = self.calculate_trailing_stop(current_price, highest_price)

        # Only move stop UP (never down)
        if new_stop > position.stop_loss:
            position.stop_loss = new_stop
            print(f"   📈 Trailing stop moved to ${new_stop:.2f} for {symbol}")

    # ================================================================
    # RISK METRICS
    # ================================================================

    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics.

        Returns:
            Dictionary with risk metrics
        """
        # Calculate Sharpe ratio (simplified)
        if len(self.trade_history) > 0:
            returns = [t['pnl_pct'] for t in self.trade_history]
            sharpe = np.mean(returns) / (np.std(returns) + 0.01)
        else:
            sharpe = 0

        # Calculate win rate
        if len(self.trade_history) > 0:
            winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(self.trade_history) * 100
        else:
            win_rate = 0

        # Calculate profit factor
        if len(self.trade_history) > 0:
            gross_profit = sum(t['pnl'] for t in self.trade_history if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in self.trade_history if t['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            profit_factor = 0

        return {
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "current_drawdown": (self.peak_capital - self.current_capital) / self.peak_capital * 100,
            "max_drawdown": self.max_drawdown,
            "open_positions": len(self.open_positions),
            "total_trades": len(self.trade_history),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "daily_pnl": self.daily_pnl
        }

    def get_position_summary(self) -> str:
        """
        Get formatted summary of open positions.
        """
        if not self.open_positions:
            return "No open positions"

        summary = "\n📊 OPEN POSITIONS:\n"
        summary += "-" * 60 + "\n"

        for symbol, pos in self.open_positions.items():
            pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price * 100
            summary += f"  {symbol}: {pos.quantity:.2f} shares @ ${pos.entry_price:.2f}\n"
            summary += f"    Current: ${pos.current_price:.2f} | P&L: {pnl_pct:+.2f}%\n"
            summary += f"    Stop: ${pos.stop_loss:.2f} | Target: ${pos.take_profit:.2f}\n"

        return summary


if __name__ == "__main__":
    # Test risk manager
    print("Testing Risk Manager...")
    print("=" * 60)

    rm = RiskManager(initial_capital=10000)

    # Test position sizing
    print("\n1. Position Sizing:")
    kelly_size = rm.calculate_position_size_kelly(win_rate=0.6, avg_win=200, avg_loss=100)
    print(f"   Kelly Criterion: {kelly_size:.2%} of capital")

    # Test stop loss
    print("\n2. Stop Loss:")
    atr_stop = rm.calculate_stop_loss_atr(entry_price=100, atr=5, multiplier=2)
    print(f"   ATR-based stop: ${atr_stop:.2f}")

    percent_stop = rm.calculate_stop_loss_percent(entry_price=100, percent=0.05)
    print(f"   Percentage stop (5%): ${percent_stop:.2f}")

    # Test take profit
    print("\n3. Take Profit:")
    tp = rm.calculate_take_profit_risk_reward(entry_price=100, stop_loss=95, ratio=2)
    print(f"   Risk/Reward (2:1): ${tp:.2f}")

    # Test open position
    print("\n4. Opening Position:")
    rm.open_position("NVDA", 10, 100, stop_loss=95, take_profit=110)
    rm.update_position_price("NVDA", 105)

    # Test metrics
    print("\n5. Risk Metrics:")
    metrics = rm.get_risk_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")

    print("\nRisk Manager Ready!")