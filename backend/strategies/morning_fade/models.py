"""
Models for the Morning Fade strategy.

Includes CreditSpreadTrade for tracking two-leg spread positions.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional


@dataclass
class CreditSpreadTrade:
    """
    Represents a credit spread position (call credit spread or put credit spread).

    A credit spread involves:
    - Selling an OTM option (receiving premium)
    - Buying a further OTM option (paying premium, defining max loss)
    - Net credit = premium received - premium paid

    For Morning Fade:
    - Call Credit Spread: Sell call, buy higher strike call (bearish)
    - Put Credit Spread: Sell put, buy lower strike put (bullish)
    """

    # === Entry Information ===
    entry_time: datetime
    underlying_price: float  # Underlying at entry
    direction: str  # "CALL_SPREAD" or "PUT_SPREAD"

    # === Spread Legs ===
    short_strike: float  # Strike of option we SOLD (higher premium)
    long_strike: float  # Strike of option we BOUGHT (defines max loss)
    expiration: date  # Expiration date (usually same day for 0DTE)

    # === Pricing ===
    short_entry_price: float  # Premium received per share
    long_entry_price: float  # Premium paid per share
    net_credit: float  # short_price - long_price (per share)

    # === Risk Metrics ===
    spread_width: float  # Difference between strikes
    max_profit: float  # net_credit * 100 * contracts
    max_loss: float  # (spread_width - net_credit) * 100 * contracts
    contracts: int = 1

    # === Exit Information ===
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None

    # Exit prices
    short_exit_price: Optional[float] = None  # Cost to buy back short
    long_exit_price: Optional[float] = None  # Premium from selling long
    net_debit: Optional[float] = None  # Cost to close the spread

    # P&L
    pnl: Optional[float] = None  # Realized P&L

    # === Context ===
    opening_move_pct: float = 0.0  # How much market moved in first 30 min
    entry_iv: Optional[float] = None  # IV at entry
    notes: str = ""

    def __post_init__(self):
        """Validate and compute derived fields."""
        # Ensure spread_width is positive
        if self.spread_width <= 0:
            self.spread_width = abs(self.short_strike - self.long_strike)

        # Compute max profit/loss if not provided
        if self.max_profit == 0:
            self.max_profit = self.net_credit * 100 * self.contracts

        if self.max_loss == 0:
            self.max_loss = (self.spread_width - self.net_credit) * 100 * self.contracts

    @property
    def is_call_spread(self) -> bool:
        """True if this is a call credit spread (bearish)."""
        return self.direction == "CALL_SPREAD"

    @property
    def is_put_spread(self) -> bool:
        """True if this is a put credit spread (bullish)."""
        return self.direction == "PUT_SPREAD"

    @property
    def is_closed(self) -> bool:
        """True if the position has been closed."""
        return self.exit_time is not None

    @property
    def risk_reward_ratio(self) -> float:
        """Max loss / max profit ratio."""
        if self.max_profit <= 0:
            return float("inf")
        return self.max_loss / self.max_profit

    def calculate_current_value(
        self,
        short_price: float,
        long_price: float,
    ) -> float:
        """
        Calculate current spread value (cost to close).

        Args:
            short_price: Current price of short option
            long_price: Current price of long option

        Returns:
            Net debit to close the spread (positive = cost)
        """
        # To close: buy back short, sell long
        return (short_price - long_price) * 100 * self.contracts

    def calculate_unrealized_pnl(
        self,
        short_price: float,
        long_price: float,
    ) -> float:
        """
        Calculate unrealized P&L.

        P&L = Credit received - Current spread value

        Args:
            short_price: Current price of short option
            long_price: Current price of long option

        Returns:
            Unrealized P&L (positive = profit)
        """
        credit_received = self.net_credit * 100 * self.contracts
        current_value = self.calculate_current_value(short_price, long_price)
        return credit_received - current_value

    def close(
        self,
        exit_time: datetime,
        short_exit_price: float,
        long_exit_price: float,
        exit_reason: str,
    ) -> float:
        """
        Close the position and calculate P&L.

        Args:
            exit_time: When position was closed
            short_exit_price: Price paid to buy back short option
            long_exit_price: Price received for selling long option
            exit_reason: Reason for exit

        Returns:
            Realized P&L
        """
        self.exit_time = exit_time
        self.short_exit_price = short_exit_price
        self.long_exit_price = long_exit_price
        self.exit_reason = exit_reason
        self.net_debit = short_exit_price - long_exit_price

        # P&L = Credit received - Debit paid to close
        credit_received = self.net_credit * 100 * self.contracts
        debit_paid = self.net_debit * 100 * self.contracts
        self.pnl = credit_received - debit_paid

        return self.pnl

    def close_at_expiration(
        self,
        exit_time: datetime,
        settlement_price: float,
    ) -> float:
        """
        Close position at expiration based on settlement price.

        For 0DTE:
        - If OTM: Both legs expire worthless, keep full credit
        - If ITM: Spread settles at intrinsic value

        Args:
            exit_time: Expiration time
            settlement_price: Underlying price at settlement

        Returns:
            Realized P&L
        """
        if self.is_call_spread:
            # Call Credit Spread: Max loss if price > short_strike
            short_intrinsic = max(0, settlement_price - self.short_strike)
            long_intrinsic = max(0, settlement_price - self.long_strike)
        else:
            # Put Credit Spread: Max loss if price < short_strike
            short_intrinsic = max(0, self.short_strike - settlement_price)
            long_intrinsic = max(0, self.long_strike - settlement_price)

        return self.close(
            exit_time=exit_time,
            short_exit_price=short_intrinsic,
            long_exit_price=long_intrinsic,
            exit_reason="expiration",
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "underlying_price": self.underlying_price,
            "direction": self.direction,
            "short_strike": self.short_strike,
            "long_strike": self.long_strike,
            "expiration": self.expiration.isoformat() if self.expiration else None,
            "short_entry_price": self.short_entry_price,
            "long_entry_price": self.long_entry_price,
            "net_credit": self.net_credit,
            "spread_width": self.spread_width,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "contracts": self.contracts,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason,
            "pnl": self.pnl,
            "opening_move_pct": self.opening_move_pct,
        }


@dataclass
class MorningFadeSignal:
    """
    Signal from the morning fade strategy.

    Indicates direction of fade and entry parameters.
    """

    signal_type: str  # "CALL_SPREAD", "PUT_SPREAD", "NO_SIGNAL"
    timestamp: datetime
    underlying_price: float

    # Opening move information
    open_price: float
    direction_price: float  # Price at direction determination time
    opening_move_pct: float  # (direction_price - open) / open

    # Suggested trade parameters
    short_strike: Optional[float] = None
    long_strike: Optional[float] = None

    # State tracking
    state: str = ""  # Current state machine state
    reason: str = ""  # Human-readable reason

    def __repr__(self) -> str:
        return (
            f"MorningFadeSignal({self.signal_type}, {self.timestamp}, "
            f"move={self.opening_move_pct:.2%})"
        )
