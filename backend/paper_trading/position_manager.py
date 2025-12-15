"""
Position Manager for Paper Trading.

Handles position state, P&L tracking, and risk controls.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class PositionState(Enum):
    """State of a trading position."""
    PENDING_ENTRY = "pending_entry"
    OPEN = "open"
    PENDING_EXIT = "pending_exit"
    CLOSED = "closed"


class ExitReason(Enum):
    """Reason for position exit."""
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    FORCE_EXIT = "force_exit"  # 3:50 PM deadline
    MANUAL = "manual"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class Position:
    """
    Represents an open or closed trading position.
    """
    # Identification
    position_id: str
    option_symbol: str
    underlying_symbol: str
    option_type: str  # "call" or "put"
    strike: float
    expiration: date

    # Entry details
    entry_time: datetime
    entry_price: float
    quantity: int
    entry_signal_reason: str

    # Current state
    state: PositionState = PositionState.PENDING_ENTRY
    current_price: float = 0.0
    last_update: Optional[datetime] = None

    # Exit details (filled when closed)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None

    # Order tracking
    entry_order_id: Optional[str] = None
    profit_target_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None

    # Range info for logging
    range_high: float = 0.0
    range_low: float = 0.0

    # Settlement tracking (for EOD expiration)
    settlement_underlying: Optional[float] = None  # SPY price at exit

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L (per contract)."""
        if self.state != PositionState.OPEN:
            return 0.0
        return (self.current_price - self.entry_price) * 100  # Options are 100 shares

    @property
    def unrealized_pnl_total(self) -> float:
        """Calculate total unrealized P&L."""
        return self.unrealized_pnl * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L as percentage."""
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100

    @property
    def realized_pnl(self) -> float:
        """Calculate realized P&L (per contract) if closed."""
        if self.state != PositionState.CLOSED or self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * 100

    @property
    def realized_pnl_total(self) -> float:
        """Calculate total realized P&L."""
        return self.realized_pnl * self.quantity

    @property
    def realized_pnl_pct(self) -> float:
        """Calculate realized P&L as percentage."""
        if self.entry_price <= 0 or self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price * 100

    @property
    def profit_target_price(self) -> float:
        """Calculate profit target exit price (set externally)."""
        # This should be calculated by the caller based on config
        return 0.0

    @property
    def stop_loss_price(self) -> float:
        """Calculate stop loss exit price (set externally)."""
        # This should be calculated by the caller based on config
        return 0.0


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    trades_executed: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    high_water_mark: float = 0.0
    circuit_breaker_triggered: bool = False

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.trades_executed == 0:
            return 0.0
        return self.winning_trades / self.trades_executed * 100


class PositionManager:
    """
    Manages trading positions and risk controls.

    Responsibilities:
    - Track open and closed positions
    - Monitor P&L and risk limits
    - Enforce circuit breakers
    - Provide position state for order management
    """

    def __init__(
        self,
        max_positions: int = 1,
        daily_loss_limit: float = 200.0,
    ):
        """
        Initialize position manager.

        Args:
            max_positions: Maximum concurrent positions allowed
            daily_loss_limit: Maximum daily loss before circuit breaker
        """
        self.max_positions = max_positions
        self.daily_loss_limit = daily_loss_limit

        # Position tracking
        self.open_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.all_positions: List[Position] = []

        # Daily stats
        self.current_date: Optional[date] = None
        self.daily_stats: Optional[DailyStats] = None

        # Risk state
        self.circuit_breaker_active = False

    def reset_for_new_day(self, trading_date: date) -> None:
        """Reset state for a new trading day."""
        self.current_date = trading_date
        self.daily_stats = DailyStats(date=trading_date)
        self.open_positions = []
        self.closed_positions = []
        self.circuit_breaker_active = False
        logger.info(f"Position manager reset for {trading_date}")

    def can_open_position(self) -> tuple[bool, str]:
        """
        Check if a new position can be opened.

        Returns:
            Tuple of (can_open, reason)
        """
        # Check circuit breaker
        if self.circuit_breaker_active:
            return False, "Circuit breaker active - daily loss limit reached"

        # Check max positions
        if len(self.open_positions) >= self.max_positions:
            return False, f"Max positions ({self.max_positions}) already open"

        return True, "OK"

    def create_position(
        self,
        position_id: str,
        option_symbol: str,
        underlying_symbol: str,
        option_type: str,
        strike: float,
        expiration: date,
        entry_price: float,
        quantity: int,
        signal_reason: str,
        range_high: float = 0.0,
        range_low: float = 0.0,
    ) -> Position:
        """
        Create a new position (pending entry).

        Args:
            position_id: Unique position identifier
            option_symbol: Option contract symbol
            underlying_symbol: Underlying symbol (e.g., SPY)
            option_type: "call" or "put"
            strike: Option strike price
            expiration: Option expiration date
            entry_price: Expected entry price
            quantity: Number of contracts
            signal_reason: Reason for entry signal
            range_high: ORB range high
            range_low: ORB range low

        Returns:
            New Position object
        """
        position = Position(
            position_id=position_id,
            option_symbol=option_symbol,
            underlying_symbol=underlying_symbol,
            option_type=option_type,
            strike=strike,
            expiration=expiration,
            entry_time=datetime.now(),
            entry_price=entry_price,
            quantity=quantity,
            entry_signal_reason=signal_reason,
            state=PositionState.PENDING_ENTRY,
            range_high=range_high,
            range_low=range_low,
        )

        self.all_positions.append(position)
        logger.info(f"Created position {position_id}: {option_symbol} x{quantity} @ ${entry_price:.2f}")

        return position

    def confirm_entry(self, position: Position, fill_price: float, order_id: str) -> None:
        """
        Confirm position entry after order fill.

        Args:
            position: Position to confirm
            fill_price: Actual fill price
            order_id: Entry order ID
        """
        position.state = PositionState.OPEN
        position.entry_price = fill_price
        position.current_price = fill_price
        position.entry_order_id = order_id
        position.last_update = datetime.now()

        self.open_positions.append(position)

        logger.info(
            f"Position {position.position_id} OPENED: "
            f"{position.option_symbol} x{position.quantity} @ ${fill_price:.2f}"
        )

    def update_position_price(self, position: Position, current_price: float) -> None:
        """
        Update position with current market price.

        Args:
            position: Position to update
            current_price: Current option price
        """
        position.current_price = current_price
        position.last_update = datetime.now()

        # Update daily stats
        if self.daily_stats:
            total_unrealized = sum(p.unrealized_pnl_total for p in self.open_positions)
            total_realized = self.daily_stats.total_pnl

            current_total = total_realized + total_unrealized

            # Update high water mark
            if current_total > self.daily_stats.high_water_mark:
                self.daily_stats.high_water_mark = current_total

            # Update max drawdown
            drawdown = self.daily_stats.high_water_mark - current_total
            if drawdown > self.daily_stats.max_drawdown:
                self.daily_stats.max_drawdown = drawdown

    def close_position(
        self,
        position: Position,
        exit_price: float,
        exit_reason: ExitReason,
    ) -> None:
        """
        Close a position.

        Args:
            position: Position to close
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        position.state = PositionState.CLOSED
        position.exit_time = datetime.now()
        position.exit_price = exit_price
        position.exit_reason = exit_reason

        # Move from open to closed
        if position in self.open_positions:
            self.open_positions.remove(position)
        self.closed_positions.append(position)

        # Update daily stats
        if self.daily_stats:
            self.daily_stats.trades_executed += 1
            self.daily_stats.total_pnl += position.realized_pnl_total

            if position.realized_pnl > 0:
                self.daily_stats.winning_trades += 1
            else:
                self.daily_stats.losing_trades += 1

            # Check circuit breaker
            if self.daily_stats.total_pnl <= -self.daily_loss_limit:
                self._trigger_circuit_breaker()

        logger.info(
            f"Position {position.position_id} CLOSED: "
            f"{exit_reason.value} @ ${exit_price:.2f} | "
            f"P&L: ${position.realized_pnl_total:.2f} ({position.realized_pnl_pct:+.1f}%)"
        )

    def _trigger_circuit_breaker(self) -> None:
        """Trigger circuit breaker due to daily loss limit."""
        self.circuit_breaker_active = True
        if self.daily_stats:
            self.daily_stats.circuit_breaker_triggered = True

        logger.warning(
            f"CIRCUIT BREAKER TRIGGERED: Daily loss ${self.daily_stats.total_pnl:.2f} "
            f"exceeds limit ${self.daily_loss_limit:.2f}"
        )

    def get_open_position(self) -> Optional[Position]:
        """Get the current open position (assumes max 1)."""
        return self.open_positions[0] if self.open_positions else None

    def calculate_settlement_value(
        self,
        position: Position,
        underlying_price: float,
    ) -> float:
        """
        Calculate option settlement value at expiration.

        For 0DTE options:
        - Calls: max(0, underlying - strike) per share
        - Puts: max(0, strike - underlying) per share
        - OTM options = $0 (expire worthless)

        Args:
            position: The position to calculate settlement for
            underlying_price: Current/settlement price of underlying (SPY)

        Returns:
            Settlement value per contract (intrinsic value)
        """
        if position.option_type == "call":
            intrinsic = underlying_price - position.strike
        else:  # put
            intrinsic = position.strike - underlying_price

        return max(0.0, intrinsic)

    def get_daily_pnl(self) -> float:
        """Get current daily P&L (realized + unrealized)."""
        realized = self.daily_stats.total_pnl if self.daily_stats else 0.0
        unrealized = sum(p.unrealized_pnl_total for p in self.open_positions)
        return realized + unrealized

    def get_summary(self) -> dict:
        """Get position manager summary."""
        return {
            "date": str(self.current_date),
            "open_positions": len(self.open_positions),
            "closed_positions": len(self.closed_positions),
            "daily_pnl": self.get_daily_pnl(),
            "realized_pnl": self.daily_stats.total_pnl if self.daily_stats else 0.0,
            "unrealized_pnl": sum(p.unrealized_pnl_total for p in self.open_positions),
            "trades_executed": self.daily_stats.trades_executed if self.daily_stats else 0,
            "win_rate": self.daily_stats.win_rate if self.daily_stats else 0.0,
            "circuit_breaker": self.circuit_breaker_active,
        }
