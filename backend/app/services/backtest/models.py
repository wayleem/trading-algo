"""Trade models for backtest simulation."""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List, Tuple

from app.models.schemas import SignalType


@dataclass
class SimulatedTrade:
    """
    Represents a trade during backtest simulation.

    Tracks all aspects of a trade including:
    - Entry/exit details
    - Position sizing (contracts, averaging down)
    - P&L calculation
    - Option specifics (strike, expiration)
    """

    # Required entry fields
    entry_time: datetime
    entry_price: float
    signal_type: SignalType
    underlying_price: float
    profit_target: float
    stop_loss: float

    # Entry context
    entry_bar_index: int = 0
    entry_rsi: float = 0.0
    strike: float = 0.0
    expiration: Optional[date] = None

    # Price tracking for interpolation
    intraday_prices: List = field(default_factory=list)
    rsi_history: List[Tuple[datetime, float, float]] = field(default_factory=list)

    # Exit fields
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None

    # Contract scaling
    initial_contracts: int = 1
    remaining_contracts: int = 1
    partial_tp_taken: bool = False
    partial_tp_pnl: float = 0.0

    # Averaging down
    total_contracts: int = 1
    avg_entry_price: float = 0.0
    add_on_entries: List[Tuple[float, int, datetime]] = field(default_factory=list)

    # Pattern tracking for position sizing
    pattern_strength: float = 0.0
    pattern_name: Optional[str] = None

    # Kelly Criterion tracking
    kelly_multiplier: float = 1.0

    # Extended target for partial exits
    extended_target: Optional[float] = None

    # Parallel mode tracking
    timeframe: str = ""

    # Settlement tracking (for EOD expiration)
    settlement_underlying: float = 0.0

    # VWAP tracking (for VWAP fade strategy)
    entry_vwap: Optional[float] = None

    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_time is None

    @property
    def is_call(self) -> bool:
        """Check if this is a call trade."""
        return self.signal_type == SignalType.BUY_CALL

    @property
    def is_put(self) -> bool:
        """Check if this is a put trade."""
        return self.signal_type == SignalType.BUY_PUT

    def get_effective_entry_price(self) -> float:
        """Get the effective entry price (avg if averaged down, otherwise entry)."""
        return self.avg_entry_price if self.avg_entry_price > 0 else self.entry_price
