"""
Morning Fade Strategy configuration.
"""

from dataclasses import dataclass
from datetime import time
from strategies.base.config import StrategyConfig


@dataclass
class MorningFadeConfig(StrategyConfig):
    """
    Morning Fade strategy configuration.

    Trades the morning fade pattern:
    - Collects 9:30-10:00 AM prices to determine opening direction
    - Entry window: 10:00-10:30 AM ET
    - Fades the opening direction with a credit spread
    - Exit by: 2:00 PM ET (or on PT/SL)

    Key Parameters:
        fade_threshold_pct: Minimum opening move to trigger fade (e.g., 0.3%)
        spread_width: Width of credit spread in dollars
        short_strike_offset: How far OTM to place short strike
    """

    # === Time Settings (ET) ===
    # Opening range collection: 9:30-10:00 AM
    range_end_hour_et: int = 10
    range_end_minute_et: int = 0

    # Entry window: 10:00-10:30 AM ET
    entry_start_time_et: time = None  # Set in __post_init__
    entry_end_time_et: time = None  # Set in __post_init__

    # Exit by 2:00 PM ET
    exit_by_time_et: time = None  # Set in __post_init__
    exit_buffer_minutes: int = 30  # Stop entering this many min before exit

    # === Fade Logic ===
    fade_threshold_pct: float = 0.3  # Min opening move to trigger (0.3% = ~$1.80 on SPY)
    fade_strong_threshold_pct: float = 0.5  # Strong move for larger position
    require_gap: bool = False  # Require gap open (previous close to open)
    gap_threshold_pct: float = 0.3  # Min gap size if require_gap=True

    # === Credit Spread Settings ===
    spread_width: float = 3.0  # $3 wide spread
    short_strike_offset: float = 2.0  # Short leg $2 OTM from current price
    use_delta_targeting: bool = False  # Use delta instead of fixed offset
    short_strike_delta: float = 0.30  # Target 30 delta for short strike

    # === Exit Settings ===
    profit_target_pct: float = 0.50  # Close at 50% of max profit
    stop_loss_pct: float = 1.50  # Stop at 150% of credit received
    time_stop_enabled: bool = True  # Exit at exit_by_time

    # === Position Sizing ===
    base_contracts: int = 1
    strong_move_contracts: int = 2  # Extra contracts for strong moves

    # === RSI Confirmation (optional) ===
    use_rsi_confirmation: bool = False  # Require RSI confirmation
    rsi_period: int = 14
    rsi_oversold: float = 30.0  # Confirm call spread fade
    rsi_overbought: float = 70.0  # Confirm put spread fade

    def __post_init__(self):
        """Initialize time objects."""
        if self.entry_start_time_et is None:
            self.entry_start_time_et = time(10, 0)
        if self.entry_end_time_et is None:
            self.entry_end_time_et = time(10, 30)
        if self.exit_by_time_et is None:
            self.exit_by_time_et = time(14, 0)

    def get_entry_cutoff_utc(self) -> int:
        """Convert exit time to UTC hour for cutoff."""
        # ET to UTC (EST = +5)
        return self.exit_by_time_et.hour + 5

    def get_contracts_for_move(self, move_pct: float) -> int:
        """
        Determine contracts based on opening move size.

        Args:
            move_pct: Absolute opening move percentage

        Returns:
            Number of contracts
        """
        if abs(move_pct) >= self.fade_strong_threshold_pct:
            return self.base_contracts + self.strong_move_contracts
        return self.base_contracts

    def to_dict(self):
        """Override to handle time serialization."""
        result = super().to_dict()
        # Convert time objects to strings
        if self.exit_by_time_et:
            result["exit_by_time_et"] = self.exit_by_time_et.isoformat()
        return result
