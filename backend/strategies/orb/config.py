"""
ORB (Opening Range Breakout) Strategy configuration.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class ORBConfig(StrategyConfig):
    """
    Opening Range Breakout strategy configuration.

    Tracks high/low from market open, then trades breakouts in either direction.

    Key Parameters:
        range_minutes: Duration of opening range (30 or 60 minutes typical)
        breakout_buffer: Extra buffer beyond range to confirm breakout
        require_close: If True, bar must CLOSE above/below range
        max_entries_per_day: Maximum number of entries allowed per day
    """

    # === Opening Range Settings ===
    range_minutes: int = 60  # 60 min = 9:30-10:30 AM ET
    breakout_buffer: float = 0.0  # Extra $ buffer beyond range
    require_close: bool = True  # Require bar CLOSE above/below range

    # === Entry Settings ===
    max_entries_per_day: int = 3  # Max entries if stopped out and re-breaks
    entry_cutoff_hour_utc: int = 18  # 1 PM ET = no new entries after

    # === Exit Settings ===
    profit_target_pct: float = 0.30  # 30% profit target (breakouts run)
    stop_loss_pct: float = 0.20  # 20% stop loss

    # === Position Sizing ===
    scale_with_range_size: bool = False  # Smaller range = larger position
    min_range_size: float = 1.0  # Minimum range size in $ to trade
    max_range_size: float = 5.0  # Maximum range size in $ to trade

    def is_range_tradeable(self, range_size: float) -> bool:
        """
        Check if opening range is within tradeable bounds.

        Args:
            range_size: Size of opening range in dollars

        Returns:
            True if range is tradeable
        """
        return self.min_range_size <= range_size <= self.max_range_size

    def get_contracts_for_range(self, range_size: float) -> int:
        """
        Calculate contracts based on range size (if scaling enabled).

        Smaller ranges = potentially larger moves = more contracts.

        Args:
            range_size: Size of opening range in dollars

        Returns:
            Number of contracts to trade
        """
        if not self.scale_with_range_size:
            return self.base_contracts

        # Inverse scaling: smaller range = more contracts
        if range_size <= self.min_range_size:
            return self.max_contracts

        if range_size >= self.max_range_size:
            return self.base_contracts

        # Linear interpolation
        range_span = self.max_range_size - self.min_range_size
        position = (self.max_range_size - range_size) / range_span
        contracts = self.base_contracts + int(
            (self.max_contracts - self.base_contracts) * position
        )
        return contracts
