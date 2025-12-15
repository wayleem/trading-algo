"""
Delta Targeting Strategy configuration.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class DeltaTargetingConfig(StrategyConfig):
    """
    Delta Targeting strategy configuration.

    Selects options based on target delta for consistent exposure.

    Key Parameters:
        target_delta: Desired delta for option selection (0.0-1.0)
        delta_tolerance: Acceptable deviation from target delta
    """

    # === Delta Selection ===
    target_delta: float = 0.40  # Target ~40 delta options
    delta_tolerance: float = 0.10  # Accept 30-50 delta

    # === Entry Conditions ===
    momentum_threshold: float = 0.15  # % move to trigger entry
    momentum_lookback: int = 10  # Bars to measure momentum
    entry_start_minutes: int = 15  # Wait after open
    entry_cutoff_hour_utc: int = 18  # No entries after 1 PM ET

    # === Exit Settings ===
    profit_target_pct: float = 0.20  # 20% profit target
    stop_loss_pct: float = 0.15  # 15% stop loss

    # === Trade Management ===
    max_trades_per_day: int = 2
