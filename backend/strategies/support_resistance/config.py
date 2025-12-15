"""
Support/Resistance Bounce Strategy configuration.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class SupportResistanceConfig(StrategyConfig):
    """
    Support/Resistance Bounce strategy configuration.

    Trades bounces off key S/R levels.

    Key Parameters:
        sr_lookback_bars: Bars for S/R detection
        tolerance_pct: Price tolerance for S/R clustering
        min_level_strength: Minimum touches for valid S/R level
    """

    # === S/R Detection ===
    sr_lookback_bars: int = 100  # Bars to analyze for S/R
    tolerance_pct: float = 0.3  # 0.3% tolerance for level clustering
    min_level_strength: int = 2  # Minimum touches
    round_number_interval: float = 5.0  # For psychological levels

    # === Entry Rules ===
    entry_distance_pct: float = 0.15  # Enter within 0.15% of S/R level
    require_momentum_confirmation: bool = True  # Require price moving away from level
    momentum_confirm_bars: int = 2  # Bars to confirm bounce

    # === Entry Timing ===
    entry_start_minutes: int = 15  # Wait 15 min after open
    entry_cutoff_hour_utc: int = 18  # No entries after 1 PM ET

    # === Exit Settings ===
    profit_target_pct: float = 0.20  # 20% profit target
    stop_loss_pct: float = 0.15  # 15% stop loss

    # === Trade Management ===
    max_trades_per_day: int = 2
