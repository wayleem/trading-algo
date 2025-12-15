"""
First Pullback Strategy configuration.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class FirstPullbackConfig(StrategyConfig):
    """
    First Pullback strategy configuration.

    Waits for initial trend, then enters on first pullback.

    Key Parameters:
        initial_move_pct: Minimum % move to establish trend
        pullback_pct: Minimum pullback from high/low to trigger
        max_pullback_pct: Maximum pullback (too deep = trend broken)
    """

    # === Trend Establishment ===
    initial_move_lookback: int = 30  # Bars to establish initial move
    initial_move_pct: float = 0.3  # 0.3% move to establish trend
    trend_ema_period: int = 20  # EMA to confirm trend

    # === Pullback Settings ===
    pullback_pct: float = 0.10  # 0.10% pullback from extreme to trigger
    max_pullback_pct: float = 0.25  # Max pullback before trend invalid
    pullback_bars_min: int = 3  # Minimum bars for valid pullback
    pullback_bars_max: int = 15  # Maximum bars for pullback to complete

    # === Entry Confirmation ===
    require_higher_low: bool = True  # For bullish: pullback must make higher low
    require_momentum_reversal: bool = True  # Momentum turning back to trend

    # === Entry Timing ===
    entry_start_minutes: int = 30  # Wait for trend to establish
    entry_cutoff_hour_utc: int = 18  # 1 PM ET = no new entries

    # === Exit Settings ===
    profit_target_pct: float = 0.20  # 20% profit target
    stop_loss_pct: float = 0.15  # 15% stop loss
    use_swing_stop: bool = True  # Stop below pullback low/high

    # === Position Sizing ===
    scale_with_trend_strength: bool = False  # Stronger trend = larger position

    def is_valid_pullback(self, pullback_pct: float) -> bool:
        """
        Check if pullback is within valid range.

        Args:
            pullback_pct: Current pullback percentage (absolute)

        Returns:
            True if pullback is valid for entry
        """
        return self.pullback_pct <= pullback_pct <= self.max_pullback_pct
