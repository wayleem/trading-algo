"""
Momentum Scalping Strategy configuration.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class MomentumScalpConfig(StrategyConfig):
    """
    Momentum Scalping strategy configuration.

    Trades strong momentum moves with tight profit targets.

    Key Parameters:
        momentum_threshold: Minimum % move in lookback period to trigger
        volume_surge_threshold: Minimum volume vs SMA to confirm momentum
        ema_period: EMA period for trend confirmation
    """

    # === Momentum Settings ===
    momentum_lookback: int = 5  # Bars to measure momentum
    momentum_threshold: float = 0.15  # 0.15% move in lookback period
    use_acceleration: bool = True  # Require accelerating momentum

    # === Volume Confirmation ===
    require_volume_surge: bool = True  # Volume must be elevated
    volume_sma_period: int = 20  # Periods for volume SMA
    volume_surge_threshold: float = 1.5  # Volume must be 1.5x SMA

    # === Trend Confirmation ===
    use_ema_filter: bool = True  # Only trade in EMA direction
    ema_period: int = 20  # EMA period for trend
    require_price_above_ema: bool = True  # For calls, price must be above EMA

    # === Entry Timing ===
    entry_start_minutes: int = 5  # Wait N minutes after open
    entry_cutoff_hour_utc: int = 19  # 2 PM ET = no new entries

    # === Exit Settings ===
    profit_target_pct: float = 0.10  # 10% profit target (scalping)
    stop_loss_pct: float = 0.08  # 8% stop loss (tight)
    trailing_stop_pct: float = 0.05  # 5% trailing stop after breakeven

    # === Position Sizing ===
    scale_with_momentum: bool = False  # Stronger momentum = larger position
    max_momentum_threshold: float = 0.5  # Cap momentum at this level

    def get_momentum_multiplier(self, momentum_pct: float) -> float:
        """
        Calculate position size multiplier based on momentum strength.

        Args:
            momentum_pct: Current momentum percentage (absolute)

        Returns:
            Multiplier for base contracts (1.0 to 2.0)
        """
        if not self.scale_with_momentum:
            return 1.0

        if momentum_pct <= self.momentum_threshold:
            return 1.0

        if momentum_pct >= self.max_momentum_threshold:
            return 2.0

        # Linear scaling
        range_span = self.max_momentum_threshold - self.momentum_threshold
        position = (momentum_pct - self.momentum_threshold) / range_span
        return 1.0 + position
