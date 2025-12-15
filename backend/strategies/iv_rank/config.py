"""
IV Rank Strategy configuration.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class IVRankConfig(StrategyConfig):
    """
    IV Rank Filter strategy configuration.

    Filters RSI mean reversion signals by IV Rank threshold.
    Higher IV = higher option premiums = better mean reversion opportunities.

    Key Parameters:
        min_iv_rank: Minimum IV rank to allow trade entry (0-100)
        iv_rank_lookback_days: Days for IV Rank calculation
        iv_rank_require_valid: If True, block entries when IV data unavailable
    """

    # === IV Rank Filter Settings ===
    min_iv_rank: float = 50.0  # Minimum IV rank to trade (0-100)
    max_iv_rank: float = 100.0  # Maximum IV rank (cap for extreme volatility)
    iv_rank_lookback_days: int = 45  # Trading days for IV Rank calculation
    iv_rank_min_history: int = 20  # Minimum history days required
    iv_rank_require_valid: bool = False  # Block if IV unavailable

    # === Position Sizing Based on IV ===
    scale_size_with_iv: bool = False  # Higher IV = larger position
    iv_scale_factor: float = 0.5  # Scaling factor (0.5 = 50% more at max IV)
    iv_scale_threshold: float = 70.0  # Only scale above this IV rank

    # === RSI Settings (inherited base strategy) ===
    # Uses RSI mean reversion as the underlying signal generator
    rsi_period: int = 14
    rsi_sma_period: int = 10
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # === Exit Settings ===
    profit_target_pct: float = 0.20  # 20% profit target
    stop_loss_pct: float = 0.25  # 25% stop loss

    def get_contract_multiplier_for_iv_rank(self, iv_rank: float) -> int:
        """
        Calculate contract multiplier based on IV rank.

        Higher IV rank = more contracts (if scale_size_with_iv enabled).

        Args:
            iv_rank: Current IV rank (0-100)

        Returns:
            Contract multiplier (1 = base, higher = more contracts)
        """
        if not self.scale_size_with_iv:
            return self.contract_multiplier

        if iv_rank < self.iv_scale_threshold:
            return self.contract_multiplier

        # Linear scaling from threshold to 100
        scale_range = 100.0 - self.iv_scale_threshold
        iv_above_threshold = iv_rank - self.iv_scale_threshold
        scale = 1.0 + (iv_above_threshold / scale_range) * self.iv_scale_factor

        return int(self.contract_multiplier * scale)
