"""
VWAP Reversion Strategy configuration.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class VWAPReversionConfig(StrategyConfig):
    """
    VWAP Reversion strategy configuration.

    Trades mean reversion to VWAP when price deviates significantly.

    Key Parameters:
        vwap_deviation_pct: Minimum % deviation from VWAP to trigger entry
        vwap_bands_std: Standard deviation multiplier for VWAP bands
        require_volume_confirmation: Require declining volume on deviation
    """

    # === VWAP Settings ===
    vwap_deviation_pct: float = 0.3  # 0.3% deviation from VWAP to trigger
    vwap_bands_std: float = 2.0  # Standard deviation for VWAP bands
    use_vwap_bands: bool = True  # Use statistical bands vs simple %

    # === Volume Confirmation ===
    require_volume_confirmation: bool = True  # Volume declining on deviation
    volume_sma_period: int = 20  # Periods for volume SMA
    volume_threshold: float = 0.8  # Volume must be below this % of SMA

    # === Entry Timing ===
    entry_start_minutes: int = 30  # Wait N minutes after open
    entry_cutoff_hour_utc: int = 19  # 2 PM ET = no new entries after

    # === Exit Settings ===
    profit_target_pct: float = 0.15  # 15% profit target (reversion is quick)
    stop_loss_pct: float = 0.20  # 20% stop loss
    exit_at_vwap: bool = True  # Exit when price returns to VWAP

    # === Position Sizing ===
    scale_with_deviation: bool = False  # Larger deviation = larger position
    max_deviation_pct: float = 1.0  # Maximum deviation to trade

    def get_deviation_multiplier(self, deviation_pct: float) -> float:
        """
        Calculate position size multiplier based on deviation.

        Larger deviations = stronger mean reversion potential.

        Args:
            deviation_pct: Current deviation from VWAP (absolute)

        Returns:
            Multiplier for base contracts (1.0 to 2.0)
        """
        if not self.scale_with_deviation:
            return 1.0

        if deviation_pct <= self.vwap_deviation_pct:
            return 1.0

        if deviation_pct >= self.max_deviation_pct:
            return 2.0

        # Linear scaling
        range_span = self.max_deviation_pct - self.vwap_deviation_pct
        position = (deviation_pct - self.vwap_deviation_pct) / range_span
        return 1.0 + position
