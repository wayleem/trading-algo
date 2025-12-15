"""
Gap Fade Strategy configuration.

Fades overnight gaps expecting mean reversion during the trading day.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class GapFadeConfig(StrategyConfig):
    """
    Gap Fade strategy configuration.

    Thesis: Overnight gaps tend to fill (partially or fully) during the trading day.
    We fade the gap direction:
    - Gap UP -> BUY PUT (expect price to fall)
    - Gap DOWN -> BUY CALL (expect price to rise)

    Key Parameters:
        gap_threshold_pct: Minimum gap size to trade (filters noise)
        max_gap_pct: Maximum gap size (large gaps often trend)
        entry_delay_minutes: Wait after open for volatility to settle
        entry_window_minutes: Window during which entry is allowed
    """

    # === Gap Detection Settings ===
    gap_threshold_pct: float = 0.3  # Minimum gap to trade (0.3% = ~$1.50 on SPY)
    max_gap_pct: float = 1.5  # Maximum gap (large gaps less likely to fill)

    # === Direction Control ===
    fade_gap_up: bool = True  # Trade downside fades (gap up -> buy put)
    fade_gap_down: bool = True  # Trade upside fades (gap down -> buy call)

    # === Entry Timing ===
    # Wait 5-10 min after open for volatility to settle
    entry_delay_minutes: int = 5  # Wait 5 min after open before entry
    entry_window_minutes: int = 15  # Entry window: 9:35-9:50 AM ET
    entry_cutoff_hour_utc: int = 15  # No new entries after 10:30 AM ET (15 UTC)

    # === Position Management ===
    max_entries_per_day: int = 1  # One gap fade per day (gaps only happen at open)

    # === Exit Parameters ===
    profit_target_pct: float = 0.30  # 30% profit target
    stop_loss_pct: float = 0.10  # 10% stop (tight, per ORB optimization)

    # === Strike Selection ===
    strike_offset: float = 0.5  # OTM distance

    def is_gap_tradeable(self, gap_pct: float) -> bool:
        """
        Check if gap size is within tradeable bounds.

        Args:
            gap_pct: Gap percentage (positive for gap up, negative for gap down)

        Returns:
            True if gap is tradeable
        """
        abs_gap = abs(gap_pct)
        return self.gap_threshold_pct <= abs_gap <= self.max_gap_pct

    def should_fade_direction(self, gap_pct: float) -> bool:
        """
        Check if we should fade this gap direction.

        Args:
            gap_pct: Gap percentage (positive for gap up, negative for gap down)

        Returns:
            True if we should trade this direction
        """
        if gap_pct > 0:
            return self.fade_gap_up
        else:
            return self.fade_gap_down
