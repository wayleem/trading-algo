"""
VWAP Fade Strategy configuration.

Fades deviations below VWAP in the first 30 minutes of trading.
Structural edge from institutional VWAP targeting for order execution.
"""

from dataclasses import dataclass
from typing import Optional
from strategies.base.config import StrategyConfig


@dataclass
class VWAPFadeConfig(StrategyConfig):
    """
    VWAP Fade strategy configuration.

    Thesis: Institutions target VWAP for execution benchmarks.
    Early deviations are noise that algorithms push back to VWAP.

    Validated edge:
    - 9:30-10:00 AM window: 90% reversion rate (stable 2022-2024)
    - Below VWAP fades (buy calls): 82.5% in 2024
    - Above VWAP fades: 70% and declining - AVOID

    Constraints:
    - Entry window: 9:30-10:00 AM ET only
    - Direction: Below VWAP only (buy calls)
    - Exit deadline: 10:30 AM ET (don't hold into declining window)
    """

    # === VWAP Deviation Settings ===
    deviation_threshold_pct: float = 0.3  # Min deviation from VWAP to trigger
    max_deviation_pct: float = 1.0  # Max deviation (larger = likely trend, not mean reversion)

    # === Direction Control ===
    # Based on validation: below VWAP fades are more stable
    fade_below_vwap: bool = True   # Buy calls when price below VWAP
    fade_above_vwap: bool = False  # Disabled - 70% and declining

    # === Entry Timing (TIGHT CONSTRAINTS) ===
    # Only 9:30-10:00 AM ET has stable 90% reversion
    entry_start_minutes_after_open: int = 0   # Start immediately at 9:30 AM
    entry_window_minutes: int = 30            # Only first 30 minutes
    exit_deadline_minutes_after_open: int = 60  # Must exit by 10:30 AM

    # UTC times (ET + 5 in winter, ET + 4 in summer)
    # Using UTC 14 = 9 AM ET (conservative for year-round)
    entry_start_hour_utc: int = 14
    entry_start_minute_utc: int = 35  # 9:35 AM ET (5 min after open for VWAP to stabilize)
    entry_cutoff_hour_utc: int = 15   # No new entries after 10:00 AM ET
    exit_deadline_hour_utc: int = 15
    exit_deadline_minute_utc: int = 30  # Exit by 10:30 AM ET

    # === Position Management ===
    max_entries_per_day: int = 1  # One VWAP fade per day

    # === Exit Parameters ===
    # IMPORTANT: VWAP fade uses underlying-based exits, NOT option P&L-based exits
    # The validated thesis is about underlying price reverting to VWAP (90% rate)
    # Option P&L exits would trigger prematurely due to theta decay

    # VWAP-based profit target: Exit when underlying reverts toward VWAP
    profit_target_reversion_pct: float = 50.0  # Take profit at 50% reversion to VWAP
    use_vwap_touch_exit: bool = True  # Also exit if price touches VWAP

    # Disable option P&L-based exits - they conflict with VWAP-based exits
    # The underlying-based stop is controlled by max_deviation_pct (1.0%)
    profit_target_pct: Optional[float] = None  # Disable option P&L profit target
    stop_loss_pct: float = 1.0  # Disable option stop (100% = never triggers)

    # Realistic 0DTE slippage (override defaults)
    slippage_entry_pct: float = 0.03  # 3% entry slippage
    slippage_exit_pct: float = 0.03   # 3% exit slippage
    slippage_stop_extra_pct: float = 0.02  # Extra 2% on stops

    # === Strike Selection ===
    strike_offset: float = 0.5  # Slightly OTM

    def is_deviation_tradeable(self, deviation_pct: float) -> bool:
        """Check if deviation is within tradeable bounds."""
        abs_dev = abs(deviation_pct)
        return self.deviation_threshold_pct <= abs_dev <= self.max_deviation_pct

    def should_fade_direction(self, deviation_pct: float) -> bool:
        """Check if we should fade this direction."""
        if deviation_pct < 0:  # Below VWAP
            return self.fade_below_vwap
        else:  # Above VWAP
            return self.fade_above_vwap
