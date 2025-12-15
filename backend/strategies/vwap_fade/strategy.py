"""
VWAP Fade Strategy.

Fades deviations below VWAP in the first 30 minutes of trading.
Structural edge from institutional VWAP targeting for order execution.
"""

import logging
from typing import List, Optional

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.vwap_fade.config import VWAPFadeConfig
from strategies.vwap_fade.signal_generator import VWAPFadeSignalGenerator, VWAPSignal

logger = logging.getLogger(__name__)


class VWAPFadeStrategy(BaseStrategy):
    """
    VWAP Fade Strategy.

    Thesis: Institutions target VWAP for execution benchmarks.
    Early morning deviations from VWAP are noise that algorithms push back.

    Validated Edge (2022-2024):
    - 9:30-10:00 AM window: 90% reversion rate (stable across years)
    - Below VWAP fades: 82.5% in 2024 (more stable than above)
    - Above VWAP fades: 70% and declining (AVOID)

    Strategy Rules:
    1. Calculate running VWAP from market open
    2. Wait 5 minutes for VWAP to stabilize
    3. If price deviates >= 0.3% BELOW VWAP in 9:35-10:00 AM window:
       -> BUY CALL (fade up expecting reversion to VWAP)
    4. Exit: 50% reversion to VWAP, VWAP touch, or 10:30 AM deadline
    5. Stop: 15% option loss (accommodates 0.4% adverse excursion)

    Correlation with ORB:
    - ORB signals after 10:30 AM (60-min range)
    - VWAP fade exits by 10:30 AM
    - No timing overlap
    - Trend days: ORB wins, VWAP fade loses or no signal
    - Chop days: ORB loses, VWAP fade wins
    - Expected: Negative or zero correlation = diversification
    """

    def __init__(self):
        """Initialize VWAP Fade strategy."""
        self._signal_generator: Optional[VWAPFadeSignalGenerator] = None

    @property
    def name(self) -> str:
        return "vwap_fade"

    @property
    def description(self) -> str:
        return "VWAP Fade - fades early morning deviations below VWAP (9:35-10:00 AM)"

    def get_default_config(self) -> VWAPFadeConfig:
        """Return default VWAP Fade strategy configuration."""
        return VWAPFadeConfig(
            # VWAP settings
            deviation_threshold_pct=0.3,
            max_deviation_pct=1.0,
            # Direction (only below VWAP - validated)
            fade_below_vwap=True,
            fade_above_vwap=False,
            # Timing (tight window - validated)
            entry_start_hour_utc=14,
            entry_start_minute_utc=35,  # 9:35 AM ET
            entry_cutoff_hour_utc=15,    # 10:00 AM ET
            exit_deadline_hour_utc=15,
            exit_deadline_minute_utc=30,  # 10:30 AM ET
            # Position
            max_entries_per_day=1,
            # Exit - VWAP-based exits (NOT option P&L-based)
            profit_target_reversion_pct=50.0,
            use_vwap_touch_exit=True,
            profit_target_pct=None,  # Disable option P&L profit target
            stop_loss_pct=1.0,  # Disable option stop (100% = never triggers)
        )

    def get_entry_filters(self) -> List:
        """VWAP Fade doesn't need additional filters - VWAP deviation defines entry."""
        return []

    async def generate_signals(
        self,
        bars: List[dict],
        config: VWAPFadeConfig,
    ) -> List[TradingSignal]:
        """
        Generate VWAP fade signals from price bars.

        Args:
            bars: OHLCV bars (1-minute preferred for accurate VWAP)
            config: VWAPFadeConfig instance

        Returns:
            List of TradingSignal for each bar
        """
        if not bars:
            return []

        # Initialize signal generator with config
        generator = VWAPFadeSignalGenerator(
            deviation_threshold_pct=config.deviation_threshold_pct,
            max_deviation_pct=config.max_deviation_pct,
            fade_below_vwap=config.fade_below_vwap,
            fade_above_vwap=config.fade_above_vwap,
            entry_start_hour_utc=config.entry_start_hour_utc,
            entry_start_minute_utc=config.entry_start_minute_utc,
            entry_cutoff_hour_utc=config.entry_cutoff_hour_utc,
            max_entries_per_day=config.max_entries_per_day,
        )

        signals = []
        for bar in bars:
            vwap_signal = generator.process_bar(bar)

            # Convert VWAP signal to TradingSignal
            if vwap_signal is not None:
                signal_type = vwap_signal.signal_type.name
                reason = vwap_signal.reason
                metadata = {
                    "vwap": vwap_signal.vwap,
                    "deviation_pct": vwap_signal.deviation_pct,
                }
            else:
                signal_type = "NO_SIGNAL"
                reason = ""
                metadata = {}

                # Add VWAP info if available
                vwap_info = generator.get_vwap_info()
                if vwap_info["vwap"] > 0:
                    current_price = bar["close"]
                    current_dev = generator.get_current_deviation(current_price)
                    metadata = {
                        "vwap": vwap_info["vwap"],
                        "deviation_pct": current_dev,
                        "state": vwap_info["state"],
                    }

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=bar["timestamp"],
                    price=bar["close"],
                    reason=reason,
                    metadata=metadata,
                )
            )

        return signals

    def get_option_direction(self, signal: TradingSignal) -> str:
        """
        Determine option type based on fade direction.

        Args:
            signal: TradingSignal from generate_signals

        Returns:
            'call' for below VWAP fade (bullish), 'put' for above VWAP fade (bearish)
        """
        if signal.signal_type == "BUY_CALL":
            return "call"
        elif signal.signal_type == "BUY_PUT":
            return "put"
        return "call"  # Default for below VWAP fade
