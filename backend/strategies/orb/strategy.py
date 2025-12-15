"""
ORB (Opening Range Breakout) Strategy.

Trades breakouts from the opening range established in the first 30-60 minutes.
"""

import logging
from typing import List, Optional

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.orb.config import ORBConfig
from app.services.orb_signal_generator import ORBSignalGenerator, ORBSignal

logger = logging.getLogger(__name__)


class ORBStrategy(BaseStrategy):
    """
    Opening Range Breakout Strategy.

    Establishes a high/low range during the first N minutes of market open,
    then trades breakouts in either direction.

    Rationale:
    - Opening range captures overnight positioning and early momentum
    - Breakout indicates directional conviction
    - 0DTE options benefit from strong directional moves

    Strategy Rules:
    1. Track high/low from 9:30-10:30 AM ET (configurable)
    2. After range is defined, watch for bar CLOSE above/below range
    3. Breakout UP -> BUY CALL
    4. Breakout DOWN -> BUY PUT
    5. Allow re-entry if stopped out and new breakout occurs
    """

    def __init__(self):
        """Initialize ORB strategy."""
        self._signal_generator: Optional[ORBSignalGenerator] = None

    @property
    def name(self) -> str:
        return "orb"

    @property
    def description(self) -> str:
        return "Opening Range Breakout - trades first hour range breakouts"

    def get_default_config(self) -> ORBConfig:
        """Return default ORB strategy configuration."""
        return ORBConfig(
            # Range settings
            range_minutes=60,
            breakout_buffer=0.0,
            require_close=True,
            max_entries_per_day=3,
            # Exit settings
            profit_target_pct=0.30,
            stop_loss_pct=0.20,
            # Entry cutoff
            entry_cutoff_hour_utc=18,  # 1 PM ET
        )

    def get_entry_filters(self) -> List:
        """ORB doesn't need additional filters - range defines entry."""
        return []

    async def generate_signals(
        self,
        bars: List[dict],
        config: ORBConfig,
    ) -> List[TradingSignal]:
        """
        Generate ORB signals from price bars.

        Args:
            bars: OHLCV bars (1-minute preferred)
            config: ORBConfig instance

        Returns:
            List of TradingSignal for each bar
        """
        if not bars:
            return []

        # Initialize signal generator with config
        generator = ORBSignalGenerator(
            range_minutes=config.range_minutes,
            entry_cutoff_hour_utc=config.entry_cutoff_hour_utc,
            breakout_buffer=config.breakout_buffer,
            require_close=config.require_close,
        )
        generator.max_entries_per_day = config.max_entries_per_day

        signals = []
        for bar in bars:
            orb_signal = generator.process_bar(bar)

            # Convert ORB signal to TradingSignal
            if orb_signal is not None:
                signal_type = orb_signal.signal_type.name
                reason = orb_signal.reason
                metadata = {
                    "range_high": orb_signal.range_high,
                    "range_low": orb_signal.range_low,
                    "range_size": orb_signal.range_high - orb_signal.range_low,
                }

                # Check if range is tradeable
                range_size = orb_signal.range_high - orb_signal.range_low
                if not config.is_range_tradeable(range_size):
                    signal_type = "NO_SIGNAL"
                    reason = f"Range size ${range_size:.2f} outside tradeable bounds"
            else:
                signal_type = "NO_SIGNAL"
                reason = ""
                metadata = {}

                # Add range info if available
                range_info = generator.get_range_info()
                if range_info:
                    metadata = {
                        "range_high": range_info["high"],
                        "range_low": range_info["low"],
                        "range_size": range_info["range_size"],
                        "state": range_info["state"],
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
        Determine option type based on breakout direction.

        Args:
            signal: TradingSignal from generate_signals

        Returns:
            'call' for upside breakout, 'put' for downside breakout
        """
        if signal.signal_type == "BUY_CALL":
            return "call"
        elif signal.signal_type == "BUY_PUT":
            return "put"
        return "call"  # Default
