"""
Morning Fade Strategy - fades the opening move with credit spreads.

This strategy:
1. Monitors the first 30 minutes (9:30-10:00 AM) to determine direction
2. If significant move, fades it with a credit spread (10:00-10:30 AM)
3. Exits by 2:00 PM or on profit target / stop loss
"""

import logging
from datetime import time
from typing import List, Optional

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.base.filter import TimeWindowFilter, ExitTimeFilter, CompositeFilter
from strategies.morning_fade.config import MorningFadeConfig
from strategies.morning_fade.signal_generator import (
    MorningFadeSignalGenerator,
    generate_morning_fade_signals,
)
from strategies.morning_fade.models import MorningFadeSignal

logger = logging.getLogger(__name__)


class MorningFadeStrategy(BaseStrategy):
    """
    Morning Fade Strategy for 0DTE options.

    Strategy Logic:
    1. Wait for 10:00 AM ET (30 min after open)
    2. Check direction of morning move (open to 10 AM)
    3. If move > threshold, fade the direction:
       - Strong up move -> PUT_SPREAD (bearish credit spread)
       - Strong down move -> CALL_SPREAD (bullish credit spread)
    4. Exit by 2 PM ET or on profit target/stop loss

    Rationale:
    - Morning moves often overextend on retail FOMO
    - Mean reversion typically occurs mid-day
    - Credit spreads define max loss and benefit from theta decay
    - Early exit (2 PM) avoids EOD theta crush

    Note: This strategy generates signals for credit spreads, which are
    different from the simple long options in the base backtest infrastructure.
    The signals are converted to BUY_PUT/BUY_CALL for compatibility, but
    the metadata contains the full spread information.
    """

    @property
    def name(self) -> str:
        return "morning_fade"

    @property
    def description(self) -> str:
        return "Fades the morning move (10:00-10:30 AM entry) with credit spreads, exits by 2 PM"

    def get_default_config(self) -> MorningFadeConfig:
        """Return default morning fade configuration."""
        return MorningFadeConfig(
            # Time settings
            entry_start_time_et=time(10, 0),
            entry_end_time_et=time(10, 30),
            exit_by_time_et=time(14, 0),
            # Fade settings
            fade_threshold_pct=0.3,  # 0.3% move to trigger
            spread_width=3.0,  # $3 wide spread
            short_strike_offset=2.0,  # $2 OTM
            # Exit settings
            profit_target_pct=0.50,  # 50% of credit
            stop_loss_pct=1.50,  # 150% of credit
            # Position sizing
            base_contracts=1,
        )

    def get_entry_filters(self) -> List:
        """
        Return filters for morning fade strategy.

        Filters:
        1. TimeWindowFilter: Only enter 10:00-10:30 AM ET
        2. ExitTimeFilter: Don't enter too close to 2 PM exit
        """
        config = self.get_default_config()

        return [
            TimeWindowFilter(
                start_hour_et=config.entry_start_time_et.hour,
                start_minute_et=config.entry_start_time_et.minute,
                end_hour_et=config.entry_end_time_et.hour,
                end_minute_et=config.entry_end_time_et.minute,
            ),
            ExitTimeFilter(
                exit_hour_et=config.exit_by_time_et.hour,
                exit_minute_et=config.exit_by_time_et.minute,
                buffer_minutes=config.exit_buffer_minutes,
            ),
        ]

    async def generate_signals(
        self,
        bars: List[dict],
        config: MorningFadeConfig,
    ) -> List[TradingSignal]:
        """
        Generate morning fade signals.

        Uses state machine to:
        1. Track opening range
        2. Determine direction at 10 AM
        3. Generate fade signal if move > threshold

        The signals are converted to TradingSignal format for compatibility
        with the base framework, with spread details in metadata.

        Args:
            bars: OHLCV bars
            config: MorningFadeConfig

        Returns:
            List of TradingSignal with spread metadata
        """
        if not bars:
            return []

        # Generate morning fade specific signals
        morning_signals = generate_morning_fade_signals(bars, config)

        # Convert to TradingSignal format
        signals = []
        for mf_signal in morning_signals:
            # Map spread direction to simple signal type
            # This is for compatibility with existing infrastructure
            if mf_signal.signal_type == "PUT_SPREAD":
                # Bearish credit spread -> similar to buying put direction
                signal_type = "BUY_PUT"
            elif mf_signal.signal_type == "CALL_SPREAD":
                # Bullish credit spread -> similar to buying call direction
                signal_type = "BUY_CALL"
            else:
                signal_type = "NO_SIGNAL"

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=mf_signal.timestamp,
                    price=mf_signal.underlying_price,
                    reason=mf_signal.reason,
                    metadata={
                        "spread_type": mf_signal.signal_type,
                        "short_strike": mf_signal.short_strike,
                        "long_strike": mf_signal.long_strike,
                        "opening_move_pct": mf_signal.opening_move_pct,
                        "open_price": mf_signal.open_price,
                        "direction_price": mf_signal.direction_price,
                        "state": mf_signal.state,
                        "is_spread": mf_signal.signal_type in ("PUT_SPREAD", "CALL_SPREAD"),
                    },
                )
            )

        return signals

    def validate_config(self, config: MorningFadeConfig) -> List[str]:
        """Validate morning fade specific configuration."""
        errors = super().validate_config(config)

        if config.fade_threshold_pct <= 0:
            errors.append("fade_threshold_pct must be positive")

        if config.spread_width <= 0:
            errors.append("spread_width must be positive")

        if config.profit_target_pct <= 0 or config.profit_target_pct >= 1:
            errors.append("profit_target_pct should be between 0 and 1 (e.g., 0.50 for 50%)")

        if config.entry_start_time_et >= config.entry_end_time_et:
            errors.append("entry_start_time must be before entry_end_time")

        if config.entry_end_time_et >= config.exit_by_time_et:
            errors.append("entry_end_time must be before exit_by_time")

        return errors
