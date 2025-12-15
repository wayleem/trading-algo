"""
Signal generator for Morning Fade strategy.

Uses a state machine to:
1. Collect opening range (9:30-10:00 AM)
2. Determine opening direction at 10:00 AM
3. Generate fade signals during entry window (10:00-10:30 AM)
"""

import logging
from datetime import datetime, time, timedelta, timezone
from enum import Enum, auto
from typing import Dict, List, Optional

from strategies.morning_fade.models import MorningFadeSignal
from strategies.morning_fade.config import MorningFadeConfig

logger = logging.getLogger(__name__)


class MorningFadeState(Enum):
    """State machine states for morning fade strategy."""

    WAITING_FOR_OPEN = auto()  # Before market open
    COLLECTING_OPENING_RANGE = auto()  # 9:30-10:00 AM - collecting open price
    CALCULATING_DIRECTION = auto()  # At 10:00 AM - determine direction
    ENTRY_WINDOW = auto()  # 10:00-10:30 AM - can enter trade
    POSITION_OPEN = auto()  # Trade is open, waiting for exit
    CUTOFF_REACHED = auto()  # Past entry window, no new trades
    NO_TRADE_TODAY = auto()  # No clear direction, skip day


class MorningFadeSignalGenerator:
    """
    Generate signals for Morning Fade strategy.

    State Machine Flow (per day):
    1. WAITING_FOR_OPEN: Before 9:30 AM ET
    2. COLLECTING_OPENING_RANGE: 9:30-10:00 AM - Record open price
    3. CALCULATING_DIRECTION: At 10:00 AM - Compute move direction
    4. ENTRY_WINDOW: 10:00-10:30 AM - Generate signal if move > threshold
       - Up move -> PUT_SPREAD (fade up with bearish spread)
       - Down move -> CALL_SPREAD (fade down with bullish spread)
    5. CUTOFF_REACHED: After entry window or 2:00 PM exit time

    Usage:
        generator = MorningFadeSignalGenerator(config)
        for bar in bars:
            signal = generator.process_bar(bar)
            if signal and signal.signal_type != "NO_SIGNAL":
                # Enter trade
    """

    # Market hours in ET
    MARKET_OPEN_ET = time(9, 30)
    MARKET_CLOSE_ET = time(16, 0)

    def __init__(self, config: MorningFadeConfig):
        """
        Initialize signal generator.

        Args:
            config: MorningFadeConfig with strategy parameters
        """
        self._config = config

        # State tracking (reset daily)
        self._state = MorningFadeState.WAITING_FOR_OPEN
        self._current_day: Optional[datetime] = None

        # Opening range data
        self._open_price: Optional[float] = None
        self._direction_price: Optional[float] = None
        self._opening_move_pct: Optional[float] = None
        self._fade_direction: Optional[str] = None

        # Trade tracking
        self._trade_entered_today = False
        self._signal_generated_today = False

    def reset_day(self) -> None:
        """Reset state for new trading day."""
        self._state = MorningFadeState.WAITING_FOR_OPEN
        self._open_price = None
        self._direction_price = None
        self._opening_move_pct = None
        self._fade_direction = None
        self._trade_entered_today = False
        self._signal_generated_today = False

    def process_bar(self, bar: dict) -> Optional[MorningFadeSignal]:
        """
        Process a single bar and potentially generate a signal.

        Args:
            bar: OHLCV bar with timestamp, open, high, low, close, volume

        Returns:
            MorningFadeSignal or None
        """
        timestamp = bar["timestamp"]
        bar_date = timestamp.date() if hasattr(timestamp, "date") else timestamp

        # Check for new day
        if self._current_day is None or bar_date != self._current_day:
            self.reset_day()
            self._current_day = bar_date

        # Get ET time
        et_time = self._to_et(timestamp)
        current_time = et_time.time() if hasattr(et_time, "time") else et_time

        close_price = bar["close"]
        open_price = bar["open"]

        # State machine transitions
        if self._state == MorningFadeState.WAITING_FOR_OPEN:
            if current_time >= self.MARKET_OPEN_ET:
                self._state = MorningFadeState.COLLECTING_OPENING_RANGE
                self._open_price = open_price
                logger.debug(f"Market open: {open_price:.2f}")

        elif self._state == MorningFadeState.COLLECTING_OPENING_RANGE:
            # Store first open price
            if self._open_price is None:
                self._open_price = open_price

            # Check if we've reached direction time (10:00 AM)
            direction_time = time(
                self._config.range_end_hour_et,
                self._config.range_end_minute_et,
            )
            if current_time >= direction_time:
                self._state = MorningFadeState.CALCULATING_DIRECTION
                self._direction_price = close_price
                self._calculate_direction()

        elif self._state == MorningFadeState.CALCULATING_DIRECTION:
            # Transition immediately to entry window or no trade
            if self._fade_direction:
                self._state = MorningFadeState.ENTRY_WINDOW
            else:
                self._state = MorningFadeState.NO_TRADE_TODAY

        elif self._state == MorningFadeState.ENTRY_WINDOW:
            # Check if still in entry window
            if current_time > self._config.entry_end_time_et:
                self._state = MorningFadeState.CUTOFF_REACHED
            elif not self._signal_generated_today:
                # Generate signal
                self._signal_generated_today = True
                return self._create_signal(timestamp, close_price)

        elif self._state == MorningFadeState.POSITION_OPEN:
            # Check for exit time
            if current_time >= self._config.exit_by_time_et:
                self._state = MorningFadeState.CUTOFF_REACHED

        # Default: no signal
        return MorningFadeSignal(
            signal_type="NO_SIGNAL",
            timestamp=timestamp,
            underlying_price=close_price,
            open_price=self._open_price or open_price,
            direction_price=self._direction_price or close_price,
            opening_move_pct=self._opening_move_pct or 0.0,
            state=self._state.name,
            reason=f"State: {self._state.name}",
        )

    def _calculate_direction(self) -> None:
        """Calculate opening direction and determine fade direction."""
        if self._open_price is None or self._direction_price is None:
            return

        self._opening_move_pct = (
            (self._direction_price - self._open_price) / self._open_price * 100
        )

        threshold = self._config.fade_threshold_pct

        if self._opening_move_pct >= threshold:
            # Market moved UP -> Fade with PUT SPREAD (bearish)
            self._fade_direction = "PUT_SPREAD"
            logger.info(
                f"Morning fade: UP move {self._opening_move_pct:.2f}% -> PUT_SPREAD"
            )
        elif self._opening_move_pct <= -threshold:
            # Market moved DOWN -> Fade with CALL SPREAD (bullish)
            self._fade_direction = "CALL_SPREAD"
            logger.info(
                f"Morning fade: DOWN move {self._opening_move_pct:.2f}% -> CALL_SPREAD"
            )
        else:
            # No clear direction
            self._fade_direction = None
            logger.info(
                f"Morning fade: No clear direction {self._opening_move_pct:.2f}% "
                f"(threshold: {threshold}%)"
            )

    def _create_signal(
        self,
        timestamp: datetime,
        underlying_price: float,
    ) -> MorningFadeSignal:
        """Create a trade signal."""
        if not self._fade_direction:
            return MorningFadeSignal(
                signal_type="NO_SIGNAL",
                timestamp=timestamp,
                underlying_price=underlying_price,
                open_price=self._open_price or underlying_price,
                direction_price=self._direction_price or underlying_price,
                opening_move_pct=self._opening_move_pct or 0.0,
                state=self._state.name,
                reason="No clear direction",
            )

        # Calculate strikes
        short_strike, long_strike = self._calculate_strikes(
            underlying_price, self._fade_direction
        )

        return MorningFadeSignal(
            signal_type=self._fade_direction,
            timestamp=timestamp,
            underlying_price=underlying_price,
            open_price=self._open_price,
            direction_price=self._direction_price,
            opening_move_pct=self._opening_move_pct,
            short_strike=short_strike,
            long_strike=long_strike,
            state=self._state.name,
            reason=f"Fade {self._opening_move_pct:.2f}% opening move",
        )

    def _calculate_strikes(
        self,
        underlying_price: float,
        direction: str,
    ) -> tuple:
        """
        Calculate short and long strikes for the spread.

        Args:
            underlying_price: Current underlying price
            direction: "CALL_SPREAD" or "PUT_SPREAD"

        Returns:
            (short_strike, long_strike)
        """
        offset = self._config.short_strike_offset
        width = self._config.spread_width
        interval = self._config.strike_interval

        # Round to strike interval
        atm_strike = round(underlying_price / interval) * interval

        if direction == "PUT_SPREAD":
            # Bearish: Sell put above, buy put below
            short_strike = atm_strike - offset
            long_strike = short_strike - width
        else:  # CALL_SPREAD
            # Bullish: Sell call below, buy call above
            short_strike = atm_strike + offset
            long_strike = short_strike + width

        return (short_strike, long_strike)

    def _to_et(self, timestamp: datetime) -> datetime:
        """Convert timestamp to Eastern Time."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        et_offset = timezone(timedelta(hours=-5))  # EST (simplified)
        return timestamp.astimezone(et_offset)

    def mark_position_open(self) -> None:
        """Mark that a position has been opened."""
        self._trade_entered_today = True
        self._state = MorningFadeState.POSITION_OPEN

    def mark_position_closed(self) -> None:
        """Mark that the position has been closed."""
        self._state = MorningFadeState.CUTOFF_REACHED

    @property
    def current_state(self) -> MorningFadeState:
        """Get current state machine state."""
        return self._state

    @property
    def opening_move(self) -> Optional[float]:
        """Get the opening move percentage."""
        return self._opening_move_pct

    @property
    def fade_direction(self) -> Optional[str]:
        """Get the determined fade direction."""
        return self._fade_direction


def generate_morning_fade_signals(
    bars: List[dict],
    config: MorningFadeConfig,
) -> List[MorningFadeSignal]:
    """
    Generate morning fade signals for a list of bars.

    Convenience function for batch processing.

    Args:
        bars: List of OHLCV bars
        config: MorningFadeConfig

    Returns:
        List of MorningFadeSignal (one per bar)
    """
    generator = MorningFadeSignalGenerator(config)
    signals = []

    for bar in bars:
        signal = generator.process_bar(bar)
        signals.append(signal)

    return signals
