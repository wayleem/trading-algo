"""
First Pullback Strategy.

Enters on the first pullback after an initial trend move.
"""

import logging
import numpy as np
from typing import List, Optional
from datetime import datetime
from enum import Enum

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.first_pullback.config import FirstPullbackConfig

logger = logging.getLogger(__name__)


class TrendState(Enum):
    """State of the trend for the current day."""
    NO_TREND = "no_trend"
    BULLISH_TREND = "bullish_trend"
    BEARISH_TREND = "bearish_trend"
    BULLISH_PULLBACK = "bullish_pullback"
    BEARISH_PULLBACK = "bearish_pullback"
    SIGNAL_FIRED = "signal_fired"


class FirstPullbackStrategy(BaseStrategy):
    """
    First Pullback Strategy.

    Waits for initial trend move, then enters on first pullback.

    Rationale:
    - First pullback after a strong move often continues
    - Lower risk entry than chasing the initial move
    - Defined stop level (below pullback low)

    Strategy Rules:
    1. Wait for initial move >= threshold (establish trend)
    2. Wait for pullback of minimum % from high/low
    3. Enter when pullback shows signs of ending
    4. Stop below pullback low (bullish) or above pullback high (bearish)
    """

    def __init__(self):
        """Initialize First Pullback strategy."""
        pass

    @property
    def name(self) -> str:
        return "first_pullback"

    @property
    def description(self) -> str:
        return "First pullback entry after initial trend move"

    def get_default_config(self) -> FirstPullbackConfig:
        """Return default First Pullback configuration."""
        return FirstPullbackConfig(
            # Trend settings
            initial_move_lookback=30,
            initial_move_pct=0.3,
            trend_ema_period=20,
            # Pullback settings
            pullback_pct=0.10,
            max_pullback_pct=0.25,
            # Exit settings
            profit_target_pct=0.20,
            stop_loss_pct=0.15,
            use_swing_stop=True,
            # Entry timing
            entry_start_minutes=30,
            entry_cutoff_hour_utc=18,
        )

    def get_entry_filters(self) -> List:
        """Pullback pattern defines entry - no additional filters."""
        return []

    def _calculate_ema(
        self,
        values: List[float],
        period: int,
    ) -> List[float]:
        """
        Calculate Exponential Moving Average.

        Args:
            values: Price values
            period: EMA period

        Returns:
            List of EMA values
        """
        ema_values = []
        multiplier = 2 / (period + 1)

        for i, val in enumerate(values):
            if i == 0:
                ema_values.append(val)
            else:
                ema = (val * multiplier) + (ema_values[-1] * (1 - multiplier))
                ema_values.append(ema)

        return ema_values

    async def generate_signals(
        self,
        bars: List[dict],
        config: FirstPullbackConfig,
    ) -> List[TradingSignal]:
        """
        Generate first pullback signals from price bars.

        Args:
            bars: OHLCV bars
            config: FirstPullbackConfig instance

        Returns:
            List of TradingSignal for each bar
        """
        if not bars:
            return []

        # Extract data
        highs = [bar["high"] for bar in bars]
        lows = [bar["low"] for bar in bars]
        closes = [bar["close"] for bar in bars]
        timestamps = [bar["timestamp"] for bar in bars]

        # Calculate EMA for trend confirmation
        ema = self._calculate_ema(closes, config.trend_ema_period)

        # State tracking (resets daily)
        signals = []
        current_date = None
        state = TrendState.NO_TREND
        day_open = None
        day_high = None
        day_low = None
        trend_high = None  # Highest point in bullish trend
        trend_low = None  # Lowest point in bearish trend
        pullback_low = None  # Lowest point in bullish pullback
        pullback_high = None  # Highest point in bearish pullback
        pullback_start_bar = None

        for i in range(len(bars)):
            signal_type = "NO_SIGNAL"
            reason = ""

            ts = timestamps[i]
            bar_date = ts.date() if hasattr(ts, "date") else ts
            bar_hour = ts.hour if hasattr(ts, "hour") else 0
            bar_minute = ts.minute if hasattr(ts, "minute") else 0

            # Reset on new day
            if current_date != bar_date:
                current_date = bar_date
                state = TrendState.NO_TREND
                day_open = closes[i]
                day_high = highs[i]
                day_low = lows[i]
                trend_high = None
                trend_low = None
                pullback_low = None
                pullback_high = None
                pullback_start_bar = None

            # Update day high/low
            day_high = max(day_high, highs[i]) if day_high else highs[i]
            day_low = min(day_low, lows[i]) if day_low else lows[i]

            close = closes[i]
            high = highs[i]
            low = lows[i]

            # Check entry timing
            minutes_from_open = (bar_hour - 14) * 60 + bar_minute
            past_entry_start = minutes_from_open >= config.entry_start_minutes
            before_cutoff = bar_hour < config.entry_cutoff_hour_utc

            # State machine
            if state == TrendState.NO_TREND and past_entry_start:
                # Check for initial bullish move
                if day_open and day_high:
                    bullish_move = ((day_high - day_open) / day_open) * 100
                    if bullish_move >= config.initial_move_pct and close > ema[i]:
                        state = TrendState.BULLISH_TREND
                        trend_high = day_high
                        reason = f"Bullish trend established: +{bullish_move:.2f}%"

                # Check for initial bearish move
                if day_open and day_low:
                    bearish_move = ((day_open - day_low) / day_open) * 100
                    if bearish_move >= config.initial_move_pct and close < ema[i]:
                        state = TrendState.BEARISH_TREND
                        trend_low = day_low
                        reason = f"Bearish trend established: -{bearish_move:.2f}%"

            elif state == TrendState.BULLISH_TREND:
                # Update trend high
                if high > trend_high:
                    trend_high = high

                # Check for pullback start
                pullback_from_high = ((trend_high - close) / trend_high) * 100
                if pullback_from_high >= config.pullback_pct:
                    state = TrendState.BULLISH_PULLBACK
                    pullback_low = low
                    pullback_start_bar = i

            elif state == TrendState.BEARISH_TREND:
                # Update trend low
                if low < trend_low:
                    trend_low = low

                # Check for pullback start
                pullback_from_low = ((close - trend_low) / trend_low) * 100
                if pullback_from_low >= config.pullback_pct:
                    state = TrendState.BEARISH_PULLBACK
                    pullback_high = high
                    pullback_start_bar = i

            elif state == TrendState.BULLISH_PULLBACK and before_cutoff:
                # Update pullback low
                pullback_low = min(pullback_low, low) if pullback_low else low
                pullback_bars = i - pullback_start_bar if pullback_start_bar else 0

                # Check if pullback is too deep
                pullback_depth = ((trend_high - pullback_low) / trend_high) * 100
                if pullback_depth > config.max_pullback_pct:
                    state = TrendState.NO_TREND
                    reason = f"Pullback too deep: {pullback_depth:.2f}%"
                # Check if pullback is too long
                elif pullback_bars > config.pullback_bars_max:
                    state = TrendState.NO_TREND
                    reason = "Pullback too long"
                # Check for entry signal
                elif pullback_bars >= config.pullback_bars_min:
                    # Higher low confirmation
                    higher_low_ok = not config.require_higher_low or low > pullback_low
                    # Momentum reversal (close above previous close)
                    momentum_ok = not config.require_momentum_reversal or (i > 0 and close > closes[i - 1])

                    if higher_low_ok and momentum_ok and close > pullback_low:
                        signal_type = "BUY_CALL"
                        reason = f"Bullish pullback entry: pullback {pullback_depth:.2f}% from high"
                        state = TrendState.SIGNAL_FIRED

            elif state == TrendState.BEARISH_PULLBACK and before_cutoff:
                # Update pullback high
                pullback_high = max(pullback_high, high) if pullback_high else high
                pullback_bars = i - pullback_start_bar if pullback_start_bar else 0

                # Check if pullback is too deep
                pullback_depth = ((pullback_high - trend_low) / trend_low) * 100
                if pullback_depth > config.max_pullback_pct:
                    state = TrendState.NO_TREND
                    reason = f"Pullback too deep: {pullback_depth:.2f}%"
                # Check if pullback is too long
                elif pullback_bars > config.pullback_bars_max:
                    state = TrendState.NO_TREND
                    reason = "Pullback too long"
                # Check for entry signal
                elif pullback_bars >= config.pullback_bars_min:
                    # Lower high confirmation
                    lower_high_ok = not config.require_higher_low or high < pullback_high
                    # Momentum reversal (close below previous close)
                    momentum_ok = not config.require_momentum_reversal or (i > 0 and close < closes[i - 1])

                    if lower_high_ok and momentum_ok and close < pullback_high:
                        signal_type = "BUY_PUT"
                        reason = f"Bearish pullback entry: pullback {pullback_depth:.2f}% from low"
                        state = TrendState.SIGNAL_FIRED

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=ts,
                    price=close,
                    reason=reason,
                    metadata={
                        "state": state.value,
                        "ema": ema[i],
                        "day_high": day_high,
                        "day_low": day_low,
                        "trend_high": trend_high,
                        "trend_low": trend_low,
                        "pullback_low": pullback_low,
                        "pullback_high": pullback_high,
                    },
                )
            )

        return signals

    def get_option_direction(self, signal: TradingSignal) -> str:
        """
        Determine option type based on pullback direction.

        Args:
            signal: TradingSignal from generate_signals

        Returns:
            'call' for bullish pullback, 'put' for bearish pullback
        """
        if signal.signal_type == "BUY_CALL":
            return "call"
        elif signal.signal_type == "BUY_PUT":
            return "put"
        return "call"
