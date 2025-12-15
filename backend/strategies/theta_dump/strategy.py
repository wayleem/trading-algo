"""
Late Day Momentum Strategy (3PM Theta Dump).

Trades directional momentum after 3PM when options decay rapidly.
"""

import logging
import numpy as np
from typing import List, Optional
from datetime import datetime

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.theta_dump.config import ThetaDumpConfig

logger = logging.getLogger(__name__)


class ThetaDumpStrategy(BaseStrategy):
    """
    Late Day Momentum Strategy.

    Trades directional momentum after 3PM.

    Rationale:
    - After 3PM, theta decay accelerates on 0DTE options
    - Directional moves can still generate profit despite decay
    - Strong trends into close often continue
    - MOC order flow can drive directional momentum

    Strategy Rules:
    1. Only trade after 3:00 PM ET
    2. Identify strong intraday momentum
    3. Enter in momentum direction
    4. Quick profit target to beat theta
    """

    def __init__(self):
        """Initialize Late Day Momentum strategy."""
        pass

    @property
    def name(self) -> str:
        return "theta_dump"

    @property
    def description(self) -> str:
        return "Late day momentum - trades directional moves after 3PM"

    def get_default_config(self) -> ThetaDumpConfig:
        """Return default configuration."""
        return ThetaDumpConfig(
            late_day_start_utc=20,
            late_day_start_minute=0,
            momentum_lookback=15,
            momentum_threshold=0.12,
            profit_target_pct=0.15,
            stop_loss_pct=0.10,
        )

    def get_entry_filters(self) -> List:
        """Late day timing defines entry."""
        return []

    def _calculate_vwap(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
    ) -> List[float]:
        """Calculate cumulative VWAP."""
        vwap_values = []
        cumulative_tpv = 0.0
        cumulative_vol = 0.0

        for h, l, c, v in zip(highs, lows, closes, volumes):
            typical_price = (h + l + c) / 3
            cumulative_tpv += typical_price * v
            cumulative_vol += v

            if cumulative_vol > 0:
                vwap = cumulative_tpv / cumulative_vol
            else:
                vwap = c

            vwap_values.append(vwap)

        return vwap_values

    async def generate_signals(
        self,
        bars: List[dict],
        config: ThetaDumpConfig,
    ) -> List[TradingSignal]:
        """
        Generate late day momentum signals.

        Args:
            bars: OHLCV bars
            config: ThetaDumpConfig instance

        Returns:
            List of TradingSignal for each bar
        """
        if not bars:
            return []

        # Extract data
        highs = [bar["high"] for bar in bars]
        lows = [bar["low"] for bar in bars]
        closes = [bar["close"] for bar in bars]
        volumes = [bar.get("volume", 0) for bar in bars]
        timestamps = [bar["timestamp"] for bar in bars]

        signals = []
        current_date = None
        day_highs = []
        day_lows = []
        day_closes = []
        day_volumes = []
        signal_fired_today = False

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
                day_highs = []
                day_lows = []
                day_closes = []
                day_volumes = []
                signal_fired_today = False

            # Accumulate day data
            day_highs.append(highs[i])
            day_lows.append(lows[i])
            day_closes.append(closes[i])
            day_volumes.append(volumes[i])

            # Calculate VWAP
            day_vwap = self._calculate_vwap(day_highs, day_lows, day_closes, day_volumes)
            current_vwap = day_vwap[-1] if day_vwap else closes[i]

            close = closes[i]

            # Check if we're in late day window
            in_late_day = (
                bar_hour > config.late_day_start_utc or
                (bar_hour == config.late_day_start_utc and bar_minute >= config.late_day_start_minute)
            )

            # Calculate momentum
            momentum_direction = None
            momentum_pct = 0.0
            if len(day_closes) >= config.momentum_lookback:
                lookback_close = day_closes[-config.momentum_lookback]
                momentum_pct = ((close - lookback_close) / lookback_close) * 100

                if momentum_pct >= config.momentum_threshold:
                    momentum_direction = "bullish"
                elif momentum_pct <= -config.momentum_threshold:
                    momentum_direction = "bearish"

            # Calculate acceleration
            acceleration_ok = True
            if config.require_acceleration and len(day_closes) >= config.momentum_lookback * 2:
                half_lookback = config.momentum_lookback // 2
                if len(day_closes) > half_lookback:
                    recent_close = day_closes[-half_lookback]
                    recent_momentum = ((close - recent_close) / recent_close) * 100
                    # Recent momentum should be stronger than overall momentum per bar
                    acceleration_ok = abs(recent_momentum) > abs(momentum_pct) * (half_lookback / config.momentum_lookback)

            # VWAP confirmation
            vwap_confirms = True
            if config.use_vwap_filter:
                if momentum_direction == "bullish":
                    vwap_confirms = close > current_vwap
                elif momentum_direction == "bearish":
                    vwap_confirms = close < current_vwap

            # Generate signal
            if in_late_day and not signal_fired_today and momentum_direction:
                if vwap_confirms and acceleration_ok:
                    if momentum_direction == "bullish":
                        signal_type = "BUY_CALL"
                        reason = f"Late day bullish: {momentum_pct:.2f}% momentum, price > VWAP"
                        signal_fired_today = True
                    elif momentum_direction == "bearish":
                        signal_type = "BUY_PUT"
                        reason = f"Late day bearish: {momentum_pct:.2f}% momentum, price < VWAP"
                        signal_fired_today = True

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=ts,
                    price=close,
                    reason=reason,
                    metadata={
                        "vwap": current_vwap,
                        "momentum_pct": momentum_pct,
                        "momentum_direction": momentum_direction,
                        "in_late_day": in_late_day,
                    },
                )
            )

        return signals

    def get_option_direction(self, signal: TradingSignal) -> str:
        """Determine option type based on signal."""
        if signal.signal_type == "BUY_CALL":
            return "call"
        elif signal.signal_type == "BUY_PUT":
            return "put"
        return "call"
