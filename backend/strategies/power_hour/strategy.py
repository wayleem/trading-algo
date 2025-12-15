"""
Power Hour Momentum Strategy.

Trades momentum in the last 90 minutes (2:30-4:00 PM ET).
"""

import logging
import numpy as np
from typing import List, Optional
from datetime import datetime

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.power_hour.config import PowerHourConfig

logger = logging.getLogger(__name__)


class PowerHourStrategy(BaseStrategy):
    """
    Power Hour Momentum Strategy.

    Trades trend continuation in the last 90 minutes of the trading day.

    Rationale:
    - Last 90 minutes often see increased volume and trend acceleration
    - Institutional traders position before close
    - MOC (Market On Close) orders drive directional moves
    - 0DTE gamma exposure intensifies price moves

    Strategy Rules:
    1. Wait until 2:30 PM ET (power hour)
    2. Identify intraday trend direction (VWAP + price action)
    3. Enter in trend direction on pullback or breakout
    4. Ride momentum into the close
    """

    def __init__(self):
        """Initialize Power Hour strategy."""
        pass

    @property
    def name(self) -> str:
        return "power_hour"

    @property
    def description(self) -> str:
        return "Power hour momentum - trades last 90 min trend acceleration"

    def get_default_config(self) -> PowerHourConfig:
        """Return default Power Hour configuration."""
        return PowerHourConfig(
            # Time window
            power_hour_start_utc=19,
            power_hour_start_minute=30,
            # Trend detection
            trend_lookback=30,
            momentum_threshold=0.20,
            use_vwap_trend=True,
            # Exit settings
            profit_target_pct=0.25,
            stop_loss_pct=0.15,
        )

    def get_entry_filters(self) -> List:
        """Power hour timing defines entry - no additional filters."""
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

    def _calculate_volume_sma(
        self,
        volumes: List[float],
        period: int,
    ) -> List[float]:
        """Calculate Simple Moving Average of volume."""
        sma_values = []
        for i in range(len(volumes)):
            if i < period - 1:
                sma_values.append(np.mean(volumes[: i + 1]) if volumes[: i + 1] else 0)
            else:
                sma_values.append(np.mean(volumes[i - period + 1 : i + 1]))
        return sma_values

    async def generate_signals(
        self,
        bars: List[dict],
        config: PowerHourConfig,
    ) -> List[TradingSignal]:
        """
        Generate power hour momentum signals.

        Args:
            bars: OHLCV bars
            config: PowerHourConfig instance

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

        # Calculate indicators (reset daily)
        signals = []
        current_date = None
        day_vwap = []
        day_highs = []
        day_lows = []
        day_closes = []
        day_volumes = []
        day_high = None
        day_low = None
        day_open = None
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
                day_vwap = []
                day_highs = []
                day_lows = []
                day_closes = []
                day_volumes = []
                day_high = highs[i]
                day_low = lows[i]
                day_open = closes[i]
                signal_fired_today = False

            # Accumulate day data
            day_highs.append(highs[i])
            day_lows.append(lows[i])
            day_closes.append(closes[i])
            day_volumes.append(volumes[i])

            # Update day high/low
            day_high = max(day_high, highs[i]) if day_high else highs[i]
            day_low = min(day_low, lows[i]) if day_low else lows[i]

            # Calculate VWAP for the day
            day_vwap = self._calculate_vwap(day_highs, day_lows, day_closes, day_volumes)
            current_vwap = day_vwap[-1] if day_vwap else closes[i]

            # Calculate volume SMA
            volume_sma = self._calculate_volume_sma(day_volumes, 20)
            current_vol_sma = volume_sma[-1] if volume_sma else 1

            close = closes[i]
            high = highs[i]
            low = lows[i]
            volume = volumes[i]

            # Check if we're in power hour
            in_power_hour = (
                bar_hour > config.power_hour_start_utc or
                (bar_hour == config.power_hour_start_utc and bar_minute >= config.power_hour_start_minute)
            )

            # Calculate trend metrics
            trend_direction = None
            if len(day_closes) >= config.trend_lookback:
                lookback_close = day_closes[-config.trend_lookback]
                move_pct = ((close - lookback_close) / lookback_close) * 100

                if move_pct >= config.momentum_threshold:
                    trend_direction = "bullish"
                elif move_pct <= -config.momentum_threshold:
                    trend_direction = "bearish"

            # VWAP trend confirmation
            vwap_confirms = True
            if config.use_vwap_trend:
                if trend_direction == "bullish":
                    vwap_confirms = close > current_vwap
                elif trend_direction == "bearish":
                    vwap_confirms = close < current_vwap

            # Volume confirmation
            vol_ratio = volume / current_vol_sma if current_vol_sma > 0 else 0
            volume_confirms = not config.require_volume_surge or vol_ratio >= config.volume_surge_threshold

            # New high/low confirmation
            new_extreme = True
            if config.require_new_high_low:
                if trend_direction == "bullish":
                    new_extreme = high >= day_high * 0.999  # Within 0.1% of day high
                elif trend_direction == "bearish":
                    new_extreme = low <= day_low * 1.001  # Within 0.1% of day low

            # Generate signal
            if in_power_hour and not signal_fired_today and trend_direction:
                if vwap_confirms and volume_confirms and new_extreme:
                    if trend_direction == "bullish":
                        signal_type = "BUY_CALL"
                        reason = f"Power hour bullish: price ${close:.2f} > VWAP ${current_vwap:.2f}, vol {vol_ratio:.1f}x"
                        signal_fired_today = True
                    elif trend_direction == "bearish":
                        signal_type = "BUY_PUT"
                        reason = f"Power hour bearish: price ${close:.2f} < VWAP ${current_vwap:.2f}, vol {vol_ratio:.1f}x"
                        signal_fired_today = True

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=ts,
                    price=close,
                    reason=reason,
                    metadata={
                        "vwap": current_vwap,
                        "day_high": day_high,
                        "day_low": day_low,
                        "trend_direction": trend_direction,
                        "volume_ratio": vol_ratio,
                        "in_power_hour": in_power_hour,
                    },
                )
            )

        return signals

    def get_option_direction(self, signal: TradingSignal) -> str:
        """Determine option type based on trend direction."""
        if signal.signal_type == "BUY_CALL":
            return "call"
        elif signal.signal_type == "BUY_PUT":
            return "put"
        return "call"
