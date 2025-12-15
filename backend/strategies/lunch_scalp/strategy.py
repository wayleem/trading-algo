"""
Lunch Scalp Mean Reversion Strategy.

Trades mean reversion during the 11:30 AM - 1:30 PM ET lunch hour.
"""

import logging
import numpy as np
from typing import List, Optional
from datetime import datetime

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.lunch_scalp.config import LunchScalpConfig

logger = logging.getLogger(__name__)


class LunchScalpStrategy(BaseStrategy):
    """
    Lunch Scalp Mean Reversion Strategy.

    Trades mean reversion during the lunch hour doldrums.

    Rationale:
    - Lunch hour (11:30 AM - 1:30 PM ET) is typically low volume
    - Price tends to mean-revert and chop in a range
    - Ideal for quick scalps as trends rarely develop
    - Lower volatility means tighter stops are viable

    Strategy Rules:
    1. Only trade during lunch window (11:30 AM - 1:30 PM ET)
    2. Look for price deviation from VWAP
    3. Enter opposite direction (mean reversion)
    4. Quick profit target, tight stop loss
    """

    def __init__(self):
        """Initialize Lunch Scalp strategy."""
        pass

    @property
    def name(self) -> str:
        return "lunch_scalp"

    @property
    def description(self) -> str:
        return "Lunch hour mean reversion - scalps VWAP deviation during midday chop"

    def get_default_config(self) -> LunchScalpConfig:
        """Return default Lunch Scalp configuration."""
        return LunchScalpConfig(
            # Time window
            lunch_start_utc=16,
            lunch_start_minute=30,
            lunch_end_utc=18,
            lunch_end_minute=30,
            # Mean reversion
            vwap_deviation_pct=0.10,
            # Exit settings
            profit_target_pct=0.10,
            stop_loss_pct=0.08,
        )

    def get_entry_filters(self) -> List:
        """Lunch timing defines entry - no additional filters."""
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

    def _calculate_rsi(self, closes: List[float], period: int) -> List[float]:
        """Calculate RSI."""
        rsi_values = []
        gains = []
        losses = []

        for i in range(len(closes)):
            if i == 0:
                rsi_values.append(50.0)
                continue

            change = closes[i] - closes[i - 1]
            gain = max(0, change)
            loss = max(0, -change)

            gains.append(gain)
            losses.append(loss)

            if len(gains) < period:
                rsi_values.append(50.0)
                continue

            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))

        return rsi_values

    async def generate_signals(
        self,
        bars: List[dict],
        config: LunchScalpConfig,
    ) -> List[TradingSignal]:
        """
        Generate lunch scalp mean reversion signals.

        Args:
            bars: OHLCV bars
            config: LunchScalpConfig instance

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
        trades_today = 0

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
                trades_today = 0

            # Accumulate day data
            day_highs.append(highs[i])
            day_lows.append(lows[i])
            day_closes.append(closes[i])
            day_volumes.append(volumes[i])

            # Calculate VWAP for the day
            day_vwap = self._calculate_vwap(day_highs, day_lows, day_closes, day_volumes)
            current_vwap = day_vwap[-1] if day_vwap else closes[i]

            # Calculate volume SMA
            volume_sma = self._calculate_volume_sma(day_volumes, 20)
            current_vol_sma = volume_sma[-1] if volume_sma else 1

            # Calculate RSI for the day
            rsi_values = self._calculate_rsi(day_closes, 14)
            current_rsi = rsi_values[-1] if rsi_values else 50.0

            close = closes[i]
            volume = volumes[i]

            # Check if we're in lunch window
            in_lunch = False
            time_minutes = bar_hour * 60 + bar_minute
            lunch_start_minutes = config.lunch_start_utc * 60 + config.lunch_start_minute
            lunch_end_minutes = config.lunch_end_utc * 60 + config.lunch_end_minute

            if lunch_start_minutes <= time_minutes <= lunch_end_minutes:
                in_lunch = True

            # Calculate VWAP deviation
            vwap_deviation_pct = ((close - current_vwap) / current_vwap) * 100

            # Volume check
            vol_ratio = volume / current_vol_sma if current_vol_sma > 0 else 0
            volume_ok = not config.require_low_volume or vol_ratio <= config.volume_below_threshold

            # RSI check for confirmation
            rsi_confirms_call = not config.require_oversold_overbought or current_rsi <= config.rsi_oversold_threshold
            rsi_confirms_put = not config.require_oversold_overbought or current_rsi >= config.rsi_overbought_threshold

            # Generate signal during lunch
            if in_lunch and trades_today < config.max_trades_per_session and volume_ok:
                # Price below VWAP - expect mean reversion UP
                if vwap_deviation_pct <= -config.vwap_deviation_pct and rsi_confirms_call:
                    signal_type = "BUY_CALL"
                    reason = f"Lunch scalp long: price ${close:.2f} below VWAP ${current_vwap:.2f} ({vwap_deviation_pct:.2f}%)"
                    trades_today += 1
                # Price above VWAP - expect mean reversion DOWN
                elif vwap_deviation_pct >= config.vwap_deviation_pct and rsi_confirms_put:
                    signal_type = "BUY_PUT"
                    reason = f"Lunch scalp short: price ${close:.2f} above VWAP ${current_vwap:.2f} ({vwap_deviation_pct:.2f}%)"
                    trades_today += 1

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=ts,
                    price=close,
                    reason=reason,
                    metadata={
                        "vwap": current_vwap,
                        "vwap_deviation_pct": vwap_deviation_pct,
                        "volume_ratio": vol_ratio,
                        "rsi": current_rsi,
                        "in_lunch": in_lunch,
                        "trades_today": trades_today,
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
