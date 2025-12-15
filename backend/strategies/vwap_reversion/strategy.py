"""
VWAP Reversion Strategy.

Trades mean reversion to VWAP when price deviates significantly.
"""

import logging
import numpy as np
from typing import List, Optional
from datetime import datetime

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.vwap_reversion.config import VWAPReversionConfig

logger = logging.getLogger(__name__)


class VWAPReversionStrategy(BaseStrategy):
    """
    VWAP Mean Reversion Strategy.

    Trades when price deviates significantly from VWAP, expecting reversion.

    Rationale:
    - VWAP represents institutional fair value for the day
    - Large deviations often revert as traders take profits
    - 0DTE options benefit from quick reversion moves

    Strategy Rules:
    1. Calculate running VWAP from market open
    2. When price deviates > threshold from VWAP, enter against deviation
    3. Price ABOVE VWAP -> BUY PUT (expect reversion down)
    4. Price BELOW VWAP -> BUY CALL (expect reversion up)
    5. Exit when price returns to VWAP or hits PT/SL
    """

    def __init__(self):
        """Initialize VWAP Reversion strategy."""
        pass

    @property
    def name(self) -> str:
        return "vwap_reversion"

    @property
    def description(self) -> str:
        return "Mean reversion to VWAP on significant deviations"

    def get_default_config(self) -> VWAPReversionConfig:
        """Return default VWAP Reversion configuration."""
        return VWAPReversionConfig(
            # VWAP settings
            vwap_deviation_pct=0.3,
            vwap_bands_std=2.0,
            # Exit settings
            profit_target_pct=0.15,
            stop_loss_pct=0.20,
            exit_at_vwap=True,
            # Entry timing
            entry_start_minutes=30,
            entry_cutoff_hour_utc=19,
        )

    def get_entry_filters(self) -> List:
        """VWAP deviation defines entry - no additional filters."""
        return []

    def _calculate_vwap(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
    ) -> List[float]:
        """
        Calculate cumulative VWAP.

        VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
        Typical Price = (High + Low + Close) / 3

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            volumes: Volume values

        Returns:
            List of VWAP values
        """
        vwap_values = []
        cumulative_tpv = 0.0  # Typical Price * Volume
        cumulative_vol = 0.0

        for h, l, c, v in zip(highs, lows, closes, volumes):
            typical_price = (h + l + c) / 3
            cumulative_tpv += typical_price * v
            cumulative_vol += v

            if cumulative_vol > 0:
                vwap = cumulative_tpv / cumulative_vol
            else:
                vwap = c  # Use close if no volume

            vwap_values.append(vwap)

        return vwap_values

    def _calculate_vwap_bands(
        self,
        closes: List[float],
        vwap_values: List[float],
        std_multiplier: float = 2.0,
        window: int = 20,
    ) -> tuple:
        """
        Calculate VWAP bands based on rolling standard deviation.

        Args:
            closes: Close prices
            vwap_values: VWAP values
            std_multiplier: Standard deviation multiplier
            window: Rolling window for std calculation

        Returns:
            Tuple of (upper_band, lower_band) lists
        """
        upper_band = []
        lower_band = []

        for i in range(len(closes)):
            if i < window - 1:
                # Not enough data - use simple deviation
                dev = abs(closes[i] - vwap_values[i])
                upper_band.append(vwap_values[i] + dev * std_multiplier)
                lower_band.append(vwap_values[i] - dev * std_multiplier)
            else:
                # Calculate rolling std of price-vwap deviations
                deviations = [
                    closes[j] - vwap_values[j]
                    for j in range(i - window + 1, i + 1)
                ]
                std = np.std(deviations)
                upper_band.append(vwap_values[i] + std * std_multiplier)
                lower_band.append(vwap_values[i] - std * std_multiplier)

        return upper_band, lower_band

    def _calculate_volume_sma(
        self,
        volumes: List[float],
        period: int,
    ) -> List[float]:
        """
        Calculate Simple Moving Average of volume.

        Args:
            volumes: Volume values
            period: SMA period

        Returns:
            List of volume SMA values
        """
        sma_values = []
        for i in range(len(volumes)):
            if i < period - 1:
                sma_values.append(np.mean(volumes[: i + 1]))
            else:
                sma_values.append(np.mean(volumes[i - period + 1 : i + 1]))
        return sma_values

    async def generate_signals(
        self,
        bars: List[dict],
        config: VWAPReversionConfig,
    ) -> List[TradingSignal]:
        """
        Generate VWAP reversion signals from price bars.

        Args:
            bars: OHLCV bars
            config: VWAPReversionConfig instance

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

        # Calculate VWAP (resets daily)
        vwap_values = []
        current_date = None
        day_start_idx = 0

        for i, ts in enumerate(timestamps):
            bar_date = ts.date() if hasattr(ts, "date") else ts
            if current_date != bar_date:
                # New day - reset VWAP calculation
                current_date = bar_date
                day_start_idx = i

            # Calculate VWAP from day start
            day_highs = highs[day_start_idx : i + 1]
            day_lows = lows[day_start_idx : i + 1]
            day_closes = closes[day_start_idx : i + 1]
            day_volumes = volumes[day_start_idx : i + 1]

            day_vwap = self._calculate_vwap(day_highs, day_lows, day_closes, day_volumes)
            vwap_values.append(day_vwap[-1] if day_vwap else closes[i])

        # Calculate VWAP bands if enabled
        if config.use_vwap_bands:
            upper_band, lower_band = self._calculate_vwap_bands(
                closes, vwap_values, config.vwap_bands_std
            )
        else:
            # Use simple percentage bands
            upper_band = [v * (1 + config.vwap_deviation_pct / 100) for v in vwap_values]
            lower_band = [v * (1 - config.vwap_deviation_pct / 100) for v in vwap_values]

        # Calculate volume SMA for confirmation
        volume_sma = self._calculate_volume_sma(volumes, config.volume_sma_period)

        # Generate signals
        signals = []
        in_position = False
        entry_direction = None

        for i in range(len(bars)):
            signal_type = "NO_SIGNAL"
            reason = ""

            ts = timestamps[i]
            bar_hour = ts.hour if hasattr(ts, "hour") else 0
            bar_minute = ts.minute if hasattr(ts, "minute") else 0

            # Check entry timing
            minutes_from_open = (bar_hour - 14) * 60 + bar_minute  # 14:30 UTC = 9:30 ET
            past_entry_start = minutes_from_open >= config.entry_start_minutes
            before_cutoff = bar_hour < config.entry_cutoff_hour_utc

            close = closes[i]
            vwap = vwap_values[i]
            deviation_pct = ((close - vwap) / vwap) * 100 if vwap > 0 else 0

            # Volume confirmation
            vol_confirmed = True
            if config.require_volume_confirmation and volume_sma[i] > 0:
                vol_ratio = volumes[i] / volume_sma[i]
                vol_confirmed = vol_ratio < config.volume_threshold

            # Check for entry signals
            if not in_position and past_entry_start and before_cutoff:
                # Price above upper band -> BUY PUT (expect reversion down)
                if close > upper_band[i] and vol_confirmed:
                    if abs(deviation_pct) <= config.max_deviation_pct:
                        signal_type = "BUY_PUT"
                        reason = f"Price ${close:.2f} above VWAP band ${upper_band[i]:.2f} (dev: {deviation_pct:.2f}%)"
                        in_position = True
                        entry_direction = "short"

                # Price below lower band -> BUY CALL (expect reversion up)
                elif close < lower_band[i] and vol_confirmed:
                    if abs(deviation_pct) <= config.max_deviation_pct:
                        signal_type = "BUY_CALL"
                        reason = f"Price ${close:.2f} below VWAP band ${lower_band[i]:.2f} (dev: {deviation_pct:.2f}%)"
                        in_position = True
                        entry_direction = "long"

            # Track exit at VWAP (for reference - actual exit handled by runner)
            if in_position and config.exit_at_vwap:
                if entry_direction == "long" and close >= vwap:
                    in_position = False
                    entry_direction = None
                elif entry_direction == "short" and close <= vwap:
                    in_position = False
                    entry_direction = None

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=ts,
                    price=close,
                    reason=reason,
                    metadata={
                        "vwap": vwap,
                        "upper_band": upper_band[i],
                        "lower_band": lower_band[i],
                        "deviation_pct": deviation_pct,
                        "volume_ratio": volumes[i] / volume_sma[i] if volume_sma[i] > 0 else 0,
                    },
                )
            )

        return signals

    def get_option_direction(self, signal: TradingSignal) -> str:
        """
        Determine option type based on reversion direction.

        Args:
            signal: TradingSignal from generate_signals

        Returns:
            'call' for upside reversion, 'put' for downside reversion
        """
        if signal.signal_type == "BUY_CALL":
            return "call"
        elif signal.signal_type == "BUY_PUT":
            return "put"
        return "call"
