"""
Momentum Scalping Strategy.

Trades strong intraday momentum moves with tight profit targets.
"""

import logging
import numpy as np
from typing import List, Optional
from datetime import datetime

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.momentum_scalp.config import MomentumScalpConfig

logger = logging.getLogger(__name__)


class MomentumScalpStrategy(BaseStrategy):
    """
    Momentum Scalping Strategy.

    Trades strong directional moves with volume confirmation.

    Rationale:
    - Strong momentum often continues short-term (inertia)
    - Volume surge confirms institutional participation
    - Quick exits capture momentum without reversal risk

    Strategy Rules:
    1. Measure price momentum over N bars
    2. Require volume surge above SMA threshold
    3. Enter in direction of momentum (calls for up, puts for down)
    4. Exit quickly with tight profit target
    5. Trailing stop to protect gains
    """

    def __init__(self):
        """Initialize Momentum Scalp strategy."""
        pass

    @property
    def name(self) -> str:
        return "momentum_scalp"

    @property
    def description(self) -> str:
        return "Momentum scalping with volume confirmation and tight targets"

    def get_default_config(self) -> MomentumScalpConfig:
        """Return default Momentum Scalp configuration."""
        return MomentumScalpConfig(
            # Momentum settings
            momentum_lookback=5,
            momentum_threshold=0.15,
            use_acceleration=True,
            # Volume settings
            require_volume_surge=True,
            volume_surge_threshold=1.5,
            # Exit settings
            profit_target_pct=0.10,
            stop_loss_pct=0.08,
            trailing_stop_pct=0.05,
            # Entry timing
            entry_start_minutes=5,
            entry_cutoff_hour_utc=19,
        )

    def get_entry_filters(self) -> List:
        """Momentum and volume define entry - no additional filters."""
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

    def _calculate_momentum(
        self,
        closes: List[float],
        lookback: int,
    ) -> List[float]:
        """
        Calculate momentum as percentage change over lookback.

        Args:
            closes: Close prices
            lookback: Number of bars to measure momentum

        Returns:
            List of momentum percentages
        """
        momentum = []
        for i in range(len(closes)):
            if i < lookback:
                momentum.append(0.0)
            else:
                prev_close = closes[i - lookback]
                if prev_close > 0:
                    pct_change = ((closes[i] - prev_close) / prev_close) * 100
                    momentum.append(pct_change)
                else:
                    momentum.append(0.0)
        return momentum

    def _calculate_acceleration(
        self,
        momentum: List[float],
        lookback: int = 3,
    ) -> List[float]:
        """
        Calculate momentum acceleration (change in momentum).

        Args:
            momentum: Momentum values
            lookback: Bars to measure acceleration

        Returns:
            List of acceleration values
        """
        acceleration = []
        for i in range(len(momentum)):
            if i < lookback:
                acceleration.append(0.0)
            else:
                accel = momentum[i] - momentum[i - lookback]
                acceleration.append(accel)
        return acceleration

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
                sma_values.append(np.mean(volumes[: i + 1]) if volumes[: i + 1] else 0)
            else:
                sma_values.append(np.mean(volumes[i - period + 1 : i + 1]))
        return sma_values

    async def generate_signals(
        self,
        bars: List[dict],
        config: MomentumScalpConfig,
    ) -> List[TradingSignal]:
        """
        Generate momentum scalping signals from price bars.

        Args:
            bars: OHLCV bars
            config: MomentumScalpConfig instance

        Returns:
            List of TradingSignal for each bar
        """
        if not bars:
            return []

        # Extract data
        closes = [bar["close"] for bar in bars]
        volumes = [bar.get("volume", 0) for bar in bars]
        timestamps = [bar["timestamp"] for bar in bars]

        # Calculate indicators
        momentum = self._calculate_momentum(closes, config.momentum_lookback)
        acceleration = self._calculate_acceleration(momentum) if config.use_acceleration else [0] * len(closes)
        volume_sma = self._calculate_volume_sma(volumes, config.volume_sma_period)
        ema = self._calculate_ema(closes, config.ema_period) if config.use_ema_filter else closes

        # Generate signals
        signals = []
        cooldown = 0  # Bars to wait after a signal

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
            mom = momentum[i]
            accel = acceleration[i]
            vol_ratio = volumes[i] / volume_sma[i] if volume_sma[i] > 0 else 0

            # Volume confirmation
            vol_confirmed = not config.require_volume_surge or vol_ratio >= config.volume_surge_threshold

            # Acceleration confirmation
            accel_confirmed = not config.use_acceleration or (
                (mom > 0 and accel > 0) or (mom < 0 and accel < 0)
            )

            # EMA trend confirmation
            ema_confirmed = True
            if config.use_ema_filter:
                if mom > 0:  # Bullish momentum
                    ema_confirmed = close > ema[i]
                else:  # Bearish momentum
                    ema_confirmed = close < ema[i]

            # Check for entry signals
            if cooldown <= 0 and past_entry_start and before_cutoff:
                # Bullish momentum -> BUY CALL
                if mom >= config.momentum_threshold and vol_confirmed and accel_confirmed and ema_confirmed:
                    signal_type = "BUY_CALL"
                    reason = f"Bullish momentum {mom:.2f}% with volume surge {vol_ratio:.1f}x"
                    cooldown = config.momentum_lookback  # Avoid overlapping signals

                # Bearish momentum -> BUY PUT
                elif mom <= -config.momentum_threshold and vol_confirmed and accel_confirmed and ema_confirmed:
                    signal_type = "BUY_PUT"
                    reason = f"Bearish momentum {mom:.2f}% with volume surge {vol_ratio:.1f}x"
                    cooldown = config.momentum_lookback

            cooldown = max(0, cooldown - 1)

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=ts,
                    price=close,
                    reason=reason,
                    metadata={
                        "momentum": mom,
                        "acceleration": accel,
                        "volume_ratio": vol_ratio,
                        "ema": ema[i],
                        "above_ema": close > ema[i],
                    },
                )
            )

        return signals

    def get_option_direction(self, signal: TradingSignal) -> str:
        """
        Determine option type based on momentum direction.

        Args:
            signal: TradingSignal from generate_signals

        Returns:
            'call' for bullish momentum, 'put' for bearish momentum
        """
        if signal.signal_type == "BUY_CALL":
            return "call"
        elif signal.signal_type == "BUY_PUT":
            return "put"
        return "call"
