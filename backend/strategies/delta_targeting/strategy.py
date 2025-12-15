"""
Delta Targeting Strategy.

Selects options based on target delta for consistent Greek exposure.
"""

import logging
from typing import List, Optional
from datetime import datetime

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.delta_targeting.config import DeltaTargetingConfig

logger = logging.getLogger(__name__)


class DeltaTargetingStrategy(BaseStrategy):
    """
    Delta Targeting Strategy.

    Trades momentum using delta-targeted option selection.

    Rationale:
    - Fixed delta provides consistent directional exposure
    - ~40 delta offers good balance of cost and leverage
    - Avoids cheap OTM options with low probability
    - Better risk-adjusted returns than fixed strike offset

    Strategy Rules:
    1. Identify momentum direction
    2. Select option with target delta
    3. Enter on momentum confirmation
    4. Standard PT/SL exit
    """

    def __init__(self):
        """Initialize Delta Targeting strategy."""
        pass

    @property
    def name(self) -> str:
        return "delta_targeting"

    @property
    def description(self) -> str:
        return "Delta targeting - selects options based on target delta for consistent exposure"

    def get_default_config(self) -> DeltaTargetingConfig:
        """Return default configuration."""
        return DeltaTargetingConfig(
            target_delta=0.40,
            delta_tolerance=0.10,
            momentum_threshold=0.15,
            profit_target_pct=0.20,
            stop_loss_pct=0.15,
        )

    def get_entry_filters(self) -> List:
        """No additional filters."""
        return []

    async def generate_signals(
        self,
        bars: List[dict],
        config: DeltaTargetingConfig,
    ) -> List[TradingSignal]:
        """
        Generate delta-targeted momentum signals.

        Args:
            bars: OHLCV bars
            config: DeltaTargetingConfig instance

        Returns:
            List of TradingSignal for each bar
        """
        if not bars:
            return []

        # Extract data
        closes = [bar["close"] for bar in bars]
        timestamps = [bar["timestamp"] for bar in bars]

        signals = []
        current_date = None
        day_closes = []
        trades_today = 0
        bars_since_open = 0

        for i in range(len(bars)):
            signal_type = "NO_SIGNAL"
            reason = ""

            ts = timestamps[i]
            bar_date = ts.date() if hasattr(ts, "date") else ts
            bar_hour = ts.hour if hasattr(ts, "hour") else 0

            # Reset on new day
            if current_date != bar_date:
                current_date = bar_date
                day_closes = []
                trades_today = 0
                bars_since_open = 0

            day_closes.append(closes[i])
            bars_since_open += 1

            close = closes[i]

            # Check entry timing
            can_trade = bars_since_open >= config.entry_start_minutes // 5
            past_cutoff = bar_hour >= config.entry_cutoff_hour_utc

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

            # Generate signal
            if can_trade and not past_cutoff and trades_today < config.max_trades_per_day:
                if momentum_direction == "bullish":
                    signal_type = "BUY_CALL"
                    reason = f"Delta target {config.target_delta}: bullish momentum {momentum_pct:.2f}%"
                    trades_today += 1
                elif momentum_direction == "bearish":
                    signal_type = "BUY_PUT"
                    reason = f"Delta target {config.target_delta}: bearish momentum {momentum_pct:.2f}%"
                    trades_today += 1

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=ts,
                    price=close,
                    reason=reason,
                    metadata={
                        "target_delta": config.target_delta,
                        "momentum_pct": momentum_pct,
                        "momentum_direction": momentum_direction,
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

    def get_target_delta(self, config: DeltaTargetingConfig) -> float:
        """Return target delta for option selection."""
        return config.target_delta
