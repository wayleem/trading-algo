"""
Support/Resistance Bounce Strategy.

Trades bounces off key support and resistance levels.
"""

import logging
from typing import List, Optional
from datetime import datetime

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.support_resistance.config import SupportResistanceConfig
from app.services.support_resistance import SupportResistanceAnalyzer, PriceLevel

logger = logging.getLogger(__name__)


class SupportResistanceStrategy(BaseStrategy):
    """
    Support/Resistance Bounce Strategy.

    Trades bounces off key S/R levels.

    Rationale:
    - Key S/R levels act as magnets for price
    - Bounces off strong levels offer favorable R:R
    - Round numbers have psychological significance
    - Multi-touch levels are more reliable

    Strategy Rules:
    1. Detect key S/R levels from price history
    2. Wait for price to approach S/R level
    3. Enter on bounce confirmation
    4. Target move away from level
    """

    def __init__(self):
        """Initialize S/R strategy."""
        self.sr_analyzer = None

    @property
    def name(self) -> str:
        return "support_resistance"

    @property
    def description(self) -> str:
        return "Support/Resistance bounce - trades bounces off key price levels"

    def get_default_config(self) -> SupportResistanceConfig:
        """Return default configuration."""
        return SupportResistanceConfig(
            sr_lookback_bars=100,
            tolerance_pct=0.3,
            min_level_strength=2,
            entry_distance_pct=0.15,
            profit_target_pct=0.20,
            stop_loss_pct=0.15,
        )

    def get_entry_filters(self) -> List:
        """No additional filters."""
        return []

    def _init_analyzer(self, config: SupportResistanceConfig):
        """Initialize S/R analyzer with config."""
        self.sr_analyzer = SupportResistanceAnalyzer(
            lookback_bars=config.sr_lookback_bars,
            tolerance_pct=config.tolerance_pct,
            min_touches=config.min_level_strength,
            round_number_interval=config.round_number_interval,
        )

    async def generate_signals(
        self,
        bars: List[dict],
        config: SupportResistanceConfig,
    ) -> List[TradingSignal]:
        """
        Generate S/R bounce signals.

        Args:
            bars: OHLCV bars
            config: SupportResistanceConfig instance

        Returns:
            List of TradingSignal for each bar
        """
        if not bars:
            return []

        # Initialize analyzer
        self._init_analyzer(config)

        # Extract data
        closes = [bar["close"] for bar in bars]
        highs = [bar["high"] for bar in bars]
        lows = [bar["low"] for bar in bars]
        timestamps = [bar["timestamp"] for bar in bars]

        signals = []
        current_date = None
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
                trades_today = 0
                bars_since_open = 0

            bars_since_open += 1
            close = closes[i]
            high = highs[i]
            low = lows[i]

            # Check entry timing
            can_trade = bars_since_open >= config.entry_start_minutes // 5  # Assuming 5-min bars
            past_cutoff = bar_hour >= config.entry_cutoff_hour_utc

            # Find S/R levels using bars up to current point
            if can_trade and not past_cutoff and trades_today < config.max_trades_per_day:
                # Use historical bars up to current point
                historical_bars = bars[max(0, i - config.sr_lookback_bars) : i + 1]

                if len(historical_bars) >= 20:  # Need minimum bars
                    support_levels, resistance_levels = self.sr_analyzer.find_levels(historical_bars)

                    # Check for bounce off support
                    nearest_support = None
                    for level in support_levels:
                        if level.price < close:
                            if nearest_support is None or level.price > nearest_support.price:
                                nearest_support = level

                    # Check for bounce off resistance
                    nearest_resistance = None
                    for level in resistance_levels:
                        if level.price > close:
                            if nearest_resistance is None or level.price < nearest_resistance.price:
                                nearest_resistance = level

                    # Check support bounce
                    if nearest_support:
                        distance_pct = ((close - nearest_support.price) / nearest_support.price) * 100
                        if distance_pct <= config.entry_distance_pct:
                            # Price near support, check for bounce confirmation
                            bounce_confirmed = True
                            if config.require_momentum_confirmation and i >= config.momentum_confirm_bars:
                                # Check if price is moving up from support
                                prev_close = closes[i - config.momentum_confirm_bars]
                                bounce_confirmed = close > prev_close and low <= nearest_support.price * (1 + config.entry_distance_pct / 100)

                            if bounce_confirmed:
                                signal_type = "BUY_CALL"
                                reason = f"Support bounce: ${close:.2f} near ${nearest_support.price:.2f} (strength {nearest_support.strength})"
                                trades_today += 1

                    # Check resistance bounce (only if no support signal)
                    if signal_type == "NO_SIGNAL" and nearest_resistance:
                        distance_pct = ((nearest_resistance.price - close) / nearest_resistance.price) * 100
                        if distance_pct <= config.entry_distance_pct:
                            # Price near resistance, check for rejection confirmation
                            rejection_confirmed = True
                            if config.require_momentum_confirmation and i >= config.momentum_confirm_bars:
                                # Check if price is moving down from resistance
                                prev_close = closes[i - config.momentum_confirm_bars]
                                rejection_confirmed = close < prev_close and high >= nearest_resistance.price * (1 - config.entry_distance_pct / 100)

                            if rejection_confirmed:
                                signal_type = "BUY_PUT"
                                reason = f"Resistance rejection: ${close:.2f} near ${nearest_resistance.price:.2f} (strength {nearest_resistance.strength})"
                                trades_today += 1

            # Get current levels for metadata
            nearest_support_price = None
            nearest_resistance_price = None

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=ts,
                    price=close,
                    reason=reason,
                    metadata={
                        "nearest_support": nearest_support_price,
                        "nearest_resistance": nearest_resistance_price,
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
