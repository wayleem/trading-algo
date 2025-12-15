"""
Volatility Regime Strategy.

Adapts trading parameters based on current market volatility.
"""

import logging
import numpy as np
from typing import List, Optional
from datetime import datetime

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.vix_regime.config import VixRegimeConfig

logger = logging.getLogger(__name__)


class VixRegimeStrategy(BaseStrategy):
    """
    Volatility Regime Strategy.

    Adapts trading behavior based on realized volatility.

    Rationale:
    - High volatility requires wider stops and targets
    - Low volatility allows tighter risk management
    - Different momentum thresholds for different regimes
    - Adapts to changing market conditions

    Strategy Rules:
    1. Calculate realized volatility (ATR as % of price)
    2. Classify regime: high, normal, or low volatility
    3. Apply regime-specific parameters
    4. Trade momentum with adaptive settings
    """

    def __init__(self):
        """Initialize Volatility Regime strategy."""
        pass

    @property
    def name(self) -> str:
        return "vix_regime"

    @property
    def description(self) -> str:
        return "Volatility regime adaptive - adjusts parameters based on market volatility"

    def get_default_config(self) -> VixRegimeConfig:
        """Return default configuration."""
        return VixRegimeConfig(
            volatility_lookback=50,
            high_vol_threshold=0.15,
            low_vol_threshold=0.08,
            high_vol_profit_target=0.30,
            high_vol_stop_loss=0.20,
            low_vol_profit_target=0.12,
            low_vol_stop_loss=0.10,
        )

    def get_entry_filters(self) -> List:
        """No additional filters."""
        return []

    def _calculate_atr_pct(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int,
    ) -> List[float]:
        """Calculate ATR as percentage of price."""
        atr_pct_values = []

        for i in range(len(highs)):
            if i < period:
                # Not enough data yet
                if i == 0:
                    atr_pct_values.append(0.0)
                else:
                    tr = max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]),
                    )
                    atr_pct = (tr / closes[i]) * 100 if closes[i] > 0 else 0
                    atr_pct_values.append(atr_pct)
            else:
                # Calculate ATR for the period
                tr_values = []
                for j in range(i - period + 1, i + 1):
                    if j == 0:
                        tr = highs[j] - lows[j]
                    else:
                        tr = max(
                            highs[j] - lows[j],
                            abs(highs[j] - closes[j - 1]),
                            abs(lows[j] - closes[j - 1]),
                        )
                    tr_values.append(tr)

                atr = np.mean(tr_values)
                atr_pct = (atr / closes[i]) * 100 if closes[i] > 0 else 0
                atr_pct_values.append(atr_pct)

        return atr_pct_values

    def _classify_regime(
        self,
        atr_pct: float,
        config: VixRegimeConfig,
    ) -> str:
        """Classify volatility regime."""
        if atr_pct >= config.high_vol_threshold:
            return "high"
        elif atr_pct <= config.low_vol_threshold:
            return "low"
        return "normal"

    async def generate_signals(
        self,
        bars: List[dict],
        config: VixRegimeConfig,
    ) -> List[TradingSignal]:
        """
        Generate volatility regime adaptive signals.

        Args:
            bars: OHLCV bars
            config: VixRegimeConfig instance

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

        # Calculate ATR% for all bars
        atr_pct_values = self._calculate_atr_pct(highs, lows, closes, config.volatility_lookback)

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

            # Reset on new day
            if current_date != bar_date:
                current_date = bar_date
                day_closes = []
                trades_today = 0
                bars_since_open = 0

            day_closes.append(closes[i])
            bars_since_open += 1

            close = closes[i]
            atr_pct = atr_pct_values[i]

            # Classify regime
            regime = self._classify_regime(atr_pct, config)

            # Get regime-specific parameters
            if regime == "high":
                momentum_threshold = config.high_vol_momentum_threshold
            elif regime == "low":
                momentum_threshold = config.low_vol_momentum_threshold
            else:
                momentum_threshold = config.normal_vol_momentum_threshold

            # Wait for entry start time
            can_trade = bars_since_open >= config.entry_start_minutes // 5  # Assuming 5-min bars

            # Calculate momentum
            momentum_direction = None
            momentum_pct = 0.0
            lookback = 10  # Short-term momentum

            if len(day_closes) >= lookback:
                lookback_close = day_closes[-lookback]
                momentum_pct = ((close - lookback_close) / lookback_close) * 100

                if momentum_pct >= momentum_threshold:
                    momentum_direction = "bullish"
                elif momentum_pct <= -momentum_threshold:
                    momentum_direction = "bearish"

            # Generate signal
            if can_trade and trades_today < config.max_trades_per_day and momentum_direction:
                if momentum_direction == "bullish":
                    signal_type = "BUY_CALL"
                    reason = f"VIX regime {regime}: bullish momentum {momentum_pct:.2f}%, ATR {atr_pct:.2f}%"
                    trades_today += 1
                elif momentum_direction == "bearish":
                    signal_type = "BUY_PUT"
                    reason = f"VIX regime {regime}: bearish momentum {momentum_pct:.2f}%, ATR {atr_pct:.2f}%"
                    trades_today += 1

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=ts,
                    price=close,
                    reason=reason,
                    metadata={
                        "atr_pct": atr_pct,
                        "regime": regime,
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
