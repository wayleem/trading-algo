"""
IV vs RV (Implied vs Realized Volatility) Strategy.

Trades based on disparity between implied and realized volatility.
"""

import logging
import numpy as np
from typing import List, Optional
from datetime import datetime

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.iv_rv.config import IvRvConfig

logger = logging.getLogger(__name__)


class IvRvStrategy(BaseStrategy):
    """
    IV vs RV Strategy.

    Trades based on implied vs realized volatility disparity.

    Rationale:
    - IV often overestimates actual volatility (volatility risk premium)
    - When IV >> RV: options overpriced, fade moves
    - When IV << RV: options underpriced, trade momentum
    - Exploits vol mispricing for edge

    Strategy Rules:
    1. Calculate realized volatility from price data
    2. Compare to implied volatility (from option prices)
    3. When IV > RV: trade mean reversion
    4. When IV < RV: trade momentum
    """

    def __init__(self):
        """Initialize IV vs RV strategy."""
        pass

    @property
    def name(self) -> str:
        return "iv_rv"

    @property
    def description(self) -> str:
        return "IV vs RV - trades based on implied vs realized volatility disparity"

    def get_default_config(self) -> IvRvConfig:
        """Return default configuration."""
        return IvRvConfig(
            rv_lookback=50,
            iv_premium_threshold=1.3,
            iv_discount_threshold=0.8,
            profit_target_pct=0.20,
            stop_loss_pct=0.15,
        )

    def get_entry_filters(self) -> List:
        """No additional filters."""
        return []

    def _calculate_realized_volatility(
        self,
        closes: List[float],
        lookback: int,
        annualization: float,
    ) -> List[float]:
        """
        Calculate annualized realized volatility.

        Uses standard deviation of log returns.
        """
        rv_values = []

        for i in range(len(closes)):
            if i < lookback:
                rv_values.append(0.0)
                continue

            # Calculate log returns
            returns = []
            for j in range(i - lookback + 1, i + 1):
                if closes[j - 1] > 0:
                    log_return = np.log(closes[j] / closes[j - 1])
                    returns.append(log_return)

            if len(returns) >= 2:
                # Annualized volatility
                std_dev = np.std(returns)
                # Annualize: multiply by sqrt(periods per year)
                # For 5-min bars: ~78 bars per day * 252 days
                bars_per_year = 78 * annualization
                rv = std_dev * np.sqrt(bars_per_year)
                rv_values.append(rv)
            else:
                rv_values.append(0.0)

        return rv_values

    async def generate_signals(
        self,
        bars: List[dict],
        config: IvRvConfig,
    ) -> List[TradingSignal]:
        """
        Generate IV vs RV signals.

        Args:
            bars: OHLCV bars
            config: IvRvConfig instance

        Returns:
            List of TradingSignal for each bar
        """
        if not bars:
            return []

        # Extract data
        closes = [bar["close"] for bar in bars]
        timestamps = [bar["timestamp"] for bar in bars]

        # Calculate realized volatility
        rv_values = self._calculate_realized_volatility(
            closes, config.rv_lookback, config.annualization_factor
        )

        signals = []
        current_date = None
        day_closes = []
        trades_today = 0
        bars_since_open = 0

        # Estimate IV from intraday price range (proxy when real IV not available)
        # This is a simplification - real implementation would use ThetaData Greeks
        estimated_iv = 0.20  # Default 20% IV estimate

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
            rv = rv_values[i]

            # Estimate intraday IV from price range as proxy
            if len(day_closes) >= 20:
                recent_returns = []
                for j in range(max(0, len(day_closes) - 20), len(day_closes)):
                    if j > 0 and day_closes[j - 1] > 0:
                        ret = np.log(day_closes[j] / day_closes[j - 1])
                        recent_returns.append(ret)
                if recent_returns:
                    intraday_std = np.std(recent_returns)
                    estimated_iv = intraday_std * np.sqrt(78 * 252)

            # Calculate IV/RV ratio
            iv_rv_ratio = estimated_iv / rv if rv > 0 else 1.0

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

            # Generate signals based on IV/RV regime
            if can_trade and not past_cutoff and trades_today < config.max_trades_per_day:
                if config.require_momentum and momentum_direction is None:
                    pass  # No momentum, no trade
                elif iv_rv_ratio >= config.iv_premium_threshold:
                    # IV overpriced - expect mean reversion, fade the move
                    if momentum_direction == "bullish":
                        signal_type = "BUY_PUT"
                        reason = f"IV/RV {iv_rv_ratio:.2f} (overpriced): fading bullish move"
                        trades_today += 1
                    elif momentum_direction == "bearish":
                        signal_type = "BUY_CALL"
                        reason = f"IV/RV {iv_rv_ratio:.2f} (overpriced): fading bearish move"
                        trades_today += 1
                elif iv_rv_ratio <= config.iv_discount_threshold:
                    # IV underpriced - trade with momentum
                    if momentum_direction == "bullish":
                        signal_type = "BUY_CALL"
                        reason = f"IV/RV {iv_rv_ratio:.2f} (underpriced): riding bullish momentum"
                        trades_today += 1
                    elif momentum_direction == "bearish":
                        signal_type = "BUY_PUT"
                        reason = f"IV/RV {iv_rv_ratio:.2f} (underpriced): riding bearish momentum"
                        trades_today += 1

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=ts,
                    price=close,
                    reason=reason,
                    metadata={
                        "realized_volatility": rv,
                        "estimated_iv": estimated_iv,
                        "iv_rv_ratio": iv_rv_ratio,
                        "momentum_pct": momentum_pct,
                        "momentum_direction": momentum_direction,
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
