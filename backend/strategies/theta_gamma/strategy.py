"""
Theta/Gamma Ratio Strategy.

Trades when theta/gamma ratio favors directional bets.
"""

import logging
import numpy as np
from typing import List, Optional
from datetime import datetime

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.theta_gamma.config import ThetaGammaConfig

logger = logging.getLogger(__name__)


class ThetaGammaStrategy(BaseStrategy):
    """
    Theta/Gamma Ratio Strategy.

    Trades when theta decay outweighs gamma risk.

    Rationale:
    - High theta/gamma ratio means good risk/reward for directional bets
    - Theta works in our favor while gamma risk is manageable
    - Best used when expecting modest directional moves
    - Avoids high-gamma scenarios where small moves hurt

    Strategy Rules:
    1. Calculate proxy theta/gamma from time and price dynamics
    2. Enter when ratio favors directional trades
    3. Trade momentum with favorable Greeks profile
    4. Quick exits to capture theta
    """

    def __init__(self):
        """Initialize Theta/Gamma strategy."""
        pass

    @property
    def name(self) -> str:
        return "theta_gamma"

    @property
    def description(self) -> str:
        return "Theta/Gamma ratio - trades when theta decay outweighs gamma risk"

    def get_default_config(self) -> ThetaGammaConfig:
        """Return default configuration."""
        return ThetaGammaConfig(
            min_theta_gamma_ratio=0.05,
            max_gamma=0.10,
            momentum_threshold=0.12,
            profit_target_pct=0.15,
            stop_loss_pct=0.12,
        )

    def get_entry_filters(self) -> List:
        """No additional filters."""
        return []

    def _estimate_theta_gamma_ratio(
        self,
        price: float,
        volatility: float,
        time_to_expiry_days: float,
    ) -> tuple:
        """
        Estimate theta/gamma characteristics from market conditions.

        This is a simplified proxy - real implementation would use ThetaData Greeks.

        Higher time decay relative to gamma risk is favorable.
        """
        # For 0DTE options:
        # - Theta accelerates as expiry approaches
        # - Gamma peaks at ATM and near expiry

        # Simplified theta estimate (negative, decay per day)
        # Theta increases (more negative) as expiry approaches
        if time_to_expiry_days > 0:
            theta_proxy = -volatility * price * 0.01 / np.sqrt(time_to_expiry_days)
        else:
            theta_proxy = -volatility * price * 0.1  # High decay on 0DTE

        # Simplified gamma estimate
        # Gamma highest at ATM and near expiry
        if time_to_expiry_days > 0:
            gamma_proxy = 1.0 / (price * volatility * np.sqrt(time_to_expiry_days))
        else:
            gamma_proxy = 1.0 / (price * volatility * 0.01)

        # Calculate ratio (use absolute theta)
        if gamma_proxy > 0:
            ratio = abs(theta_proxy) / gamma_proxy
        else:
            ratio = 0.0

        return theta_proxy, gamma_proxy, ratio

    async def generate_signals(
        self,
        bars: List[dict],
        config: ThetaGammaConfig,
    ) -> List[TradingSignal]:
        """
        Generate theta/gamma ratio signals.

        Args:
            bars: OHLCV bars
            config: ThetaGammaConfig instance

        Returns:
            List of TradingSignal for each bar
        """
        if not bars:
            return []

        # Extract data
        closes = [bar["close"] for bar in bars]
        highs = [bar["high"] for bar in bars]
        lows = [bar["low"] for bar in bars]
        timestamps = [bar["timestamp"] for bar in bars]

        signals = []
        current_date = None
        day_closes = []
        day_highs = []
        day_lows = []
        trades_today = 0
        bars_since_open = 0

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
                day_closes = []
                day_highs = []
                day_lows = []
                trades_today = 0
                bars_since_open = 0

            day_closes.append(closes[i])
            day_highs.append(highs[i])
            day_lows.append(lows[i])
            bars_since_open += 1

            close = closes[i]

            # Estimate time to expiry (in days)
            # Market closes at 4 PM ET (21:00 UTC)
            market_close_hour = 21
            minutes_to_close = max(0, (market_close_hour - bar_hour) * 60 - bar_minute)
            time_to_expiry_days = minutes_to_close / (6.5 * 60)  # Fraction of trading day

            # Estimate intraday volatility
            volatility = 0.20  # Default
            if len(day_closes) >= 20:
                returns = []
                for j in range(max(1, len(day_closes) - 20), len(day_closes)):
                    if day_closes[j - 1] > 0:
                        ret = (day_closes[j] - day_closes[j - 1]) / day_closes[j - 1]
                        returns.append(ret)
                if returns:
                    volatility = np.std(returns) * np.sqrt(78 * 252)

            # Calculate theta/gamma proxy
            theta_proxy, gamma_proxy, tg_ratio = self._estimate_theta_gamma_ratio(
                close, volatility, time_to_expiry_days
            )

            # Check entry timing
            can_trade = bars_since_open >= config.entry_start_minutes // 5
            past_cutoff = bar_hour >= config.entry_cutoff_hour_utc

            # Check Greeks conditions
            greeks_favorable = (
                tg_ratio >= config.min_theta_gamma_ratio and
                gamma_proxy <= config.max_gamma
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

            # Generate signal when Greeks are favorable
            if can_trade and not past_cutoff and trades_today < config.max_trades_per_day:
                if greeks_favorable and momentum_direction:
                    if momentum_direction == "bullish":
                        signal_type = "BUY_CALL"
                        reason = f"Theta/Gamma {tg_ratio:.3f}: favorable for bullish trade"
                        trades_today += 1
                    elif momentum_direction == "bearish":
                        signal_type = "BUY_PUT"
                        reason = f"Theta/Gamma {tg_ratio:.3f}: favorable for bearish trade"
                        trades_today += 1

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=ts,
                    price=close,
                    reason=reason,
                    metadata={
                        "theta_proxy": theta_proxy,
                        "gamma_proxy": gamma_proxy,
                        "theta_gamma_ratio": tg_ratio,
                        "time_to_expiry_days": time_to_expiry_days,
                        "momentum_pct": momentum_pct,
                        "greeks_favorable": greeks_favorable,
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
