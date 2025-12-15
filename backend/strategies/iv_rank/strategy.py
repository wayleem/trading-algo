"""
IV Rank Strategy - filters RSI signals by IV percentile.

This strategy wraps RSI mean reversion but only enters when IV rank is elevated,
meaning option premium is rich relative to historical levels.
"""

import logging
from datetime import date, timedelta
from typing import List, Optional

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.base.filter import IVRankFilter
from strategies.iv_rank.config import IVRankConfig
from strategies.iv_rank.calculator import IVRankCalculator, build_iv_history_from_theta

logger = logging.getLogger(__name__)


class IVRankStrategy(BaseStrategy):
    """
    IV Rank Filter Strategy.

    Uses RSI mean reversion as the base signal generator, but only enters
    when IV rank is elevated (premium is rich).

    Rationale:
    - High IV = options are expensive = better premium collection
    - High IV often means volatile market = more mean reversion opportunities
    - Low IV = range-bound, small moves, theta decay dominates

    The strategy filters RSI signals through an IV rank threshold:
    - IV Rank >= threshold: Allow entry (premium is rich)
    - IV Rank < threshold: Block entry (premium is cheap)
    """

    def __init__(self):
        """Initialize IV Rank strategy."""
        self._iv_calculator: Optional[IVRankCalculator] = None
        self._iv_history_built = False

    @property
    def name(self) -> str:
        return "iv_rank"

    @property
    def description(self) -> str:
        return "RSI mean reversion filtered by IV rank threshold"

    def get_default_config(self) -> IVRankConfig:
        """Return default IV Rank strategy configuration."""
        return IVRankConfig(
            # IV settings
            min_iv_rank=50.0,
            iv_rank_lookback_days=45,
            # RSI settings (aggressive thresholds)
            rsi_period=14,
            rsi_sma_period=10,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            # Exit settings
            profit_target_pct=0.20,
            stop_loss_pct=0.25,
        )

    def get_entry_filters(self) -> List:
        """Return IV Rank filter."""
        config = self.get_default_config()
        return [
            IVRankFilter(
                min_iv_rank=config.min_iv_rank,
                max_iv_rank=config.max_iv_rank,
                require_valid=config.iv_rank_require_valid,
            )
        ]

    async def generate_signals(
        self,
        bars: List[dict],
        config: IVRankConfig,
    ) -> List[TradingSignal]:
        """
        Generate RSI signals with IV rank metadata.

        The actual IV rank filtering is done by the IVRankFilter in get_entry_filters().
        This method generates RSI signals and attaches IV rank to metadata.

        Args:
            bars: OHLCV bars
            config: IVRankConfig instance

        Returns:
            List of TradingSignal with iv_rank in metadata
        """
        from app.services.indicators import calculate_rsi_series, calculate_sma_series

        if not bars:
            return []

        # Extract data
        closes = [bar["close"] for bar in bars]
        timestamps = [bar["timestamp"] for bar in bars]

        # Calculate RSI and SMA
        rsi_values = calculate_rsi_series(closes, config.rsi_period)
        sma_values = calculate_sma_series(rsi_values, config.rsi_sma_period)

        # Initialize IV calculator if needed
        if self._iv_calculator is None:
            self._iv_calculator = IVRankCalculator(
                symbol=config.symbol,
                lookback_days=config.iv_rank_lookback_days,
                min_history_days=config.iv_rank_min_history,
            )

        # Generate signals with RSI crossover logic
        signals = []
        prev_rsi = None
        prev_sma = None

        for i, (rsi, sma, close, timestamp) in enumerate(
            zip(rsi_values, sma_values, closes, timestamps)
        ):
            signal_type = "NO_SIGNAL"
            reason = ""

            if prev_rsi is not None and prev_sma is not None:
                # Check for RSI crossing above SMA while oversold
                if rsi > sma and prev_rsi <= prev_sma and rsi < config.rsi_oversold:
                    signal_type = "BUY_CALL"
                    reason = f"RSI {rsi:.1f} crossed above SMA {sma:.1f} (oversold)"

                # Check for RSI crossing below SMA while overbought
                elif rsi < sma and prev_rsi >= prev_sma and rsi > config.rsi_overbought:
                    signal_type = "BUY_PUT"
                    reason = f"RSI {rsi:.1f} crossed below SMA {sma:.1f} (overbought)"

            # Get IV rank for this bar (will be used by filter)
            bar_date = timestamp.date() if hasattr(timestamp, "date") else timestamp
            iv_rank_result = self._iv_calculator.calculate_iv_rank(
                current_iv=0.20,  # Default IV if not available
                as_of_date=bar_date,
            )

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=timestamp,
                    price=close,
                    reason=reason,
                    metadata={
                        "rsi": rsi,
                        "rsi_sma": sma,
                        "iv_rank": iv_rank_result.iv_rank if iv_rank_result.is_valid else None,
                        "iv_rank_valid": iv_rank_result.is_valid,
                    },
                )
            )

            prev_rsi = rsi
            prev_sma = sma

        return signals

    async def build_iv_history(
        self,
        theta_client,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> int:
        """
        Build IV history from ThetaData.

        Should be called before running backtest to populate IV history.

        Args:
            theta_client: ThetaDataClient instance
            symbol: Symbol to fetch IV for
            start_date: Start of backtest (will fetch history before this)
            end_date: End of backtest

        Returns:
            Number of IV observations added
        """
        if self._iv_calculator is None:
            config = self.get_default_config()
            self._iv_calculator = IVRankCalculator(
                symbol=symbol,
                lookback_days=config.iv_rank_lookback_days,
                min_history_days=config.iv_rank_min_history,
            )

        # Fetch history starting before start_date
        history_start = start_date - timedelta(days=90)

        days_added = await build_iv_history_from_theta(
            calculator=self._iv_calculator,
            theta_client=theta_client,
            symbol=symbol,
            start_date=history_start,
            end_date=end_date,
        )

        self._iv_history_built = True
        return days_added

    def set_iv_calculator(self, calculator: IVRankCalculator) -> None:
        """
        Set IV calculator (for testing or external initialization).

        Args:
            calculator: Pre-configured IVRankCalculator
        """
        self._iv_calculator = calculator
        self._iv_history_built = True
