"""
Gap Fade Strategy.

Fades overnight gaps in SPY using 0DTE options, expecting mean reversion.
"""

import logging
from typing import List, Optional

from strategies.base.strategy import BaseStrategy, TradingSignal
from strategies.gap_fade.config import GapFadeConfig
from strategies.gap_fade.signal_generator import GapFadeSignalGenerator, GapSignal

logger = logging.getLogger(__name__)


class GapFadeStrategy(BaseStrategy):
    """
    Gap Fade Strategy.

    Thesis: Overnight gaps tend to fill (partially or fully) during the trading day.
    We fade the gap direction with 0DTE options.

    Strategy Rules:
    1. Calculate overnight gap at market open (current open vs prior close)
    2. If gap is within tradeable range (0.3% - 1.5%):
       - Gap UP -> BUY PUT (expect price to fall)
       - Gap DOWN -> BUY CALL (expect price to rise)
    3. Wait 5-10 minutes after open for volatility to settle
    4. Exit on profit target, stop loss, or EOD

    Rationale:
    - Overnight gaps often overreact to news/events
    - Mean reversion is common during regular trading hours
    - Small gaps (< 0.3%) are noise, large gaps (> 1.5%) often trend
    - 0DTE options provide leveraged exposure to intraday moves

    Validation Metrics:
    - P&L: Standard profitability
    - Directional Accuracy: % of trades where gap actually faded
      (separate from P&L to validate thesis)
    """

    def __init__(self):
        """Initialize Gap Fade strategy."""
        self._signal_generator: Optional[GapFadeSignalGenerator] = None

    @property
    def name(self) -> str:
        return "gap_fade"

    @property
    def description(self) -> str:
        return "Gap Fade - fades overnight gaps expecting mean reversion"

    def get_default_config(self) -> GapFadeConfig:
        """Return default Gap Fade strategy configuration."""
        return GapFadeConfig(
            # Gap detection
            gap_threshold_pct=0.3,
            max_gap_pct=1.5,
            # Direction control
            fade_gap_up=True,
            fade_gap_down=True,
            # Entry timing
            entry_delay_minutes=5,
            entry_window_minutes=15,
            entry_cutoff_hour_utc=15,  # 10:30 AM ET
            # Position management
            max_entries_per_day=1,
            # Exit settings (from ORB optimization)
            profit_target_pct=0.30,
            stop_loss_pct=0.10,
        )

    def get_entry_filters(self) -> List:
        """Gap Fade doesn't need additional filters - gap defines entry."""
        return []

    async def generate_signals(
        self,
        bars: List[dict],
        config: GapFadeConfig,
    ) -> List[TradingSignal]:
        """
        Generate gap fade signals from price bars.

        Args:
            bars: OHLCV bars (1-minute preferred)
            config: GapFadeConfig instance

        Returns:
            List of TradingSignal for each bar
        """
        if not bars:
            return []

        # Initialize signal generator with config
        generator = GapFadeSignalGenerator(
            gap_threshold_pct=config.gap_threshold_pct,
            max_gap_pct=config.max_gap_pct,
            fade_gap_up=config.fade_gap_up,
            fade_gap_down=config.fade_gap_down,
            entry_delay_minutes=config.entry_delay_minutes,
            entry_window_minutes=config.entry_window_minutes,
            entry_cutoff_hour_utc=config.entry_cutoff_hour_utc,
            max_entries_per_day=config.max_entries_per_day,
        )

        signals = []
        for bar in bars:
            gap_signal = generator.process_bar(bar)

            # Convert gap signal to TradingSignal
            if gap_signal is not None:
                signal_type = gap_signal.signal_type.name
                reason = gap_signal.reason
                metadata = {
                    "gap_pct": gap_signal.gap_pct,
                    "prior_close": gap_signal.prior_close,
                    "current_open": gap_signal.current_open,
                }
            else:
                signal_type = "NO_SIGNAL"
                reason = ""
                metadata = {}

                # Add gap info if available
                gap_info = generator.get_gap_info()
                if gap_info:
                    metadata = {
                        "gap_pct": gap_info["gap_pct"],
                        "prior_close": gap_info["prior_close"],
                        "current_open": gap_info["current_open"],
                        "state": gap_info["state"],
                    }

            signals.append(
                TradingSignal(
                    signal_type=signal_type,
                    timestamp=bar["timestamp"],
                    price=bar["close"],
                    reason=reason,
                    metadata=metadata,
                )
            )

        return signals

    def get_option_direction(self, signal: TradingSignal) -> str:
        """
        Determine option type based on gap direction.

        Args:
            signal: TradingSignal from generate_signals

        Returns:
            'call' for gap down fade (bullish), 'put' for gap up fade (bearish)
        """
        if signal.signal_type == "BUY_CALL":
            return "call"
        elif signal.signal_type == "BUY_PUT":
            return "put"
        return "call"  # Default


def calculate_directional_accuracy(trades: List[dict]) -> dict:
    """
    Calculate directional accuracy of gap fade trades.

    This measures whether the gap actually faded, separate from P&L.
    Helps validate the underlying thesis.

    For gap UP (BUY_PUT): Correct if price fell from entry
    For gap DOWN (BUY_CALL): Correct if price rose from entry

    Args:
        trades: List of completed trades with metadata

    Returns:
        Dict with accuracy metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "correct_direction": 0,
            "directional_accuracy": 0.0,
            "gap_up_correct": 0,
            "gap_up_total": 0,
            "gap_down_correct": 0,
            "gap_down_total": 0,
        }

    correct = 0
    gap_up_correct = 0
    gap_up_total = 0
    gap_down_correct = 0
    gap_down_total = 0

    for trade in trades:
        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        signal_type = trade.get("signal_type", "")
        metadata = trade.get("metadata", {})
        gap_pct = metadata.get("gap_pct", 0)

        if gap_pct > 0:
            # Gap UP, faded with PUT - correct if underlying fell
            gap_up_total += 1
            # For puts, profit means underlying fell
            if trade.get("pnl_dollars", 0) > 0:
                correct += 1
                gap_up_correct += 1
        elif gap_pct < 0:
            # Gap DOWN, faded with CALL - correct if underlying rose
            gap_down_total += 1
            # For calls, profit means underlying rose
            if trade.get("pnl_dollars", 0) > 0:
                correct += 1
                gap_down_correct += 1

    total = len(trades)
    accuracy = correct / total * 100 if total > 0 else 0

    return {
        "total_trades": total,
        "correct_direction": correct,
        "directional_accuracy": accuracy,
        "gap_up_correct": gap_up_correct,
        "gap_up_total": gap_up_total,
        "gap_up_accuracy": gap_up_correct / gap_up_total * 100 if gap_up_total > 0 else 0,
        "gap_down_correct": gap_down_correct,
        "gap_down_total": gap_down_total,
        "gap_down_accuracy": gap_down_correct / gap_down_total * 100 if gap_down_total > 0 else 0,
    }
