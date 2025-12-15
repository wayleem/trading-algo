"""
Trading Signal Generation.

RSI-based signal generators for detecting entry opportunities in 0DTE
options trading. Supports multiple indicator modes including RSI crossover,
MACD filtering, and Bollinger Band strategies.
"""

from datetime import datetime
from typing import Optional
import math

import numpy as np

from app.models.schemas import SignalType, TradingSignal
from app.core.config import settings


class SignalGenerator:
    """
    Generate buy signals based on RSI strategy.

    Entry Logic:
    - BUY CALL: RSI is oversold (<30) AND RSI crosses UP through SMA
    - BUY PUT: RSI is overbought (>70) AND RSI crosses DOWN through SMA

    The RSI/SMA crossover while in extreme zone indicates momentum reversal.
    """

    def __init__(
        self,
        rsi_oversold: float = None,
        rsi_overbought: float = None,
        rsi_sma_gap: float = None,
    ):
        self.rsi_oversold = rsi_oversold or settings.rsi_oversold
        self.rsi_overbought = rsi_overbought or settings.rsi_overbought
        self.rsi_sma_gap = rsi_sma_gap or settings.rsi_sma_gap

        # Track previous values for crossover detection
        self._prev_rsi: Optional[float] = None
        self._prev_sma: Optional[float] = None

    def reset(self):
        """Reset previous values (e.g., at start of new trading day)."""
        self._prev_rsi = None
        self._prev_sma = None

    def evaluate(
        self,
        current_rsi: float,
        current_sma: float,
        close_price: float,
        timestamp: datetime,
        pattern_confirmation: Optional[tuple] = None,
    ) -> TradingSignal:
        """
        Evaluate current market state and generate signal.

        Entry Logic:
        - BUY CALL: RSI < 30 (oversold) AND RSI crosses UP through SMA
        - BUY PUT: RSI > 70 (overbought) AND RSI crosses DOWN through SMA

        Note: Candlestick patterns do NOT filter entry signals.
        Pattern info is passed through for position sizing only.

        Args:
            current_rsi: Current RSI(14) value
            current_sma: Current 14-period SMA of RSI
            close_price: Current close price of underlying
            timestamp: Current bar timestamp
            pattern_confirmation: Optional tuple for position sizing (not filtering).
                Format: (is_bullish, is_bearish, strength, pattern_name)

        Returns:
            TradingSignal with signal type and metadata
        """
        signal_type = SignalType.NO_SIGNAL
        reason = "No signal conditions met"

        if self._prev_rsi is not None and self._prev_sma is not None:
            # BUY CALL: RSI < 30 (oversold) AND RSI crosses UP through SMA
            if current_rsi < self.rsi_oversold:
                if self._prev_rsi < self._prev_sma and current_rsi >= current_sma:
                    signal_type = SignalType.BUY_CALL
                    reason = f"RSI oversold ({current_rsi:.1f}) crossed up through SMA ({current_sma:.1f})"

            # BUY PUT: RSI > 70 (overbought) AND RSI crosses DOWN through SMA
            elif current_rsi > self.rsi_overbought:
                if self._prev_rsi > self._prev_sma and current_rsi <= current_sma:
                    signal_type = SignalType.BUY_PUT
                    reason = f"RSI overbought ({current_rsi:.1f}) crossed down through SMA ({current_sma:.1f})"

        # Update previous values for next evaluation
        self._prev_rsi = current_rsi
        self._prev_sma = current_sma

        return TradingSignal(
            signal_type=signal_type,
            timestamp=timestamp,
            rsi=current_rsi,
            rsi_sma=current_sma,
            close_price=close_price,
            reason=reason,
        )

    def evaluate_series(
        self,
        rsi_values: list[float],
        sma_values: list[float],
        close_prices: list[float],
        timestamps: list[datetime],
        pattern_confirmations: Optional[list[tuple]] = None,
    ) -> list[TradingSignal]:
        """
        Evaluate a series of bars and generate signals.

        Used for backtesting - processes all bars in sequence.

        Args:
            rsi_values: List of RSI values
            sma_values: List of RSI SMA values
            close_prices: List of close prices
            timestamps: List of timestamps
            pattern_confirmations: Optional list of pattern tuples.
                Supports both legacy 2-tuple (is_bullish, is_bearish) and
                new 4-tuple (is_bullish, is_bearish, strength, pattern_name)

        Returns:
            List of TradingSignals (one per bar)
        """
        self.reset()
        signals = []

        for i in range(len(rsi_values)):
            pattern_conf = pattern_confirmations[i] if pattern_confirmations else None
            signal = self.evaluate(
                current_rsi=rsi_values[i],
                current_sma=sma_values[i],
                close_price=close_prices[i],
                timestamp=timestamps[i],
                pattern_confirmation=pattern_conf,
            )
            signals.append(signal)

        return signals


class EnhancedSignalGenerator:
    """
    Enhanced signal generator supporting multiple indicators and confirmation modes.

    Modes:
    - rsi_only: Original RSI crossover logic (backwards compatible)
    - macd_filter: RSI signals filtered by MACD histogram direction
    - independent: Each indicator generates independent signals
    - hybrid: MACD confirms RSI momentum, BB provides exit levels

    RSI Confirmation Modes:
    - none: 3-min and 5-min run independently (parallel mode)
    - both_agree: Both timeframes must signal
    - 5min_trigger: 5-min generates signal, 3-min confirms zone
    - either_triggers: Either timeframe signals, other confirms zone
    """

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        signal_mode: str = "rsi_only",
        rsi_confirmation_mode: str = "none",
        rsi_confirm_buffer: float = 5.0,
        macd_filter_calls: bool = True,
        macd_filter_puts: bool = True,
        macd_signal_threshold: float = 0.0,
        bb_entry_strategy: str = "none",
        bb_volatility_filter: bool = False,
        bb_width_threshold: float = 2.0,
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.signal_mode = signal_mode
        self.rsi_confirmation_mode = rsi_confirmation_mode
        self.rsi_confirm_buffer = rsi_confirm_buffer
        self.macd_filter_calls = macd_filter_calls
        self.macd_filter_puts = macd_filter_puts
        self.macd_signal_threshold = macd_signal_threshold
        self.bb_entry_strategy = bb_entry_strategy
        self.bb_volatility_filter = bb_volatility_filter
        self.bb_width_threshold = bb_width_threshold

        # Base RSI generator for fallback
        self._base_generator = SignalGenerator(rsi_oversold, rsi_overbought)

    def evaluate_series(
        self,
        rsi_values: list[float],
        sma_values: list[float],
        close_prices: list[float],
        timestamps: list[datetime],
        macd_histogram: Optional[list[float]] = None,
        bb_upper: Optional[list[float]] = None,
        bb_lower: Optional[list[float]] = None,
        bb_middle: Optional[list[float]] = None,
        bb_width: Optional[list[float]] = None,
        rsi_values_secondary: Optional[list[float]] = None,
        sma_values_secondary: Optional[list[float]] = None,
        timestamps_secondary: Optional[list[datetime]] = None,
        pattern_confirmations: Optional[list[tuple]] = None,
    ) -> list[TradingSignal]:
        """
        Generate signals based on configured mode.

        Args:
            rsi_values: Primary timeframe RSI values
            sma_values: Primary timeframe RSI SMA values
            close_prices: Primary timeframe close prices
            timestamps: Primary timeframe timestamps
            macd_histogram: MACD histogram values (optional)
            bb_upper/bb_lower/bb_middle: Bollinger Bands (optional)
            bb_width: Band width percentage (optional)
            rsi_values_secondary: Secondary timeframe RSI (for dual RSI)
            sma_values_secondary: Secondary timeframe RSI SMA (for dual RSI)
            timestamps_secondary: Secondary timeframe timestamps
            pattern_confirmations: Pattern data for position sizing

        Returns:
            List of TradingSignals
        """
        if self.signal_mode == "rsi_only" and self.rsi_confirmation_mode == "none":
            return self._base_generator.evaluate_series(
                rsi_values, sma_values, close_prices, timestamps, pattern_confirmations
            )
        elif self.signal_mode == "macd_filter":
            return self._evaluate_macd_filtered(
                rsi_values, sma_values, close_prices, timestamps,
                macd_histogram, bb_width, pattern_confirmations
            )
        elif self.signal_mode == "independent":
            return self._evaluate_independent(
                rsi_values, sma_values, close_prices, timestamps,
                macd_histogram, bb_upper, bb_lower, bb_width, pattern_confirmations
            )
        elif self.signal_mode == "hybrid":
            return self._evaluate_hybrid(
                rsi_values, sma_values, close_prices, timestamps,
                macd_histogram, bb_width, pattern_confirmations
            )
        else:
            # Handle dual RSI confirmation with rsi_only signal mode
            if self.rsi_confirmation_mode != "none" and rsi_values_secondary is not None:
                return self._evaluate_dual_rsi(
                    rsi_values, sma_values, close_prices, timestamps,
                    rsi_values_secondary, sma_values_secondary, timestamps_secondary,
                    pattern_confirmations
                )
            return self._base_generator.evaluate_series(
                rsi_values, sma_values, close_prices, timestamps, pattern_confirmations
            )

    def _check_rsi_crossover(
        self,
        prev_rsi: float,
        curr_rsi: float,
        prev_sma: float,
        curr_sma: float,
    ) -> Optional[SignalType]:
        """Check for RSI/SMA crossover in extreme zones."""
        if np.isnan(prev_rsi) or np.isnan(curr_rsi) or np.isnan(prev_sma) or np.isnan(curr_sma):
            return None

        # BUY CALL: RSI < oversold AND crosses UP through SMA
        if curr_rsi < self.rsi_oversold:
            if prev_rsi < prev_sma and curr_rsi >= curr_sma:
                return SignalType.BUY_CALL

        # BUY PUT: RSI > overbought AND crosses DOWN through SMA
        if curr_rsi > self.rsi_overbought:
            if prev_rsi > prev_sma and curr_rsi <= curr_sma:
                return SignalType.BUY_PUT

        return None

    def _check_rsi_in_zone(self, rsi: float, signal_type: SignalType) -> bool:
        """Check if RSI is in the confirming zone for a signal type."""
        if np.isnan(rsi):
            return False

        if signal_type == SignalType.BUY_CALL:
            return rsi < (self.rsi_oversold + self.rsi_confirm_buffer)
        elif signal_type == SignalType.BUY_PUT:
            return rsi > (self.rsi_overbought - self.rsi_confirm_buffer)
        return False

    def _apply_volatility_filter(self, bb_width: Optional[list[float]], idx: int) -> bool:
        """Check if volatility filter passes (band width >= threshold)."""
        if not self.bb_volatility_filter or bb_width is None:
            return True
        if idx >= len(bb_width) or np.isnan(bb_width[idx]):
            return True  # Pass if no data
        return bb_width[idx] >= self.bb_width_threshold

    def _evaluate_macd_filtered(
        self,
        rsi_values: list[float],
        sma_values: list[float],
        close_prices: list[float],
        timestamps: list[datetime],
        macd_histogram: Optional[list[float]],
        bb_width: Optional[list[float]],
        pattern_confirmations: Optional[list[tuple]],
    ) -> list[TradingSignal]:
        """
        RSI signals filtered by MACD histogram.

        CALL signals require positive histogram.
        PUT signals require negative histogram.
        """
        signals = []

        for i in range(len(rsi_values)):
            signal_type = SignalType.NO_SIGNAL
            reason = "No signal conditions met"

            if i > 0:
                rsi_signal = self._check_rsi_crossover(
                    rsi_values[i-1], rsi_values[i],
                    sma_values[i-1], sma_values[i]
                )

                if rsi_signal is not None:
                    # Check volatility filter
                    if not self._apply_volatility_filter(bb_width, i):
                        reason = f"RSI signal blocked by volatility filter (width too low)"
                    # Check MACD filter
                    elif macd_histogram is not None and i < len(macd_histogram):
                        hist = macd_histogram[i]
                        if not np.isnan(hist):
                            if rsi_signal == SignalType.BUY_CALL:
                                if self.macd_filter_calls and hist > self.macd_signal_threshold:
                                    signal_type = SignalType.BUY_CALL
                                    reason = f"RSI oversold ({rsi_values[i]:.1f}) + MACD histogram positive ({hist:.4f})"
                                elif not self.macd_filter_calls:
                                    signal_type = SignalType.BUY_CALL
                                    reason = f"RSI oversold ({rsi_values[i]:.1f}) crossed SMA"
                                else:
                                    reason = f"RSI call blocked by MACD histogram ({hist:.4f} <= {self.macd_signal_threshold})"
                            elif rsi_signal == SignalType.BUY_PUT:
                                if self.macd_filter_puts and hist < -self.macd_signal_threshold:
                                    signal_type = SignalType.BUY_PUT
                                    reason = f"RSI overbought ({rsi_values[i]:.1f}) + MACD histogram negative ({hist:.4f})"
                                elif not self.macd_filter_puts:
                                    signal_type = SignalType.BUY_PUT
                                    reason = f"RSI overbought ({rsi_values[i]:.1f}) crossed SMA"
                                else:
                                    reason = f"RSI put blocked by MACD histogram ({hist:.4f} >= {-self.macd_signal_threshold})"
                    else:
                        # No MACD data, pass through RSI signal
                        signal_type = rsi_signal
                        reason = f"RSI signal (no MACD data)"

            signals.append(TradingSignal(
                signal_type=signal_type,
                timestamp=timestamps[i],
                rsi=rsi_values[i],
                rsi_sma=sma_values[i],
                close_price=close_prices[i],
                reason=reason,
            ))

        return signals

    def _evaluate_independent(
        self,
        rsi_values: list[float],
        sma_values: list[float],
        close_prices: list[float],
        timestamps: list[datetime],
        macd_histogram: Optional[list[float]],
        bb_upper: Optional[list[float]],
        bb_lower: Optional[list[float]],
        bb_width: Optional[list[float]],
        pattern_confirmations: Optional[list[tuple]],
    ) -> list[TradingSignal]:
        """
        Independent signals from RSI, MACD, and Bollinger Bands.

        Each indicator can generate its own signal:
        - RSI: Standard crossover logic
        - MACD: Histogram crossover (positive->negative for puts, negative->positive for calls)
        - BB: Price touching bands (lower for calls, upper for puts)
        """
        signals = []

        for i in range(len(rsi_values)):
            signal_type = SignalType.NO_SIGNAL
            reason = "No signal conditions met"
            source = "none"

            # Check volatility filter first
            if not self._apply_volatility_filter(bb_width, i):
                signals.append(TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    timestamp=timestamps[i],
                    rsi=rsi_values[i],
                    rsi_sma=sma_values[i],
                    close_price=close_prices[i],
                    reason="Volatility filter: band width too low",
                ))
                continue

            # 1. Check RSI signal
            if i > 0:
                rsi_signal = self._check_rsi_crossover(
                    rsi_values[i-1], rsi_values[i],
                    sma_values[i-1], sma_values[i]
                )
                if rsi_signal is not None:
                    signal_type = rsi_signal
                    reason = f"RSI {'oversold' if rsi_signal == SignalType.BUY_CALL else 'overbought'} crossover"
                    source = "rsi"

            # 2. Check MACD histogram crossover (only if no RSI signal)
            if signal_type == SignalType.NO_SIGNAL and macd_histogram is not None and i > 0:
                if i < len(macd_histogram) and not np.isnan(macd_histogram[i]) and not np.isnan(macd_histogram[i-1]):
                    prev_hist = macd_histogram[i-1]
                    curr_hist = macd_histogram[i]
                    # MACD histogram crosses from negative to positive = bullish
                    if prev_hist < 0 and curr_hist >= 0:
                        signal_type = SignalType.BUY_CALL
                        reason = f"MACD histogram crossed positive ({curr_hist:.4f})"
                        source = "macd"
                    # MACD histogram crosses from positive to negative = bearish
                    elif prev_hist > 0 and curr_hist <= 0:
                        signal_type = SignalType.BUY_PUT
                        reason = f"MACD histogram crossed negative ({curr_hist:.4f})"
                        source = "macd"

            # 3. Check BB band touch (only if no other signal)
            if signal_type == SignalType.NO_SIGNAL and self.bb_entry_strategy == "touch":
                if bb_lower is not None and bb_upper is not None:
                    if i < len(bb_lower) and not np.isnan(bb_lower[i]) and not np.isnan(bb_upper[i]):
                        price = close_prices[i]
                        # Price touches or crosses lower band = bullish
                        if price <= bb_lower[i]:
                            signal_type = SignalType.BUY_CALL
                            reason = f"Price ({price:.2f}) touched lower BB ({bb_lower[i]:.2f})"
                            source = "bb"
                        # Price touches or crosses upper band = bearish
                        elif price >= bb_upper[i]:
                            signal_type = SignalType.BUY_PUT
                            reason = f"Price ({price:.2f}) touched upper BB ({bb_upper[i]:.2f})"
                            source = "bb"

            signals.append(TradingSignal(
                signal_type=signal_type,
                timestamp=timestamps[i],
                rsi=rsi_values[i],
                rsi_sma=sma_values[i],
                close_price=close_prices[i],
                reason=reason,
            ))

        return signals

    def _evaluate_hybrid(
        self,
        rsi_values: list[float],
        sma_values: list[float],
        close_prices: list[float],
        timestamps: list[datetime],
        macd_histogram: Optional[list[float]],
        bb_width: Optional[list[float]],
        pattern_confirmations: Optional[list[tuple]],
    ) -> list[TradingSignal]:
        """
        Hybrid mode: RSI entry + MACD momentum confirmation.

        Entry requires:
        1. RSI crossover signal
        2. MACD histogram confirming direction (positive for calls, negative for puts)

        BB is used for exit levels (handled in exit_strategy.py).
        """
        # Hybrid is essentially MACD filter with stricter threshold
        return self._evaluate_macd_filtered(
            rsi_values, sma_values, close_prices, timestamps,
            macd_histogram, bb_width, pattern_confirmations
        )

    def _evaluate_dual_rsi(
        self,
        rsi_primary: list[float],
        sma_primary: list[float],
        close_prices: list[float],
        timestamps: list[datetime],
        rsi_secondary: list[float],
        sma_secondary: list[float],
        timestamps_secondary: list[datetime],
        pattern_confirmations: Optional[list[tuple]],
    ) -> list[TradingSignal]:
        """
        Dual RSI confirmation mode.

        Modes:
        - both_agree: Both timeframes must signal
        - 5min_trigger: Primary (5-min) signals, secondary (3-min) confirms zone
        - either_triggers: Either signals, other confirms zone
        """
        signals = []

        # Build lookup for secondary timeframe by timestamp
        secondary_lookup = {}
        for i in range(len(timestamps_secondary)):
            ts = timestamps_secondary[i]
            secondary_lookup[ts] = {
                "rsi": rsi_secondary[i],
                "sma": sma_secondary[i],
                "prev_rsi": rsi_secondary[i-1] if i > 0 else float("nan"),
                "prev_sma": sma_secondary[i-1] if i > 0 else float("nan"),
            }

        for i in range(len(rsi_primary)):
            signal_type = SignalType.NO_SIGNAL
            reason = "No signal conditions met"

            if i > 0:
                ts = timestamps[i]

                # Get primary signal
                primary_signal = self._check_rsi_crossover(
                    rsi_primary[i-1], rsi_primary[i],
                    sma_primary[i-1], sma_primary[i]
                )

                # Get secondary data at same or nearest timestamp
                secondary_data = self._get_secondary_at_time(ts, secondary_lookup, timestamps_secondary)

                if self.rsi_confirmation_mode == "both_agree":
                    # Both must generate crossover signal
                    if primary_signal is not None and secondary_data is not None:
                        secondary_signal = self._check_rsi_crossover(
                            secondary_data["prev_rsi"], secondary_data["rsi"],
                            secondary_data["prev_sma"], secondary_data["sma"]
                        )
                        if secondary_signal == primary_signal:
                            signal_type = primary_signal
                            reason = f"Both timeframes agree: {signal_type.name}"

                elif self.rsi_confirmation_mode == "5min_trigger":
                    # Primary (5-min) triggers, secondary (3-min) confirms zone
                    if primary_signal is not None and secondary_data is not None:
                        if self._check_rsi_in_zone(secondary_data["rsi"], primary_signal):
                            signal_type = primary_signal
                            reason = f"5-min triggered, 3-min RSI ({secondary_data['rsi']:.1f}) confirms zone"

                elif self.rsi_confirmation_mode == "either_triggers":
                    # Either can trigger, other must confirm zone
                    if secondary_data is not None:
                        secondary_signal = self._check_rsi_crossover(
                            secondary_data["prev_rsi"], secondary_data["rsi"],
                            secondary_data["prev_sma"], secondary_data["sma"]
                        )

                        # Primary triggers, secondary confirms
                        if primary_signal is not None:
                            if self._check_rsi_in_zone(secondary_data["rsi"], primary_signal):
                                signal_type = primary_signal
                                reason = f"Primary triggered, secondary confirms zone"
                        # Secondary triggers, primary confirms
                        elif secondary_signal is not None:
                            if self._check_rsi_in_zone(rsi_primary[i], secondary_signal):
                                signal_type = secondary_signal
                                reason = f"Secondary triggered, primary confirms zone"

            signals.append(TradingSignal(
                signal_type=signal_type,
                timestamp=timestamps[i],
                rsi=rsi_primary[i],
                rsi_sma=sma_primary[i],
                close_price=close_prices[i],
                reason=reason,
            ))

        return signals

    def _get_secondary_at_time(
        self,
        target_ts: datetime,
        lookup: dict,
        timestamps: list[datetime],
    ) -> Optional[dict]:
        """Get secondary timeframe data at or before target timestamp."""
        if target_ts in lookup:
            return lookup[target_ts]

        # Find most recent timestamp <= target
        best_ts = None
        for ts in timestamps:
            if ts <= target_ts:
                if best_ts is None or ts > best_ts:
                    best_ts = ts

        return lookup.get(best_ts) if best_ts else None
