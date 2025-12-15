"""
Candlestick pattern recognition for trading signal confirmation.

Implements detection of major reversal and continuation patterns to filter
RSI signals and guide averaging down decisions.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class PatternType(Enum):
    """Classification of candlestick patterns."""

    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    BULLISH_CONTINUATION = "bullish_continuation"
    BEARISH_CONTINUATION = "bearish_continuation"
    NEUTRAL = "neutral"


@dataclass
class CandlestickPattern:
    """Detected candlestick pattern with metadata."""

    name: str
    pattern_type: PatternType
    strength: float  # 0.0 to 1.0 (confidence/reliability)
    bars_required: int  # Number of candles in pattern


class CandlestickAnalyzer:
    """
    Detect candlestick patterns from OHLC bar data.

    Supports both single-candle patterns (doji, hammer) and multi-candle
    patterns (engulfing, morning star).
    """

    def __init__(
        self,
        body_threshold: float = 0.1,
        shadow_ratio: float = 2.0,
        engulfing_factor: float = 1.0,
    ):
        """
        Initialize analyzer with detection thresholds.

        Args:
            body_threshold: Max body/range ratio for doji detection (default 10%)
            shadow_ratio: Min shadow/body ratio for hammer/shooting star
            engulfing_factor: How much larger engulfing candle must be
        """
        self.body_threshold = body_threshold
        self.shadow_ratio = shadow_ratio
        self.engulfing_factor = engulfing_factor

    def analyze(self, bars: list[dict], lookback: int = 3) -> Optional[CandlestickPattern]:
        """
        Analyze recent bars and return the strongest detected pattern.

        Args:
            bars: List of OHLC bar dicts (most recent last)
            lookback: Number of bars to consider

        Returns:
            CandlestickPattern if found, None otherwise
        """
        if not bars:
            return None

        recent_bars = bars[-lookback:] if len(bars) >= lookback else bars

        # Check patterns in order of reliability (most reliable first)
        patterns_to_check = [
            (self._detect_morning_star, 3),
            (self._detect_evening_star, 3),
            (self._detect_three_white_soldiers, 3),
            (self._detect_three_black_crows, 3),
            (self._detect_engulfing, 2),
            (self._detect_piercing_line, 2),
            (self._detect_dark_cloud_cover, 2),
            (self._detect_hammer, 1),
            (self._detect_inverted_hammer, 1),
            (self._detect_hanging_man, 1),
            (self._detect_shooting_star, 1),
            (self._detect_doji, 1),
            (self._detect_marubozu, 1),
        ]

        for detector, min_bars in patterns_to_check:
            if len(recent_bars) >= min_bars:
                pattern = detector(recent_bars[-min_bars:])
                if pattern:
                    return pattern

        return None

    def is_bullish_confirmation(self, bars: list[dict]) -> bool:
        """
        Check if recent pattern confirms a bullish signal (for RSI oversold).

        Args:
            bars: Recent OHLC bars (at least 1-3 depending on pattern)

        Returns:
            True if a bullish pattern is present
        """
        pattern = self.analyze(bars)
        if pattern is None:
            return False

        return pattern.pattern_type in (
            PatternType.BULLISH_REVERSAL,
            PatternType.BULLISH_CONTINUATION,
        )

    def is_bearish_confirmation(self, bars: list[dict]) -> bool:
        """
        Check if recent pattern confirms a bearish signal (for RSI overbought).

        Args:
            bars: Recent OHLC bars

        Returns:
            True if a bearish pattern is present
        """
        pattern = self.analyze(bars)
        if pattern is None:
            return False

        return pattern.pattern_type in (
            PatternType.BEARISH_REVERSAL,
            PatternType.BEARISH_CONTINUATION,
        )

    def should_add_contracts(self, bars: list[dict], position_type: str) -> bool:
        """
        Check if pattern supports averaging down on existing position.

        For calls: look for bullish continuation or reversal patterns
        For puts: look for bearish continuation or reversal patterns

        Args:
            bars: Recent OHLC bars
            position_type: "call" or "put"

        Returns:
            True if pattern supports adding to position
        """
        pattern = self.analyze(bars)
        if pattern is None:
            return False

        if position_type.lower() == "call":
            # For calls, want bullish patterns to add
            return pattern.pattern_type in (
                PatternType.BULLISH_REVERSAL,
                PatternType.BULLISH_CONTINUATION,
            )
        else:
            # For puts, want bearish patterns to add
            return pattern.pattern_type in (
                PatternType.BEARISH_REVERSAL,
                PatternType.BEARISH_CONTINUATION,
            )

    def get_confirmation_with_strength(
        self, bars: list[dict], position_type: str
    ) -> tuple[bool, float, Optional[str]]:
        """
        Get pattern confirmation with strength for position sizing decisions.

        Args:
            bars: Recent OHLC bars (at least 1-3 depending on pattern)
            position_type: "call" or "put"

        Returns:
            Tuple of (is_confirmed, strength, pattern_name)
            - is_confirmed: True if pattern confirms the signal direction
            - strength: Pattern strength 0.0-1.0 (0.0 if not confirmed)
            - pattern_name: Name of the pattern or None
        """
        pattern = self.analyze(bars)
        if pattern is None:
            return (False, 0.0, None)

        if position_type.lower() == "call":
            is_confirmed = pattern.pattern_type in (
                PatternType.BULLISH_REVERSAL,
                PatternType.BULLISH_CONTINUATION,
            )
        else:
            is_confirmed = pattern.pattern_type in (
                PatternType.BEARISH_REVERSAL,
                PatternType.BEARISH_CONTINUATION,
            )

        return (
            is_confirmed,
            pattern.strength if is_confirmed else 0.0,
            pattern.name if is_confirmed else None,
        )

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _get_body(self, bar: dict) -> float:
        """Get candle body size (absolute)."""
        return abs(bar["close"] - bar["open"])

    def _get_range(self, bar: dict) -> float:
        """Get candle full range (high - low)."""
        return bar["high"] - bar["low"]

    def _is_bullish(self, bar: dict) -> bool:
        """Check if candle closed higher than it opened."""
        return bar["close"] > bar["open"]

    def _is_bearish(self, bar: dict) -> bool:
        """Check if candle closed lower than it opened."""
        return bar["close"] < bar["open"]

    def _get_upper_shadow(self, bar: dict) -> float:
        """Get upper shadow length."""
        if self._is_bullish(bar):
            return bar["high"] - bar["close"]
        return bar["high"] - bar["open"]

    def _get_lower_shadow(self, bar: dict) -> float:
        """Get lower shadow length."""
        if self._is_bullish(bar):
            return bar["open"] - bar["low"]
        return bar["close"] - bar["low"]

    # =========================================================================
    # Single-candle pattern detectors
    # =========================================================================

    def _detect_doji(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Doji pattern (indecision, potential reversal).

        Doji: Body is very small relative to range.
        """
        bar = bars[-1]
        range_size = self._get_range(bar)

        if range_size == 0:
            return None

        body_ratio = self._get_body(bar) / range_size

        if body_ratio <= self.body_threshold:
            return CandlestickPattern(
                name="Doji",
                pattern_type=PatternType.NEUTRAL,
                strength=0.5,
                bars_required=1,
            )
        return None

    def _detect_hammer(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Hammer pattern (bullish reversal at bottom of downtrend).

        Hammer: Small body at top, long lower shadow, little/no upper shadow.
        """
        bar = bars[-1]
        body = self._get_body(bar)
        lower_shadow = self._get_lower_shadow(bar)
        upper_shadow = self._get_upper_shadow(bar)

        if body == 0:
            return None

        # Hammer criteria:
        # 1. Lower shadow >= 2x body
        # 2. Upper shadow <= 0.1x body (small or none)
        # 3. Body in upper third of range
        if (
            lower_shadow >= self.shadow_ratio * body
            and upper_shadow <= 0.3 * body
        ):
            return CandlestickPattern(
                name="Hammer",
                pattern_type=PatternType.BULLISH_REVERSAL,
                strength=0.7,
                bars_required=1,
            )
        return None

    def _detect_inverted_hammer(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Inverted Hammer (bullish reversal).

        Inverted Hammer: Small body at bottom, long upper shadow.
        """
        bar = bars[-1]
        body = self._get_body(bar)
        upper_shadow = self._get_upper_shadow(bar)
        lower_shadow = self._get_lower_shadow(bar)

        if body == 0:
            return None

        if (
            upper_shadow >= self.shadow_ratio * body
            and lower_shadow <= 0.3 * body
        ):
            return CandlestickPattern(
                name="Inverted Hammer",
                pattern_type=PatternType.BULLISH_REVERSAL,
                strength=0.6,
                bars_required=1,
            )
        return None

    def _detect_hanging_man(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Hanging Man (bearish reversal at top of uptrend).

        Same shape as hammer but appears after uptrend.
        """
        bar = bars[-1]
        body = self._get_body(bar)
        lower_shadow = self._get_lower_shadow(bar)
        upper_shadow = self._get_upper_shadow(bar)

        if body == 0:
            return None

        # Same shape as hammer
        if (
            lower_shadow >= self.shadow_ratio * body
            and upper_shadow <= 0.3 * body
        ):
            return CandlestickPattern(
                name="Hanging Man",
                pattern_type=PatternType.BEARISH_REVERSAL,
                strength=0.6,
                bars_required=1,
            )
        return None

    def _detect_shooting_star(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Shooting Star (bearish reversal).

        Shooting Star: Small body at bottom, long upper shadow.
        """
        bar = bars[-1]
        body = self._get_body(bar)
        upper_shadow = self._get_upper_shadow(bar)
        lower_shadow = self._get_lower_shadow(bar)

        if body == 0:
            return None

        if (
            upper_shadow >= self.shadow_ratio * body
            and lower_shadow <= 0.3 * body
        ):
            return CandlestickPattern(
                name="Shooting Star",
                pattern_type=PatternType.BEARISH_REVERSAL,
                strength=0.7,
                bars_required=1,
            )
        return None

    def _detect_marubozu(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Marubozu (strong momentum candle with no shadows).

        Bullish Marubozu: Open = Low, Close = High
        Bearish Marubozu: Open = High, Close = Low
        """
        bar = bars[-1]
        range_size = self._get_range(bar)

        if range_size == 0:
            return None

        body = self._get_body(bar)
        body_ratio = body / range_size

        # Marubozu: body is almost entire range (> 95%)
        if body_ratio >= 0.95:
            if self._is_bullish(bar):
                return CandlestickPattern(
                    name="Bullish Marubozu",
                    pattern_type=PatternType.BULLISH_CONTINUATION,
                    strength=0.8,
                    bars_required=1,
                )
            else:
                return CandlestickPattern(
                    name="Bearish Marubozu",
                    pattern_type=PatternType.BEARISH_CONTINUATION,
                    strength=0.8,
                    bars_required=1,
                )
        return None

    # =========================================================================
    # Two-candle pattern detectors
    # =========================================================================

    def _detect_engulfing(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Bullish or Bearish Engulfing pattern.

        Engulfing: Second candle's body completely engulfs first candle's body.
        """
        if len(bars) < 2:
            return None

        first, second = bars[-2], bars[-1]

        first_body = self._get_body(first)
        second_body = self._get_body(second)

        # Second body must be larger
        if second_body <= first_body * self.engulfing_factor:
            return None

        # Bullish engulfing: bearish first, bullish second that engulfs
        if self._is_bearish(first) and self._is_bullish(second):
            if (
                second["open"] <= first["close"]
                and second["close"] >= first["open"]
            ):
                return CandlestickPattern(
                    name="Bullish Engulfing",
                    pattern_type=PatternType.BULLISH_REVERSAL,
                    strength=0.8,
                    bars_required=2,
                )

        # Bearish engulfing: bullish first, bearish second that engulfs
        if self._is_bullish(first) and self._is_bearish(second):
            if (
                second["open"] >= first["close"]
                and second["close"] <= first["open"]
            ):
                return CandlestickPattern(
                    name="Bearish Engulfing",
                    pattern_type=PatternType.BEARISH_REVERSAL,
                    strength=0.8,
                    bars_required=2,
                )

        return None

    def _detect_piercing_line(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Piercing Line (bullish reversal).

        Piercing: Bearish candle followed by bullish candle that opens below
        prior low and closes above midpoint of prior body.
        """
        if len(bars) < 2:
            return None

        first, second = bars[-2], bars[-1]

        if not self._is_bearish(first) or not self._is_bullish(second):
            return None

        first_midpoint = (first["open"] + first["close"]) / 2

        # Second opens below first's low, closes above first's midpoint
        if second["open"] < first["low"] and second["close"] > first_midpoint:
            return CandlestickPattern(
                name="Piercing Line",
                pattern_type=PatternType.BULLISH_REVERSAL,
                strength=0.7,
                bars_required=2,
            )
        return None

    def _detect_dark_cloud_cover(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Dark Cloud Cover (bearish reversal).

        Dark Cloud: Bullish candle followed by bearish candle that opens above
        prior high and closes below midpoint of prior body.
        """
        if len(bars) < 2:
            return None

        first, second = bars[-2], bars[-1]

        if not self._is_bullish(first) or not self._is_bearish(second):
            return None

        first_midpoint = (first["open"] + first["close"]) / 2

        # Second opens above first's high, closes below first's midpoint
        if second["open"] > first["high"] and second["close"] < first_midpoint:
            return CandlestickPattern(
                name="Dark Cloud Cover",
                pattern_type=PatternType.BEARISH_REVERSAL,
                strength=0.7,
                bars_required=2,
            )
        return None

    # =========================================================================
    # Three-candle pattern detectors
    # =========================================================================

    def _detect_morning_star(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Morning Star (bullish reversal).

        Morning Star: Large bearish, small body (star), large bullish.
        Third candle should close above midpoint of first.
        """
        if len(bars) < 3:
            return None

        first, second, third = bars[-3], bars[-2], bars[-1]

        first_body = self._get_body(first)
        second_body = self._get_body(second)
        third_body = self._get_body(third)

        # First: large bearish
        if not self._is_bearish(first) or first_body < second_body:
            return None

        # Second: small body (star) - less than 50% of first
        if second_body > first_body * 0.5:
            return None

        # Third: large bullish
        if not self._is_bullish(third) or third_body < second_body:
            return None

        # Third closes above first's midpoint
        first_midpoint = (first["open"] + first["close"]) / 2
        if third["close"] > first_midpoint:
            return CandlestickPattern(
                name="Morning Star",
                pattern_type=PatternType.BULLISH_REVERSAL,
                strength=0.9,
                bars_required=3,
            )
        return None

    def _detect_evening_star(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Evening Star (bearish reversal).

        Evening Star: Large bullish, small body (star), large bearish.
        Third candle should close below midpoint of first.
        """
        if len(bars) < 3:
            return None

        first, second, third = bars[-3], bars[-2], bars[-1]

        first_body = self._get_body(first)
        second_body = self._get_body(second)
        third_body = self._get_body(third)

        # First: large bullish
        if not self._is_bullish(first) or first_body < second_body:
            return None

        # Second: small body (star)
        if second_body > first_body * 0.5:
            return None

        # Third: large bearish
        if not self._is_bearish(third) or third_body < second_body:
            return None

        # Third closes below first's midpoint
        first_midpoint = (first["open"] + first["close"]) / 2
        if third["close"] < first_midpoint:
            return CandlestickPattern(
                name="Evening Star",
                pattern_type=PatternType.BEARISH_REVERSAL,
                strength=0.9,
                bars_required=3,
            )
        return None

    def _detect_three_white_soldiers(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Three White Soldiers (bullish continuation).

        Three consecutive bullish candles, each opening within prior body
        and closing higher.
        """
        if len(bars) < 3:
            return None

        first, second, third = bars[-3], bars[-2], bars[-1]

        # All three must be bullish
        if not all(self._is_bullish(b) for b in [first, second, third]):
            return None

        # Each close higher than previous
        if not (third["close"] > second["close"] > first["close"]):
            return None

        # Each opens within prior body
        if not (first["open"] < second["open"] < second["close"]):
            return None
        if not (second["open"] < third["open"] < third["close"]):
            return None

        return CandlestickPattern(
            name="Three White Soldiers",
            pattern_type=PatternType.BULLISH_CONTINUATION,
            strength=0.85,
            bars_required=3,
        )

    def _detect_three_black_crows(self, bars: list[dict]) -> Optional[CandlestickPattern]:
        """
        Detect Three Black Crows (bearish continuation).

        Three consecutive bearish candles, each opening within prior body
        and closing lower.
        """
        if len(bars) < 3:
            return None

        first, second, third = bars[-3], bars[-2], bars[-1]

        # All three must be bearish
        if not all(self._is_bearish(b) for b in [first, second, third]):
            return None

        # Each close lower than previous
        if not (third["close"] < second["close"] < first["close"]):
            return None

        # Each opens within prior body
        if not (first["close"] < second["open"] < first["open"]):
            return None
        if not (second["close"] < third["open"] < second["open"]):
            return None

        return CandlestickPattern(
            name="Three Black Crows",
            pattern_type=PatternType.BEARISH_CONTINUATION,
            strength=0.85,
            bars_required=3,
        )
