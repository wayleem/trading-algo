"""
Support and Resistance Level Detection.

Detects key price levels from historical data for dynamic exit enhancement.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple


@dataclass
class PriceLevel:
    """Represents a support or resistance level."""

    price: float
    strength: int  # Number of touches/tests
    level_type: str  # "support" or "resistance"
    last_touch: Optional[datetime] = None


class SupportResistanceAnalyzer:
    """
    Detect and track support/resistance levels from price history.

    Uses multiple methods:
    1. Swing highs/lows (pivot points)
    2. Price clustering (areas of consolidation)
    3. Round numbers (psychological levels)
    """

    def __init__(
        self,
        lookback_bars: int = 100,
        tolerance_pct: float = 0.5,
        min_touches: int = 2,
        round_number_interval: float = 5.0,
    ):
        """
        Initialize analyzer.

        Args:
            lookback_bars: Number of bars to analyze for S/R detection
            tolerance_pct: Price tolerance for clustering (0.5 = 0.5%)
            min_touches: Minimum touches to qualify as S/R level
            round_number_interval: Interval for round number levels (e.g., 5 for $5 intervals)
        """
        self.lookback_bars = lookback_bars
        self.tolerance_pct = tolerance_pct / 100.0
        self.min_touches = min_touches
        self.round_number_interval = round_number_interval

    def find_levels(
        self, bars: List[dict]
    ) -> Tuple[List[PriceLevel], List[PriceLevel]]:
        """
        Find support and resistance levels from price history.

        Args:
            bars: List of OHLC bar dicts with keys: open, high, low, close, timestamp

        Returns:
            Tuple of (support_levels, resistance_levels) sorted by strength
        """
        if len(bars) < 5:
            return [], []

        # Use the most recent lookback_bars
        recent_bars = bars[-self.lookback_bars :] if len(bars) > self.lookback_bars else bars

        # Find pivot points (swing highs and lows)
        swing_highs, swing_lows = self._find_pivot_points(recent_bars)

        # Cluster nearby levels
        resistance_clusters = self._cluster_levels(swing_highs, recent_bars[-1]["close"])
        support_clusters = self._cluster_levels(swing_lows, recent_bars[-1]["close"])

        # Add round number levels
        current_price = recent_bars[-1]["close"]
        round_resistance, round_support = self._get_round_number_levels(
            current_price, recent_bars
        )

        # Merge round numbers with detected levels
        resistance_levels = self._merge_levels(resistance_clusters, round_resistance)
        support_levels = self._merge_levels(support_clusters, round_support)

        # Filter by minimum touches and sort by strength
        resistance_levels = [
            level for level in resistance_levels if level.strength >= self.min_touches
        ]
        support_levels = [
            level for level in support_levels if level.strength >= self.min_touches
        ]

        resistance_levels.sort(key=lambda x: x.strength, reverse=True)
        support_levels.sort(key=lambda x: x.strength, reverse=True)

        return support_levels, resistance_levels

    def _find_pivot_points(
        self, bars: List[dict], window: int = 3
    ) -> Tuple[List[float], List[float]]:
        """
        Find swing highs and lows using local extrema detection.

        Args:
            bars: List of OHLC bars
            window: Number of bars on each side to confirm pivot

        Returns:
            Tuple of (swing_highs, swing_lows) price lists
        """
        swing_highs = []
        swing_lows = []

        for i in range(window, len(bars) - window):
            high = bars[i]["high"]
            low = bars[i]["low"]

            # Check if this is a swing high
            is_swing_high = all(
                bars[j]["high"] < high for j in range(i - window, i)
            ) and all(bars[j]["high"] < high for j in range(i + 1, i + window + 1))

            # Check if this is a swing low
            is_swing_low = all(
                bars[j]["low"] > low for j in range(i - window, i)
            ) and all(bars[j]["low"] > low for j in range(i + 1, i + window + 1))

            if is_swing_high:
                swing_highs.append(high)
            if is_swing_low:
                swing_lows.append(low)

        return swing_highs, swing_lows

    def _cluster_levels(
        self, prices: List[float], current_price: float
    ) -> List[PriceLevel]:
        """
        Cluster nearby prices into single S/R levels.

        Args:
            prices: List of prices to cluster
            current_price: Current price for calculating tolerance

        Returns:
            List of PriceLevel objects with strength (touch count)
        """
        if not prices:
            return []

        levels = []
        used = set()
        tolerance = current_price * self.tolerance_pct

        for i, price in enumerate(prices):
            if i in used:
                continue

            # Find all prices within tolerance
            cluster = [price]
            for j, other_price in enumerate(prices):
                if j != i and j not in used and abs(other_price - price) <= tolerance:
                    cluster.append(other_price)
                    used.add(j)

            used.add(i)

            # Average the cluster for the level price
            avg_price = sum(cluster) / len(cluster)

            # Determine if this is above or below current price
            level_type = "resistance" if avg_price > current_price else "support"

            levels.append(
                PriceLevel(
                    price=avg_price,
                    strength=len(cluster),
                    level_type=level_type,
                )
            )

        return levels

    def _get_round_number_levels(
        self, current_price: float, bars: List[dict]
    ) -> Tuple[List[PriceLevel], List[PriceLevel]]:
        """
        Get psychological round number levels near current price.

        Args:
            current_price: Current price
            bars: Historical bars to count touches

        Returns:
            Tuple of (resistance_levels, support_levels) from round numbers
        """
        interval = self.round_number_interval
        resistance_levels = []
        support_levels = []

        # Find round numbers above and below current price (within 5%)
        price_range = current_price * 0.05

        # Round numbers above (resistance)
        level = ((current_price // interval) + 1) * interval
        while level <= current_price + price_range:
            touches = self._count_touches(level, bars)
            if touches > 0:
                resistance_levels.append(
                    PriceLevel(price=level, strength=touches, level_type="resistance")
                )
            level += interval

        # Round numbers below (support)
        level = (current_price // interval) * interval
        while level >= current_price - price_range:
            touches = self._count_touches(level, bars)
            if touches > 0:
                support_levels.append(
                    PriceLevel(price=level, strength=touches, level_type="support")
                )
            level -= interval

        return resistance_levels, support_levels

    def _count_touches(self, price: float, bars: List[dict]) -> int:
        """Count how many times price was touched in the bars."""
        tolerance = price * self.tolerance_pct
        touches = 0
        for bar in bars:
            # Price touched if high/low came within tolerance
            if abs(bar["high"] - price) <= tolerance or abs(bar["low"] - price) <= tolerance:
                touches += 1
        return touches

    def _merge_levels(
        self, levels1: List[PriceLevel], levels2: List[PriceLevel]
    ) -> List[PriceLevel]:
        """Merge two lists of levels, combining nearby ones."""
        if not levels1:
            return levels2
        if not levels2:
            return levels1

        all_levels = levels1 + levels2
        merged = []
        used = set()

        for i, level1 in enumerate(all_levels):
            if i in used:
                continue

            # Find overlapping levels
            total_strength = level1.strength
            total_price = level1.price * level1.strength
            count = 1

            for j, level2 in enumerate(all_levels):
                if j != i and j not in used:
                    tolerance = level1.price * self.tolerance_pct
                    if abs(level2.price - level1.price) <= tolerance:
                        total_strength += level2.strength
                        total_price += level2.price * level2.strength
                        count += 1
                        used.add(j)

            used.add(i)
            avg_price = total_price / total_strength
            merged.append(
                PriceLevel(
                    price=avg_price,
                    strength=total_strength,
                    level_type=level1.level_type,
                )
            )

        return merged

    def get_nearest_support(
        self, current_price: float, bars: List[dict]
    ) -> Optional[PriceLevel]:
        """Get nearest support level below current price."""
        support_levels, _ = self.find_levels(bars)
        valid = [level for level in support_levels if level.price < current_price]
        if not valid:
            return None
        return max(valid, key=lambda x: x.price)

    def get_nearest_resistance(
        self, current_price: float, bars: List[dict]
    ) -> Optional[PriceLevel]:
        """Get nearest resistance level above current price."""
        _, resistance_levels = self.find_levels(bars)
        valid = [level for level in resistance_levels if level.price > current_price]
        if not valid:
            return None
        return min(valid, key=lambda x: x.price)

    def calculate_dynamic_profit_target(
        self,
        entry_price: float,
        underlying_price: float,
        position_type: str,  # "call" or "put"
        bars: List[dict],
        base_target_pct: float = 0.10,
    ) -> float:
        """
        Adjust profit target based on S/R levels.

        For calls: target is capped by distance to resistance
        For puts: target is capped by distance to support

        Args:
            entry_price: Option entry price
            underlying_price: Current underlying price
            position_type: "call" or "put"
            bars: Historical bars for S/R detection
            base_target_pct: Base profit target percentage (default 10%)

        Returns:
            Adjusted profit target as percentage (e.g., 0.10 for 10%)
        """
        base_target = base_target_pct

        if position_type == "call":
            # For calls, look for resistance above
            resistance = self.get_nearest_resistance(underlying_price, bars)
            if resistance:
                # Calculate percentage move to resistance
                distance_pct = (resistance.price - underlying_price) / underlying_price
                # Options typically move ~0.5 delta, so option target = 2x underlying move
                option_move_estimate = distance_pct * 2
                # Use the smaller of base target or estimated move to resistance
                if option_move_estimate > 0 and option_move_estimate < base_target:
                    return max(option_move_estimate, base_target * 0.5)  # At least 50% of base
        else:
            # For puts, look for support below
            support = self.get_nearest_support(underlying_price, bars)
            if support:
                # Calculate percentage move to support
                distance_pct = (underlying_price - support.price) / underlying_price
                option_move_estimate = distance_pct * 2
                if option_move_estimate > 0 and option_move_estimate < base_target:
                    return max(option_move_estimate, base_target * 0.5)

        return base_target

    def calculate_dynamic_stop_loss(
        self,
        entry_price: float,
        underlying_price: float,
        position_type: str,  # "call" or "put"
        bars: List[dict],
        base_stop_pct: float = 0.45,
    ) -> float:
        """
        Adjust stop loss based on S/R levels.

        Place stop just beyond key S/R level for better positioning.

        Args:
            entry_price: Option entry price
            underlying_price: Current underlying price
            position_type: "call" or "put"
            bars: Historical bars for S/R detection
            base_stop_pct: Base stop loss percentage (default 45%)

        Returns:
            Adjusted stop loss as percentage (e.g., 0.45 for 45%)
        """
        base_stop = base_stop_pct

        if position_type == "call":
            # For calls, look for support below (stop when support breaks)
            support = self.get_nearest_support(underlying_price, bars)
            if support and support.strength >= 3:  # Strong support
                # Tighten stop if strong support is close
                distance_pct = (underlying_price - support.price) / underlying_price
                if distance_pct > 0.01:  # Support at least 1% below
                    # Widen stop to give support level room (add 10% buffer beyond support)
                    option_stop_estimate = (distance_pct + 0.001) * 2
                    if option_stop_estimate < base_stop:
                        return max(option_stop_estimate, base_stop * 0.5)
        else:
            # For puts, look for resistance above (stop when resistance breaks)
            resistance = self.get_nearest_resistance(underlying_price, bars)
            if resistance and resistance.strength >= 3:
                distance_pct = (resistance.price - underlying_price) / underlying_price
                if distance_pct > 0.01:
                    option_stop_estimate = (distance_pct + 0.001) * 2
                    if option_stop_estimate < base_stop:
                        return max(option_stop_estimate, base_stop * 0.5)

        return base_stop
