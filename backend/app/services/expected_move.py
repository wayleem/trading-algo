"""
Expected Move Calculator for 0DTE Options.

The Expected Move (EM) is calculated from the ATM straddle premium:
  EM = ATM_Call_Price + ATM_Put_Price
  EM_Percentage = EM / Underlying_Price × 100

This tells us how much the market expects the underlying to move by expiration.
If price stays within the EM range, neutral strategies (Iron Condors) tend to profit.
"""

from typing import Optional, Tuple
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


def calculate_expected_move(
    atm_call_price: float,
    atm_put_price: float,
    underlying_price: float,
) -> Tuple[float, float]:
    """
    Calculate Expected Move from ATM straddle premium.

    Args:
        atm_call_price: ATM call option price (mid)
        atm_put_price: ATM put option price (mid)
        underlying_price: Current underlying price

    Returns:
        Tuple of (expected_move_dollars, expected_move_percent)
    """
    if underlying_price <= 0:
        return (0.0, 0.0)

    em_dollars = atm_call_price + atm_put_price
    em_percent = (em_dollars / underlying_price) * 100

    return (em_dollars, em_percent)


def calculate_em_boundaries(
    underlying_price: float,
    expected_move: float,
) -> Tuple[float, float]:
    """
    Calculate the Expected Move price boundaries.

    Args:
        underlying_price: Current underlying price
        expected_move: Expected move in dollars

    Returns:
        Tuple of (lower_boundary, upper_boundary)
    """
    lower = underlying_price - expected_move
    upper = underlying_price + expected_move
    return (lower, upper)


def get_price_vs_em(
    current_price: float,
    open_price: float,
    expected_move: float,
) -> float:
    """
    Calculate how far price has moved relative to the Expected Move.

    Returns a value between 0 and 1+ where:
    - 0.0 = Price unchanged from open
    - 0.5 = Price moved 50% of expected move
    - 1.0 = Price at expected move boundary
    - >1.0 = Price exceeded expected move

    Args:
        current_price: Current underlying price
        open_price: Market open price
        expected_move: Expected move in dollars

    Returns:
        Ratio of actual move to expected move (0.0 to 1.0+)
    """
    if expected_move <= 0:
        return 0.0

    actual_move = abs(current_price - open_price)
    return actual_move / expected_move


class ExpectedMoveCalculator:
    """
    Calculate and track Expected Move throughout the trading day.

    Designed to be initialized at market open with the ATM straddle premium,
    then queried throughout the day to check if price is within the EM range.
    """

    def __init__(self):
        self.open_price: Optional[float] = None
        self.expected_move: Optional[float] = None
        self.em_percent: Optional[float] = None
        self.lower_boundary: Optional[float] = None
        self.upper_boundary: Optional[float] = None
        self.calculation_time: Optional[datetime] = None

    def set_expected_move(
        self,
        underlying_price: float,
        atm_call_price: float,
        atm_put_price: float,
        calculation_time: Optional[datetime] = None,
    ) -> None:
        """
        Initialize Expected Move for the day.

        Call this at market open with ATM straddle prices.

        Args:
            underlying_price: Underlying price at open
            atm_call_price: ATM call option price (mid)
            atm_put_price: ATM put option price (mid)
            calculation_time: Time of calculation (for logging)
        """
        self.open_price = underlying_price
        self.expected_move, self.em_percent = calculate_expected_move(
            atm_call_price, atm_put_price, underlying_price
        )
        self.lower_boundary, self.upper_boundary = calculate_em_boundaries(
            underlying_price, self.expected_move
        )
        self.calculation_time = calculation_time or datetime.now()

        logger.info(
            f"Expected Move set: ${self.expected_move:.2f} ({self.em_percent:.2f}%) "
            f"Range: ${self.lower_boundary:.2f} - ${self.upper_boundary:.2f}"
        )

    def is_within_em(self, current_price: float) -> bool:
        """
        Check if current price is within the Expected Move range.

        Args:
            current_price: Current underlying price

        Returns:
            True if price is within EM boundaries
        """
        if self.lower_boundary is None or self.upper_boundary is None:
            return True  # Default to True if EM not set

        return self.lower_boundary <= current_price <= self.upper_boundary

    def get_em_ratio(self, current_price: float) -> float:
        """
        Get how far price has moved relative to Expected Move.

        Args:
            current_price: Current underlying price

        Returns:
            Ratio from 0.0 (at open) to 1.0+ (at/beyond EM boundary)
        """
        if self.open_price is None or self.expected_move is None:
            return 0.0

        return get_price_vs_em(current_price, self.open_price, self.expected_move)

    def get_em_info(self) -> dict:
        """
        Get all Expected Move information as a dictionary.

        Returns:
            Dict with all EM data
        """
        return {
            "open_price": self.open_price,
            "expected_move": self.expected_move,
            "em_percent": self.em_percent,
            "lower_boundary": self.lower_boundary,
            "upper_boundary": self.upper_boundary,
            "calculation_time": self.calculation_time,
        }

    def reset(self) -> None:
        """Reset Expected Move for a new trading day."""
        self.open_price = None
        self.expected_move = None
        self.em_percent = None
        self.lower_boundary = None
        self.upper_boundary = None
        self.calculation_time = None


def estimate_em_from_price_data(
    highs: list[float],
    lows: list[float],
    lookback: int = 20,
) -> float:
    """
    Estimate Expected Move from historical price range.

    Fallback method when ATM straddle data is not available.
    Uses average daily range over lookback period.

    Args:
        highs: List of high prices
        lows: List of low prices
        lookback: Number of days to average

    Returns:
        Estimated expected move in dollars
    """
    if len(highs) < lookback or len(lows) < lookback:
        return 0.0

    daily_ranges = []
    for i in range(-lookback, 0):
        daily_range = highs[i] - lows[i]
        daily_ranges.append(daily_range)

    return sum(daily_ranges) / len(daily_ranges)
