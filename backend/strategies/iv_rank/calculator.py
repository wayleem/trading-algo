"""
IV Rank Calculator for options premium filtering.

IV Rank measures where current IV sits relative to its historical range:
- IV Rank = 0%: Current IV equals the lowest IV in lookback period
- IV Rank = 50%: Current IV is at the median of the range
- IV Rank = 100%: Current IV equals the highest IV in lookback period

Higher IV Rank = premium is expensive relative to history = better for selling
"""

import json
import logging
from collections import deque
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class IVHistoryPoint:
    """Single day's IV observation."""

    observation_date: date
    atm_iv: float  # Average of ATM call/put IV (annualized, e.g., 0.20 = 20%)
    underlying_price: float
    atm_call_iv: Optional[float] = None
    atm_put_iv: Optional[float] = None

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "observation_date": self.observation_date.isoformat(),
            "atm_iv": self.atm_iv,
            "underlying_price": self.underlying_price,
            "atm_call_iv": self.atm_call_iv,
            "atm_put_iv": self.atm_put_iv,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "IVHistoryPoint":
        """Deserialize from dictionary."""
        return cls(
            observation_date=datetime.fromisoformat(data["observation_date"]).date(),
            atm_iv=data["atm_iv"],
            underlying_price=data["underlying_price"],
            atm_call_iv=data.get("atm_call_iv"),
            atm_put_iv=data.get("atm_put_iv"),
        )


@dataclass
class IVRankResult:
    """Result of IV Rank calculation."""

    iv_rank: float  # 0-100 percentile
    current_iv: float
    min_iv: float
    max_iv: float
    lookback_days: int
    history_days_available: int
    is_valid: bool = True
    reason: str = ""


class IVHistoryStore:
    """
    Stores and manages historical IV data for IV Rank calculation.

    Supports:
    - In-memory storage for backtest
    - JSON persistence for live trading
    - Rolling window maintenance
    """

    def __init__(
        self,
        symbol: str,
        lookback_days: int = 45,
        data_dir: Optional[Path] = None,
        persist: bool = False,
    ):
        """
        Initialize IV history store.

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            lookback_days: Number of trading days for IV Rank calculation
            data_dir: Directory for persistent storage
            persist: Whether to persist to file
        """
        self.symbol = symbol.upper()
        self.lookback_days = lookback_days
        self.persist = persist

        # Use deque for efficient rolling window (add buffer for safety)
        self._history: deque = deque(maxlen=lookback_days + 60)

        # Persistence setup
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "iv_history"
        self.data_dir = data_dir
        self._file_path = self.data_dir / f"{self.symbol}_iv_history.json"

        if persist:
            self._load_from_file()

    def add_observation(self, point: IVHistoryPoint) -> None:
        """
        Add a new IV observation.

        Avoids duplicates by checking existing dates.
        """
        existing_dates = {p.observation_date for p in self._history}

        if point.observation_date not in existing_dates:
            self._history.append(point)
            # Keep sorted by date
            self._history = deque(
                sorted(self._history, key=lambda x: x.observation_date),
                maxlen=self._history.maxlen,
            )

            if self.persist:
                self._save_to_file()

    def get_iv_history(self, as_of_date: date) -> List[IVHistoryPoint]:
        """
        Get IV history for lookback period ending on as_of_date.

        Args:
            as_of_date: Reference date (get history up to this date)

        Returns:
            List of IVHistoryPoint within lookback window
        """
        # Calculate cutoff date (~1.5x lookback to account for weekends/holidays)
        cutoff_date = as_of_date - timedelta(days=int(self.lookback_days * 1.5))

        # Filter to history within range
        relevant = [
            p for p in self._history
            if cutoff_date <= p.observation_date <= as_of_date
        ]

        # Return last N trading days
        return relevant[-self.lookback_days:]

    def get_latest_iv(self) -> Optional[IVHistoryPoint]:
        """Get most recent IV observation."""
        return self._history[-1] if self._history else None

    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()

    def __len__(self) -> int:
        """Return number of history points."""
        return len(self._history)

    def _load_from_file(self) -> None:
        """Load history from JSON file."""
        if not self._file_path.exists():
            return

        try:
            with open(self._file_path) as f:
                data = json.load(f)

            for item in data.get("history", []):
                point = IVHistoryPoint.from_dict(item)
                self._history.append(point)

            logger.info(f"Loaded {len(self._history)} IV history points for {self.symbol}")

        except Exception as e:
            logger.warning(f"Failed to load IV history: {e}")

    def _save_to_file(self) -> None:
        """Save history to JSON file."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "symbol": self.symbol,
            "lookback_days": self.lookback_days,
            "last_updated": datetime.now().isoformat(),
            "history": [p.to_dict() for p in self._history],
        }

        with open(self._file_path, "w") as f:
            json.dump(data, f, indent=2)


class IVRankCalculator:
    """
    Calculate IV Rank for filtering trade entries.

    IV Rank = (current_IV - min_IV) / (max_IV - min_IV) * 100

    Usage:
        calculator = IVRankCalculator("SPY", lookback_days=45)

        # Add historical observations
        calculator.add_iv_observation(date(2024, 1, 1), 0.15, 600.0)
        calculator.add_iv_observation(date(2024, 1, 2), 0.16, 601.0)
        # ... more observations

        # Calculate IV Rank
        result = calculator.calculate_iv_rank(0.18, date(2024, 2, 1))
        print(f"IV Rank: {result.iv_rank:.1f}%")
    """

    def __init__(
        self,
        symbol: str,
        lookback_days: int = 45,
        min_history_days: int = 20,
        persist: bool = False,
    ):
        """
        Initialize IV Rank calculator.

        Args:
            symbol: Underlying symbol
            lookback_days: Days of history for IV Rank calculation
            min_history_days: Minimum history required for valid IV Rank
            persist: Whether to persist history to file
        """
        self.symbol = symbol.upper()
        self.lookback_days = lookback_days
        self.min_history_days = min_history_days

        self._store = IVHistoryStore(
            symbol=symbol,
            lookback_days=lookback_days,
            persist=persist,
        )

    def calculate_iv_rank(
        self,
        current_iv: float,
        as_of_date: date,
    ) -> IVRankResult:
        """
        Calculate IV Rank for the given IV value.

        Args:
            current_iv: Current ATM IV (as decimal, e.g., 0.20 for 20%)
            as_of_date: Reference date for historical comparison

        Returns:
            IVRankResult with IV Rank and metadata
        """
        history = self._store.get_iv_history(as_of_date)

        # Validate we have enough history
        if len(history) < self.min_history_days:
            return IVRankResult(
                iv_rank=50.0,  # Default to middle
                current_iv=current_iv,
                min_iv=0.0,
                max_iv=0.0,
                lookback_days=self.lookback_days,
                history_days_available=len(history),
                is_valid=False,
                reason=f"Insufficient history: {len(history)}/{self.min_history_days} days",
            )

        # Handle edge cases
        if current_iv <= 0:
            return IVRankResult(
                iv_rank=0.0,
                current_iv=current_iv,
                min_iv=0.0,
                max_iv=0.0,
                lookback_days=self.lookback_days,
                history_days_available=len(history),
                is_valid=False,
                reason="Current IV is zero or negative",
            )

        # Extract valid IV values
        iv_values = [p.atm_iv for p in history if p.atm_iv > 0]

        if not iv_values:
            return IVRankResult(
                iv_rank=50.0,
                current_iv=current_iv,
                min_iv=0.0,
                max_iv=0.0,
                lookback_days=self.lookback_days,
                history_days_available=len(history),
                is_valid=False,
                reason="No valid IV values in history",
            )

        min_iv = min(iv_values)
        max_iv = max(iv_values)

        # Avoid division by zero
        if max_iv == min_iv:
            iv_rank = 50.0  # All IVs are the same
        else:
            iv_rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100
            # Clamp to 0-100 range (current_iv might be outside historical range)
            iv_rank = max(0.0, min(100.0, iv_rank))

        return IVRankResult(
            iv_rank=iv_rank,
            current_iv=current_iv,
            min_iv=min_iv,
            max_iv=max_iv,
            lookback_days=self.lookback_days,
            history_days_available=len(iv_values),
            is_valid=True,
            reason="",
        )

    def add_iv_observation(
        self,
        observation_date: date,
        atm_iv: float,
        underlying_price: float,
        atm_call_iv: Optional[float] = None,
        atm_put_iv: Optional[float] = None,
    ) -> None:
        """
        Add a new IV observation to history.

        Args:
            observation_date: Date of observation
            atm_iv: Average ATM IV (as decimal, e.g., 0.20)
            underlying_price: Underlying price at observation
            atm_call_iv: ATM call IV (optional)
            atm_put_iv: ATM put IV (optional)
        """
        point = IVHistoryPoint(
            observation_date=observation_date,
            atm_iv=atm_iv,
            underlying_price=underlying_price,
            atm_call_iv=atm_call_iv,
            atm_put_iv=atm_put_iv,
        )
        self._store.add_observation(point)

    def should_allow_entry(
        self,
        current_iv: float,
        as_of_date: date,
        threshold: float = 50.0,
        require_above: bool = True,
    ) -> Tuple[bool, IVRankResult]:
        """
        Check if IV Rank allows trade entry.

        Args:
            current_iv: Current ATM IV
            as_of_date: Reference date
            threshold: IV Rank threshold (0-100)
            require_above: If True, allow entry when IV Rank >= threshold
                          If False, allow entry when IV Rank <= threshold

        Returns:
            Tuple of (should_allow, IVRankResult)
        """
        result = self.calculate_iv_rank(current_iv, as_of_date)

        # If not valid, allow entry by default (don't block on missing data)
        if not result.is_valid:
            return (True, result)

        if require_above:
            should_allow = result.iv_rank >= threshold
        else:
            should_allow = result.iv_rank <= threshold

        return (should_allow, result)

    def get_history_stats(self) -> Dict:
        """Get statistics about IV history."""
        history = list(self._store._history)

        if not history:
            return {"count": 0, "date_range": None}

        iv_values = [p.atm_iv for p in history if p.atm_iv > 0]

        return {
            "count": len(history),
            "date_range": (
                history[0].observation_date.isoformat(),
                history[-1].observation_date.isoformat(),
            ),
            "iv_mean": sum(iv_values) / len(iv_values) if iv_values else 0,
            "iv_min": min(iv_values) if iv_values else 0,
            "iv_max": max(iv_values) if iv_values else 0,
        }

    def clear_history(self) -> None:
        """Clear all IV history."""
        self._store.clear()


async def build_iv_history_from_theta(
    calculator: IVRankCalculator,
    theta_client,
    symbol: str,
    start_date: date,
    end_date: date,
) -> int:
    """
    Build IV history from ThetaData for the date range.

    Fetches ATM IV for each trading day and stores in calculator.

    Args:
        calculator: IVRankCalculator instance
        theta_client: ThetaDataClient instance
        symbol: Symbol to fetch IV for
        start_date: Start of history range
        end_date: End of history range

    Returns:
        Number of days successfully added
    """
    days_added = 0
    current_date = start_date

    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        try:
            # Get ATM IV for this date
            atm_iv = await _fetch_atm_iv_for_date(
                theta_client, symbol, current_date
            )

            if atm_iv is not None and atm_iv > 0:
                # Get underlying price
                underlying_price = await theta_client.get_stock_quote_for_date(
                    symbol, current_date
                )

                calculator.add_iv_observation(
                    observation_date=current_date,
                    atm_iv=atm_iv,
                    underlying_price=underlying_price or 0.0,
                )
                days_added += 1

        except Exception as e:
            logger.debug(f"Failed to get IV for {current_date}: {e}")

        current_date += timedelta(days=1)

    logger.info(f"Built IV history: {days_added} days for {symbol}")
    return days_added


async def _fetch_atm_iv_for_date(
    theta_client,
    symbol: str,
    target_date: date,
) -> Optional[float]:
    """
    Fetch ATM IV for a specific date.

    Uses average of ATM call and put IV.
    """
    try:
        # Get expirations
        expirations = await theta_client.get_expirations(symbol)
        if not expirations:
            return None

        # Find expiration that was active on target_date
        valid_exp = None
        for exp in expirations:
            if exp >= target_date:
                valid_exp = exp
                break

        if not valid_exp:
            return None

        # Get strikes
        strikes = await theta_client.get_strikes(symbol, valid_exp)
        if not strikes:
            return None

        # Use middle strike as ATM approximation
        atm_strike = strikes[len(strikes) // 2]

        # Fetch call and put IV
        call_greeks = await theta_client.get_historical_greeks(
            symbol=symbol,
            expiration=valid_exp,
            strike=atm_strike,
            right="C",
            start_date=target_date,
            end_date=target_date,
        )

        put_greeks = await theta_client.get_historical_greeks(
            symbol=symbol,
            expiration=valid_exp,
            strike=atm_strike,
            right="P",
            start_date=target_date,
            end_date=target_date,
        )

        # Average call and put IV
        call_iv = call_greeks[0].implied_volatility if call_greeks else None
        put_iv = put_greeks[0].implied_volatility if put_greeks else None

        if call_iv and put_iv:
            return (call_iv + put_iv) / 2
        elif call_iv:
            return call_iv
        elif put_iv:
            return put_iv

        return None

    except Exception as e:
        logger.debug(f"Error fetching IV for {target_date}: {e}")
        return None
