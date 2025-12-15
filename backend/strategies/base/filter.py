"""
Entry filters for strategies.

Filters provide pluggable entry conditions that can be applied
after signal generation. Strategies compose filters to add
time-based, volatility-based, or other entry restrictions.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone, time
from typing import Optional


class EntryFilter(ABC):
    """
    Abstract base class for entry filters.

    Filters can reject signals based on:
    - Time of day
    - Volatility conditions (IV rank)
    - Market regime
    - Technical indicators
    - etc.

    Filters are applied after signal generation but before trade entry.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Filter identifier for logging and debugging."""
        pass

    @abstractmethod
    def should_allow_entry(
        self,
        signal_type: str,
        bar: dict,
        timestamp: datetime,
        **kwargs,
    ) -> bool:
        """
        Check if entry should be allowed.

        Args:
            signal_type: BUY_CALL, BUY_PUT, etc.
            bar: Current OHLCV bar
            timestamp: Current timestamp
            **kwargs: Additional context (e.g., iv_rank, adx)

        Returns:
            True if entry is allowed, False to reject
        """
        pass

    def get_rejection_reason(self) -> str:
        """
        Return reason for last rejection.

        Override to provide detailed rejection messages.
        """
        return f"Blocked by {self.name}"


class TimeWindowFilter(EntryFilter):
    """
    Filter entries to specific time windows.

    Only allows entries within the specified ET time window.
    Useful for strategies like morning fade (10:00-10:30 entry).
    """

    def __init__(
        self,
        start_hour_et: int,
        start_minute_et: int = 0,
        end_hour_et: int = 16,
        end_minute_et: int = 0,
    ):
        """
        Initialize time window filter.

        Args:
            start_hour_et: Start hour (ET, 0-23)
            start_minute_et: Start minute (0-59)
            end_hour_et: End hour (ET, 0-23)
            end_minute_et: End minute (0-59)
        """
        self._start_hour = start_hour_et
        self._start_minute = start_minute_et
        self._end_hour = end_hour_et
        self._end_minute = end_minute_et
        self._start_time = time(start_hour_et, start_minute_et)
        self._end_time = time(end_hour_et, end_minute_et)
        self._last_rejection_reason = ""

    @property
    def name(self) -> str:
        return f"time_window_{self._start_hour}:{self._start_minute:02d}-{self._end_hour}:{self._end_minute:02d}"

    def should_allow_entry(
        self,
        signal_type: str,
        bar: dict,
        timestamp: datetime,
        **kwargs,
    ) -> bool:
        """Check if timestamp is within allowed time window."""
        et_time = self._to_et(timestamp)
        current_time = et_time.time() if hasattr(et_time, "time") else et_time

        in_window = self._start_time <= current_time <= self._end_time

        if not in_window:
            self._last_rejection_reason = (
                f"Time {current_time.strftime('%H:%M')} outside window "
                f"{self._start_time.strftime('%H:%M')}-{self._end_time.strftime('%H:%M')}"
            )

        return in_window

    def get_rejection_reason(self) -> str:
        return self._last_rejection_reason or super().get_rejection_reason()

    def _to_et(self, timestamp: datetime) -> datetime:
        """Convert timestamp to Eastern Time."""
        # Handle naive and aware datetimes
        if timestamp.tzinfo is None:
            # Assume UTC for naive datetimes
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # EST offset (simplified - production should handle DST)
        et_offset = timezone(timedelta(hours=-5))
        return timestamp.astimezone(et_offset)


class IVRankFilter(EntryFilter):
    """
    Filter entries based on IV rank threshold.

    Controls when to trade based on implied volatility percentile:
    - High IV rank (>50): Premium is rich, good for selling
    - Low IV rank (<50): Premium is cheap, good for buying

    Can be configured to require IV rank above or below threshold.
    """

    def __init__(
        self,
        min_iv_rank: float = 0.0,
        max_iv_rank: float = 100.0,
        require_valid: bool = False,
    ):
        """
        Initialize IV rank filter.

        Args:
            min_iv_rank: Minimum IV rank to allow entry (0-100)
            max_iv_rank: Maximum IV rank to allow entry (0-100)
            require_valid: If True, reject when IV rank is unavailable
        """
        self._min_iv_rank = min_iv_rank
        self._max_iv_rank = max_iv_rank
        self._require_valid = require_valid
        self._last_rejection_reason = ""

    @property
    def name(self) -> str:
        return f"iv_rank_{self._min_iv_rank:.0f}-{self._max_iv_rank:.0f}"

    def should_allow_entry(
        self,
        signal_type: str,
        bar: dict,
        timestamp: datetime,
        **kwargs,
    ) -> bool:
        """
        Check if IV rank allows entry.

        Expects iv_rank in kwargs. If not provided and require_valid=False,
        allows entry (permissive behavior).

        Args:
            signal_type: Signal type (unused)
            bar: Current bar (unused)
            timestamp: Current timestamp (unused)
            **kwargs: Must contain 'iv_rank' key

        Returns:
            True if IV rank is within range, False otherwise
        """
        iv_rank = kwargs.get("iv_rank")

        # Handle missing IV rank
        if iv_rank is None:
            if self._require_valid:
                self._last_rejection_reason = "IV rank not available"
                return False
            return True  # Permissive: allow if no IV data

        # Check range
        in_range = self._min_iv_rank <= iv_rank <= self._max_iv_rank

        if not in_range:
            self._last_rejection_reason = (
                f"IV rank {iv_rank:.1f}% outside range "
                f"[{self._min_iv_rank:.0f}%-{self._max_iv_rank:.0f}%]"
            )

        return in_range

    def get_rejection_reason(self) -> str:
        return self._last_rejection_reason or super().get_rejection_reason()


class CompositeFilter(EntryFilter):
    """
    Combines multiple filters with AND logic.

    Entry is allowed only if ALL filters allow it.
    """

    def __init__(self, filters: list):
        """
        Initialize composite filter.

        Args:
            filters: List of EntryFilter instances
        """
        self._filters = filters
        self._last_rejection_reason = ""

    @property
    def name(self) -> str:
        filter_names = "+".join(f.name for f in self._filters)
        return f"composite({filter_names})"

    def should_allow_entry(
        self,
        signal_type: str,
        bar: dict,
        timestamp: datetime,
        **kwargs,
    ) -> bool:
        """Check all filters, return False if any rejects."""
        for f in self._filters:
            if not f.should_allow_entry(signal_type, bar, timestamp, **kwargs):
                self._last_rejection_reason = f.get_rejection_reason()
                return False
        return True

    def get_rejection_reason(self) -> str:
        return self._last_rejection_reason or super().get_rejection_reason()


class ExitTimeFilter(EntryFilter):
    """
    Filter that tracks forced exit time.

    Used by strategies like morning fade that require exit by a specific time.
    Blocks new entries close to exit time.
    """

    def __init__(
        self,
        exit_hour_et: int,
        exit_minute_et: int = 0,
        buffer_minutes: int = 30,
    ):
        """
        Initialize exit time filter.

        Args:
            exit_hour_et: Exit hour (ET)
            exit_minute_et: Exit minute
            buffer_minutes: Block entries this many minutes before exit
        """
        self._exit_hour = exit_hour_et
        self._exit_minute = exit_minute_et
        self._buffer_minutes = buffer_minutes
        self._exit_time = time(exit_hour_et, exit_minute_et)
        self._last_rejection_reason = ""

    @property
    def name(self) -> str:
        return f"exit_time_{self._exit_hour}:{self._exit_minute:02d}"

    def should_allow_entry(
        self,
        signal_type: str,
        bar: dict,
        timestamp: datetime,
        **kwargs,
    ) -> bool:
        """Block entries that are too close to exit time."""
        et_time = self._to_et(timestamp)
        current_time = et_time.time() if hasattr(et_time, "time") else et_time

        # Calculate cutoff time (exit_time - buffer)
        exit_dt = datetime.combine(et_time.date(), self._exit_time)
        cutoff_dt = exit_dt - timedelta(minutes=self._buffer_minutes)
        cutoff_time = cutoff_dt.time()

        # Reject if after cutoff
        if current_time >= cutoff_time:
            self._last_rejection_reason = (
                f"Too close to exit time ({self._exit_time.strftime('%H:%M')})"
            )
            return False

        return True

    def get_rejection_reason(self) -> str:
        return self._last_rejection_reason or super().get_rejection_reason()

    def _to_et(self, timestamp: datetime) -> datetime:
        """Convert timestamp to Eastern Time."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        et_offset = timezone(timedelta(hours=-5))
        return timestamp.astimezone(et_offset)
