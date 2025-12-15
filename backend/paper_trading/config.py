"""
Paper Trading Configuration for ORB Strategy.

Contains all validated parameters for live paper trading.
"""

from dataclasses import dataclass, field
from datetime import time, date
from typing import Optional
import os


@dataclass
class PaperTradingConfig:
    """
    Configuration for ORB paper trading.

    All times are in Eastern Time (ET) unless otherwise noted.
    """

    # === Symbol Settings ===
    underlying_symbol: str = "SPY"

    # === ORB Strategy Parameters (Validated) ===
    range_minutes: int = 60  # 9:30-10:30 AM ET
    breakout_buffer: float = 0.0  # No buffer, trade on close
    require_close: bool = True  # Require bar CLOSE above/below range

    # === Exit Parameters (Validated from backtest) ===
    profit_target_pct: float = 0.30  # 30% profit target
    stop_loss_pct: float = 0.20  # 20% stop loss (validated - NOT 6%!)

    # === Time Parameters ===
    # All times in ET (Eastern Time)
    market_open_time: time = field(default_factory=lambda: time(9, 30))
    range_end_time: time = field(default_factory=lambda: time(10, 30))
    entry_cutoff_time: time = field(default_factory=lambda: time(13, 0))  # 1:00 PM ET (validated)
    force_exit_time: time = field(default_factory=lambda: time(15, 50))  # 3:50 PM ET
    market_close_time: time = field(default_factory=lambda: time(16, 0))

    # UTC equivalents (ET + 5 during EST, ET + 4 during EDT)
    entry_cutoff_hour_utc: int = 18  # 1 PM ET = 18:00 UTC (validated)

    # === Position Sizing ===
    contracts_per_trade: int = 1
    max_positions: int = 1  # Only 1 position at a time

    # === Risk Controls ===
    daily_loss_limit: float = 200.0  # $200 = 2% of $10k
    account_size: float = 10000.0  # For percentage calculations

    # === Option Selection ===
    strike_offset: float = 0.5  # Slightly OTM (validated - better risk/reward)
    min_delta: float = 0.40  # Minimum delta for option selection
    max_spread_pct: float = 0.10  # Max bid-ask spread as % of mid

    # === Range Filters ===
    min_range_size: float = 1.0  # Minimum $1 range to trade
    max_range_size: float = 5.0  # Maximum $5 range to trade

    # === Order Settings ===
    use_market_orders_for_entry: bool = True  # Speed over price
    use_limit_orders_for_exit: bool = True  # Profit target as limit
    order_timeout_seconds: int = 30  # Cancel unfilled limit orders after

    # === Logging ===
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    trade_log_csv: bool = True

    # === Data Settings ===
    bar_timeframe: str = "1Min"  # 1-minute bars for ORB
    price_poll_interval: float = 1.0  # Seconds between price checks

    # === Environment ===
    alpaca_paper: bool = True  # ALWAYS paper trading

    @classmethod
    def from_env(cls) -> "PaperTradingConfig":
        """
        Create config from environment variables.

        Environment variables (all optional, defaults used if not set):
            PAPER_TRADING_SYMBOL: Underlying symbol (default: SPY)
            PAPER_TRADING_CONTRACTS: Contracts per trade (default: 1)
            PAPER_TRADING_DAILY_LOSS_LIMIT: Daily loss limit in $ (default: 200)
            PAPER_TRADING_PROFIT_TARGET_PCT: Profit target % (default: 0.30)
            PAPER_TRADING_STOP_LOSS_PCT: Stop loss % (default: 0.20)
        """
        return cls(
            underlying_symbol=os.getenv("PAPER_TRADING_SYMBOL", "SPY"),
            contracts_per_trade=int(os.getenv("PAPER_TRADING_CONTRACTS", "1")),
            daily_loss_limit=float(os.getenv("PAPER_TRADING_DAILY_LOSS_LIMIT", "200")),
            profit_target_pct=float(os.getenv("PAPER_TRADING_PROFIT_TARGET_PCT", "0.30")),
            stop_loss_pct=float(os.getenv("PAPER_TRADING_STOP_LOSS_PCT", "0.20")),
        )


@dataclass
class TradingSchedule:
    """
    Trading schedule configuration.

    Handles market hours, holidays, and trading windows.
    """

    # Days of week to trade (0=Monday, 4=Friday)
    trading_days: list = field(default_factory=lambda: [0, 1, 2, 3, 4])

    # US Market Holidays 2024-2025 (dates when market is CLOSED)
    holidays: list = field(default_factory=lambda: [
        # 2024
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # MLK Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving
        date(2024, 12, 25), # Christmas
        # 2025
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # MLK Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving
        date(2025, 12, 25), # Christmas
    ])

    # Early close days (1 PM ET close)
    early_close_days: list = field(default_factory=lambda: [
        date(2024, 7, 3),   # Day before July 4th
        date(2024, 11, 29), # Day after Thanksgiving
        date(2024, 12, 24), # Christmas Eve
        date(2025, 7, 3),
        date(2025, 11, 28),
        date(2025, 12, 24),
    ])

    def is_trading_day(self, dt: date) -> bool:
        """Check if given date is a trading day."""
        # Check if weekend
        if dt.weekday() not in self.trading_days:
            return False
        # Check if holiday
        if dt in self.holidays:
            return False
        return True

    def is_early_close(self, dt: date) -> bool:
        """Check if given date is an early close day."""
        return dt in self.early_close_days


# Default configuration instance
DEFAULT_CONFIG = PaperTradingConfig()
DEFAULT_SCHEDULE = TradingSchedule()
