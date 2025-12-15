"""
Trade Logger for Paper Trading.

Handles logging to console, file, and CSV trade log.
"""

import csv
import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict

from paper_trading.position_manager import Position, DailyStats, ExitReason


@dataclass
class TradeLogEntry:
    """A single trade log entry for CSV output."""
    date: str
    entry_time: str
    exit_time: str
    symbol: str
    option_type: str
    strike: float
    underlying_symbol: str
    quantity: int
    entry_price: float
    exit_price: float
    pnl_dollars: float
    pnl_percent: float
    exit_reason: str
    signal_reason: str
    range_high: float
    range_low: float
    range_size: float


@dataclass
class SignalLogEntry:
    """A signal log entry."""
    timestamp: str
    signal_type: str
    underlying_price: float
    range_high: float
    range_low: float
    reason: str
    action_taken: str


class TradeLogger:
    """
    Comprehensive logging for paper trading.

    Features:
    - Console logging with colors
    - File logging for detailed debug
    - CSV trade log for analysis
    - Daily summaries
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
        csv_enabled: bool = True,
    ):
        """
        Initialize trade logger.

        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_to_console: Enable console logging
            log_to_file: Enable file logging
            csv_enabled: Enable CSV trade log
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.csv_enabled = csv_enabled

        # Setup logging
        self.logger = self._setup_logger()

        # CSV tracking
        self.trade_entries: List[TradeLogEntry] = []
        self.signal_entries: List[SignalLogEntry] = []

        # Current session info
        self.session_date: Optional[date] = None
        self.session_start: Optional[datetime] = None

    def _setup_logger(self) -> logging.Logger:
        """Setup the main logger."""
        logger = logging.getLogger("paper_trader")
        logger.setLevel(self.log_level)
        logger.handlers = []  # Clear existing handlers

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        if self.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler (created per day)
        if self.log_to_file:
            today = datetime.now().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"paper_trader_{today}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Always detailed in file
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def start_session(self, trading_date: date) -> None:
        """Start a new trading session."""
        self.session_date = trading_date
        self.session_start = datetime.now()

        self.logger.info("=" * 60)
        self.logger.info(f"PAPER TRADING SESSION STARTED: {trading_date}")
        self.logger.info("=" * 60)

    def end_session(self, daily_stats: Optional[DailyStats] = None) -> None:
        """End the current trading session."""
        self.logger.info("=" * 60)
        self.logger.info("PAPER TRADING SESSION ENDED")

        if daily_stats:
            self.log_daily_summary(daily_stats)

        self.logger.info("=" * 60)

        # Write CSV if enabled
        if self.csv_enabled and self.trade_entries:
            self._write_trade_csv()

    def log_range_formed(
        self,
        range_high: float,
        range_low: float,
        range_size: float,
    ) -> None:
        """Log when opening range is formed."""
        self.logger.info(
            f"OPENING RANGE FORMED: "
            f"High=${range_high:.2f} | Low=${range_low:.2f} | "
            f"Size=${range_size:.2f}"
        )

    def log_signal(
        self,
        signal_type: str,
        underlying_price: float,
        range_high: float,
        range_low: float,
        reason: str,
        action_taken: str,
    ) -> None:
        """Log a trading signal."""
        self.logger.info(
            f"SIGNAL: {signal_type} | "
            f"SPY=${underlying_price:.2f} | "
            f"Range=[${range_low:.2f}-${range_high:.2f}] | "
            f"{reason}"
        )
        self.logger.info(f"  -> Action: {action_taken}")

        self.signal_entries.append(SignalLogEntry(
            timestamp=datetime.now().isoformat(),
            signal_type=signal_type,
            underlying_price=underlying_price,
            range_high=range_high,
            range_low=range_low,
            reason=reason,
            action_taken=action_taken,
        ))

    def log_order_submitted(
        self,
        order_type: str,
        symbol: str,
        side: str,
        quantity: int,
        price: Optional[float],
        order_id: str,
    ) -> None:
        """Log order submission."""
        price_str = f"@ ${price:.2f}" if price else "(MARKET)"
        self.logger.info(
            f"ORDER SUBMITTED: {side.upper()} {quantity}x {symbol} "
            f"{price_str} | Type: {order_type} | ID: {order_id}"
        )

    def log_order_filled(
        self,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        order_id: str,
    ) -> None:
        """Log order fill."""
        self.logger.info(
            f"ORDER FILLED: {side.upper()} {quantity}x {symbol} "
            f"@ ${fill_price:.2f} | ID: {order_id}"
        )

    def log_order_cancelled(self, order_id: str, reason: str) -> None:
        """Log order cancellation."""
        self.logger.info(f"ORDER CANCELLED: {order_id} | Reason: {reason}")

    def log_position_opened(
        self,
        symbol: str,
        option_type: str,
        strike: float,
        quantity: int,
        entry_price: float,
        profit_target: float,
        stop_loss: float,
    ) -> None:
        """Log position opening."""
        self.logger.info("-" * 50)
        self.logger.info(
            f"POSITION OPENED: {symbol} ({option_type.upper()}) "
            f"Strike=${strike:.2f}"
        )
        self.logger.info(
            f"  Entry: {quantity}x @ ${entry_price:.2f} "
            f"(${entry_price * quantity * 100:.2f} total)"
        )
        self.logger.info(
            f"  PT: ${profit_target:.2f} (+{(profit_target/entry_price-1)*100:.1f}%) | "
            f"SL: ${stop_loss:.2f} ({(stop_loss/entry_price-1)*100:.1f}%)"
        )
        self.logger.info("-" * 50)

    def log_position_update(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        unrealized_pnl: float,
        pnl_pct: float,
    ) -> None:
        """Log position price update (for debugging)."""
        self.logger.debug(
            f"Position Update: {symbol} | "
            f"Price: ${current_price:.2f} | "
            f"P&L: ${unrealized_pnl:.2f} ({pnl_pct:+.1f}%)"
        )

    def log_position_closed(self, position: Position) -> None:
        """Log position closing."""
        self.logger.info("-" * 50)
        self.logger.info(
            f"POSITION CLOSED: {position.option_symbol} | "
            f"Reason: {position.exit_reason.value if position.exit_reason else 'unknown'}"
        )
        self.logger.info(
            f"  Entry: ${position.entry_price:.2f} -> "
            f"Exit: ${position.exit_price:.2f}"
        )
        self.logger.info(
            f"  P&L: ${position.realized_pnl_total:.2f} "
            f"({position.realized_pnl_pct:+.1f}%)"
        )
        self.logger.info("-" * 50)

        # Add to trade entries for CSV
        self.trade_entries.append(TradeLogEntry(
            date=str(self.session_date),
            entry_time=position.entry_time.strftime("%H:%M:%S") if position.entry_time else "",
            exit_time=position.exit_time.strftime("%H:%M:%S") if position.exit_time else "",
            symbol=position.option_symbol,
            option_type=position.option_type,
            strike=position.strike,
            underlying_symbol=position.underlying_symbol,
            quantity=position.quantity,
            entry_price=position.entry_price,
            exit_price=position.exit_price or 0.0,
            pnl_dollars=position.realized_pnl_total,
            pnl_percent=position.realized_pnl_pct,
            exit_reason=position.exit_reason.value if position.exit_reason else "unknown",
            signal_reason=position.entry_signal_reason,
            range_high=position.range_high,
            range_low=position.range_low,
            range_size=position.range_high - position.range_low,
        ))

    def log_option_expired_worthless(
        self,
        symbol: str,
        strike: float,
        underlying_price: float,
        option_type: str,
        entry_price: float,
        quantity: int,
    ) -> None:
        """Log option expiring worthless (OTM at settlement)."""
        self.logger.warning("-" * 50)
        self.logger.warning(
            f"EXPIRED WORTHLESS: {symbol} ({option_type.upper()})"
        )
        self.logger.warning(
            f"  Strike: ${strike:.2f} | SPY: ${underlying_price:.2f}"
        )
        self.logger.warning(
            f"  Intrinsic Value: $0.00 (OTM)"
        )
        loss = entry_price * quantity * 100
        self.logger.warning(
            f"  Total Loss: ${loss:.2f} (100%)"
        )
        self.logger.warning("-" * 50)

    def log_circuit_breaker(self, current_loss: float, limit: float) -> None:
        """Log circuit breaker trigger."""
        self.logger.warning("=" * 60)
        self.logger.warning("CIRCUIT BREAKER TRIGGERED!")
        self.logger.warning(
            f"  Daily Loss: ${current_loss:.2f} exceeds limit ${limit:.2f}"
        )
        self.logger.warning("  Trading halted for the day")
        self.logger.warning("=" * 60)

    def log_force_exit_deadline(self) -> None:
        """Log force exit deadline reached."""
        self.logger.warning(
            "FORCE EXIT DEADLINE: 3:50 PM ET - Closing all positions"
        )

    def log_entry_cutoff(self) -> None:
        """Log entry cutoff time reached."""
        self.logger.info(
            "ENTRY CUTOFF: 2:00 PM ET - No new entries allowed"
        )

    def log_daily_summary(self, stats: DailyStats) -> None:
        """Log daily trading summary."""
        self.logger.info("")
        self.logger.info("=" * 50)
        self.logger.info("DAILY SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"  Date: {stats.date}")
        self.logger.info(f"  Trades: {stats.trades_executed}")
        self.logger.info(f"  Wins: {stats.winning_trades} | Losses: {stats.losing_trades}")
        self.logger.info(f"  Win Rate: {stats.win_rate:.1f}%")
        self.logger.info(f"  Total P&L: ${stats.total_pnl:.2f}")
        self.logger.info(f"  Max Drawdown: ${stats.max_drawdown:.2f}")
        self.logger.info(f"  Circuit Breaker: {'TRIGGERED' if stats.circuit_breaker_triggered else 'No'}")
        self.logger.info("=" * 50)

    def log_market_status(self, is_open: bool, next_event: str, event_time: datetime) -> None:
        """Log market status."""
        status = "OPEN" if is_open else "CLOSED"
        self.logger.info(
            f"Market Status: {status} | "
            f"Next {next_event}: {event_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )

    def log_waiting(self, reason: str, until: Optional[datetime] = None) -> None:
        """Log waiting status."""
        if until:
            self.logger.info(f"Waiting: {reason} (until {until.strftime('%H:%M:%S')})")
        else:
            self.logger.info(f"Waiting: {reason}")

    def log_error(self, message: str, exception: Optional[Exception] = None) -> None:
        """Log error."""
        if exception:
            self.logger.error(f"ERROR: {message} | {type(exception).__name__}: {exception}")
        else:
            self.logger.error(f"ERROR: {message}")

    def log_warning(self, message: str) -> None:
        """Log warning."""
        self.logger.warning(f"WARNING: {message}")

    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def _write_trade_csv(self) -> None:
        """Write trade entries to CSV file."""
        if not self.trade_entries:
            return

        csv_file = self.log_dir / f"trades_{self.session_date}.csv"

        # Check if file exists to determine if we need headers
        file_exists = csv_file.exists()

        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.trade_entries[0]).keys()))

            if not file_exists:
                writer.writeheader()

            for entry in self.trade_entries:
                writer.writerow(asdict(entry))

        self.logger.info(f"Trade log written to: {csv_file}")
        self.trade_entries = []  # Clear after writing

    def _write_signal_csv(self) -> None:
        """Write signal entries to CSV file."""
        if not self.signal_entries:
            return

        csv_file = self.log_dir / f"signals_{self.session_date}.csv"

        file_exists = csv_file.exists()

        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.signal_entries[0]).keys()))

            if not file_exists:
                writer.writeheader()

            for entry in self.signal_entries:
                writer.writerow(asdict(entry))

        self.signal_entries = []


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
) -> TradeLogger:
    """
    Setup logging for paper trading.

    Args:
        log_dir: Directory for log files
        log_level: Logging level

    Returns:
        TradeLogger instance
    """
    return TradeLogger(
        log_dir=log_dir,
        log_level=log_level,
        log_to_console=True,
        log_to_file=True,
        csv_enabled=True,
    )
