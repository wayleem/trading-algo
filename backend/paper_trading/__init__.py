"""
Paper Trading Module for ORB Strategy.

This module provides live paper trading capabilities for the
Opening Range Breakout (ORB) strategy using Alpaca's paper trading API.

Components:
- config: Trading configuration and schedule
- position_manager: Position state and risk management
- order_manager: Order submission and bracket order logic
- trade_logger: Comprehensive logging to console, file, and CSV
- paper_trader: Main orchestrator

Usage:
    python paper_trading/paper_trader.py

Environment Variables:
    ALPACA_API_KEY: Alpaca API key
    ALPACA_SECRET_KEY: Alpaca secret key
"""

from paper_trading.config import (
    PaperTradingConfig,
    TradingSchedule,
    DEFAULT_CONFIG,
    DEFAULT_SCHEDULE,
)
from paper_trading.position_manager import (
    PositionManager,
    Position,
    PositionState,
    ExitReason,
    DailyStats,
)
from paper_trading.order_manager import (
    OrderManager,
    ManagedOrder,
    OrderType,
    OrderPurpose,
)
from paper_trading.trade_logger import (
    TradeLogger,
    setup_logging,
)
from paper_trading.paper_trader import PaperTrader

__all__ = [
    # Config
    "PaperTradingConfig",
    "TradingSchedule",
    "DEFAULT_CONFIG",
    "DEFAULT_SCHEDULE",
    # Position Management
    "PositionManager",
    "Position",
    "PositionState",
    "ExitReason",
    "DailyStats",
    # Order Management
    "OrderManager",
    "ManagedOrder",
    "OrderType",
    "OrderPurpose",
    # Logging
    "TradeLogger",
    "setup_logging",
    # Main
    "PaperTrader",
]
