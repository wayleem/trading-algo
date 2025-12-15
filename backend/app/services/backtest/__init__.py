"""Backtest module for historical strategy simulation.

This module provides a modular backtesting framework for RSI-based
options trading strategies.

Usage:
    from app.services.backtest import BacktestService, BacktestConfig

    config = BacktestConfig(
        symbol="SPY",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        profit_target_dollars=0.50,
        stop_loss_pct=0.40,
    )

    service = BacktestService()
    result = await service.run_backtest(config)

Module Structure:
    - config.py: BacktestConfig dataclass with all configuration options
    - models.py: SimulatedTrade dataclass for tracking trades
    - option_pricing.py: ThetaData integration and price estimation
    - exit_strategy.py: Pluggable exit strategies (stop loss, profit target)
    - position_manager.py: Trade closing and P&L calculation
    - simulator.py: Core simulation loops
    - metrics.py: Performance metrics calculation
    - service.py: Main BacktestService orchestrator
"""

# Main exports for backward compatibility
from .config import BacktestConfig
from .models import SimulatedTrade
from .service import BacktestService

# Additional exports for advanced usage
from .exit_strategy import (
    ExitStrategy,
    StandardExitStrategy,
    ScaledDollarExitStrategy,
    PercentageExitStrategy,
    RSIConvergenceExitStrategy,
    get_exit_strategy,
)
from .metrics import MetricsCalculator, calculate_holding_periods, calculate_exit_breakdown
from .option_pricing import OptionPricingService
from .position_manager import PositionManager
from .simulator import TradeSimulator

__all__ = [
    # Main exports
    "BacktestConfig",
    "BacktestService",
    "SimulatedTrade",
    # Exit strategies
    "ExitStrategy",
    "StandardExitStrategy",
    "ScaledDollarExitStrategy",
    "PercentageExitStrategy",
    "RSIConvergenceExitStrategy",
    "get_exit_strategy",
    # Components
    "MetricsCalculator",
    "OptionPricingService",
    "PositionManager",
    "TradeSimulator",
    # Utility functions
    "calculate_holding_periods",
    "calculate_exit_breakdown",
]
