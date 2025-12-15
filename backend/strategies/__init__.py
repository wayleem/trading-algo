"""
Modular Strategy Framework for Marcus Trading Bot.

This package provides a pluggable architecture for testing different
trading strategies independently.

Usage:
    from strategies.registry import StrategyRegistry

    # List available strategies
    strategies = StrategyRegistry.list_strategies()

    # Get a specific strategy
    strategy = StrategyRegistry.get_strategy("morning_fade")

    # Run backtest
    result = await strategy.run_backtest(start_date, end_date)
"""

from strategies.registry import StrategyRegistry
from strategies.base.strategy import BaseStrategy
from strategies.base.config import StrategyConfig

__all__ = ["StrategyRegistry", "BaseStrategy", "StrategyConfig"]
