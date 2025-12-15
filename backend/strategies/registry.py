"""
Strategy registry for discovering and loading strategies.

Strategies are auto-discovered from the strategies/ directory.
Each strategy module must have a `strategy.py` file with a class
inheriting from BaseStrategy.
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, Type, List, Optional

from strategies.base.strategy import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Registry for discovering and loading strategies.

    Strategies are auto-discovered from subdirectories in the strategies/ folder.
    Each strategy directory should contain:
    - __init__.py
    - strategy.py (with class inheriting from BaseStrategy)
    - config.py (optional, for strategy-specific config)

    Usage:
        # Discover all strategies
        StrategyRegistry.discover_strategies()

        # List available strategies
        names = StrategyRegistry.list_strategies()

        # Get specific strategy
        strategy = StrategyRegistry.get_strategy("morning_fade")

        # Run backtest
        result = await strategy.run_backtest(start_date, end_date)
    """

    _strategies: Dict[str, Type[BaseStrategy]] = {}
    _initialized: bool = False

    @classmethod
    def discover_strategies(cls) -> None:
        """
        Discover all strategies in the strategies/ directory.

        Scans for subdirectories containing strategy.py files and
        loads any classes inheriting from BaseStrategy.
        """
        if cls._initialized:
            return

        strategies_dir = Path(__file__).parent

        # Directories to skip
        skip_dirs = {"base", "__pycache__"}

        for item in strategies_dir.iterdir():
            if not item.is_dir():
                continue
            if item.name.startswith("_") or item.name in skip_dirs:
                continue

            strategy_module = item / "strategy.py"
            if strategy_module.exists():
                cls._load_strategy_from_module(item.name)

        cls._initialized = True
        logger.info(f"Discovered {len(cls._strategies)} strategies: {list(cls._strategies.keys())}")

    @classmethod
    def _load_strategy_from_module(cls, module_name: str) -> None:
        """
        Load a strategy from its module.

        Args:
            module_name: Name of the strategy directory (e.g., "morning_fade")
        """
        try:
            module = importlib.import_module(f"strategies.{module_name}.strategy")

            # Find the strategy class (inherits from BaseStrategy)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseStrategy)
                    and attr is not BaseStrategy
                ):
                    # Instantiate to get the name
                    try:
                        instance = attr()
                        cls._strategies[instance.name] = attr
                        logger.debug(f"Loaded strategy: {instance.name} from {module_name}")
                    except Exception as e:
                        logger.warning(f"Could not instantiate {attr_name}: {e}")
                    break

        except Exception as e:
            logger.warning(f"Could not load strategy from {module_name}: {e}")

    @classmethod
    def get_strategy(cls, name: str) -> Optional[BaseStrategy]:
        """
        Get a strategy instance by name.

        Args:
            name: Strategy name (e.g., "morning_fade")

        Returns:
            Strategy instance or None if not found
        """
        cls.discover_strategies()

        strategy_cls = cls._strategies.get(name)
        if strategy_cls:
            return strategy_cls()
        return None

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        List all available strategy names.

        Returns:
            List of strategy names
        """
        cls.discover_strategies()
        return list(cls._strategies.keys())

    @classmethod
    def get_all_strategies(cls) -> List[BaseStrategy]:
        """
        Get instances of all registered strategies.

        Returns:
            List of strategy instances
        """
        cls.discover_strategies()
        return [cls() for cls in cls._strategies.values()]

    @classmethod
    def get_strategy_info(cls) -> List[Dict]:
        """
        Get information about all registered strategies.

        Returns:
            List of dicts with strategy name and description
        """
        cls.discover_strategies()
        info = []
        for strategy_cls in cls._strategies.values():
            try:
                instance = strategy_cls()
                info.append({
                    "name": instance.name,
                    "description": instance.description,
                    "class": strategy_cls.__name__,
                })
            except Exception:
                pass
        return info

    @classmethod
    def reset(cls) -> None:
        """
        Reset the registry (useful for testing).

        Clears all registered strategies and resets initialization state.
        """
        cls._strategies.clear()
        cls._initialized = False

    @classmethod
    def register(cls, strategy_cls: Type[BaseStrategy]) -> None:
        """
        Manually register a strategy class.

        Useful for testing or dynamic registration.

        Args:
            strategy_cls: Strategy class to register
        """
        instance = strategy_cls()
        cls._strategies[instance.name] = strategy_cls
        logger.debug(f"Manually registered strategy: {instance.name}")
