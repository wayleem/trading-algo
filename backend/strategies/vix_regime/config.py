"""
Volatility Regime Strategy configuration.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class VixRegimeConfig(StrategyConfig):
    """
    Volatility Regime strategy configuration.

    Adapts trading based on realized volatility levels.

    Key Parameters:
        volatility_lookback: Bars to calculate realized volatility
        high_vol_threshold: ATR % threshold for high volatility
        low_vol_threshold: ATR % threshold for low volatility
    """

    # === Volatility Measurement ===
    volatility_lookback: int = 50  # Bars to measure volatility
    high_vol_threshold: float = 0.15  # ATR% above this is high volatility
    low_vol_threshold: float = 0.08  # ATR% below this is low volatility

    # === High Volatility Settings ===
    high_vol_profit_target: float = 0.30  # Wider target in high vol
    high_vol_stop_loss: float = 0.20  # Wider stop in high vol
    high_vol_momentum_threshold: float = 0.20  # Need bigger moves

    # === Low Volatility Settings ===
    low_vol_profit_target: float = 0.12  # Tighter target in low vol
    low_vol_stop_loss: float = 0.10  # Tighter stop in low vol
    low_vol_momentum_threshold: float = 0.10  # Smaller moves trigger entry

    # === Normal Volatility Settings ===
    normal_vol_profit_target: float = 0.20
    normal_vol_stop_loss: float = 0.15
    normal_vol_momentum_threshold: float = 0.15

    # === Entry Settings ===
    entry_start_minutes: int = 15  # Wait 15 min after open
    max_trades_per_day: int = 2
