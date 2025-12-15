"""
Momentum Scalping Strategy.

Trades strong intraday momentum moves with tight profit targets.
"""

from strategies.momentum_scalp.strategy import MomentumScalpStrategy
from strategies.momentum_scalp.config import MomentumScalpConfig

__all__ = ["MomentumScalpStrategy", "MomentumScalpConfig"]
