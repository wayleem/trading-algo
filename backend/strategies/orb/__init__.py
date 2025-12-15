"""
ORB (Opening Range Breakout) Strategy.

Trades breakouts from the first 30-60 minute range after market open.
"""

from strategies.orb.strategy import ORBStrategy
from strategies.orb.config import ORBConfig

__all__ = ["ORBStrategy", "ORBConfig"]
