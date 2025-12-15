"""
VWAP Reversion Strategy.

Trades mean reversion to VWAP when price deviates significantly.
"""

from strategies.vwap_reversion.strategy import VWAPReversionStrategy
from strategies.vwap_reversion.config import VWAPReversionConfig

__all__ = ["VWAPReversionStrategy", "VWAPReversionConfig"]
