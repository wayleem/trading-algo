"""
VWAP Fade Strategy.

Fades early morning deviations below VWAP, exploiting institutional VWAP targeting.
"""

from strategies.vwap_fade.strategy import VWAPFadeStrategy
from strategies.vwap_fade.config import VWAPFadeConfig

__all__ = ["VWAPFadeStrategy", "VWAPFadeConfig"]
