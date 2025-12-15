"""
Gap Fade Strategy.

Fades overnight gaps expecting mean reversion during the trading day.
"""

from strategies.gap_fade.strategy import GapFadeStrategy, calculate_directional_accuracy
from strategies.gap_fade.config import GapFadeConfig

__all__ = ["GapFadeStrategy", "GapFadeConfig", "calculate_directional_accuracy"]
