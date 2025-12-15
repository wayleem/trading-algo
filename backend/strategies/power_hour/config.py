"""
Power Hour Momentum Strategy configuration.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class PowerHourConfig(StrategyConfig):
    """
    Power Hour Momentum strategy configuration.

    Trades momentum in the last 90 minutes when trends tend to accelerate.

    Key Parameters:
        power_hour_start_utc: Hour (UTC) when power hour begins
        trend_lookback: Bars to determine intraday trend direction
        momentum_threshold: Minimum % move to confirm trend
    """

    # === Time Window ===
    power_hour_start_utc: int = 19  # 2:30 PM ET = 19:30 UTC
    power_hour_start_minute: int = 30  # Start at 2:30 PM ET

    # === Trend Detection ===
    trend_lookback: int = 30  # Bars to measure day's trend
    momentum_threshold: float = 0.20  # 0.20% move to confirm trend
    use_vwap_trend: bool = True  # Use VWAP to confirm trend direction

    # === Entry Confirmation ===
    require_volume_surge: bool = True  # Volume must be elevated
    volume_surge_threshold: float = 1.3  # 1.3x average volume
    require_new_high_low: bool = True  # Price making new intraday high/low

    # === Exit Settings ===
    profit_target_pct: float = 0.25  # 25% profit target (trends run)
    stop_loss_pct: float = 0.15  # 15% stop loss
    hold_to_close: bool = False  # Option to hold until market close

    # === Position Sizing ===
    scale_with_trend_strength: bool = False  # Stronger trend = larger position
