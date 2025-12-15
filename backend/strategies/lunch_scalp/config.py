"""
Lunch Scalp Strategy configuration.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class LunchScalpConfig(StrategyConfig):
    """
    Lunch Scalp strategy configuration.

    Trades mean reversion during the low-volatility lunch hour.

    Key Parameters:
        lunch_start_utc: Hour (UTC) when lunch window begins
        lunch_end_utc: Hour (UTC) when lunch window ends
        vwap_deviation_pct: % deviation from VWAP to trigger entry
    """

    # === Time Window ===
    lunch_start_utc: int = 16  # 11:30 AM ET = 16:30 UTC
    lunch_start_minute: int = 30
    lunch_end_utc: int = 18  # 1:30 PM ET = 18:30 UTC
    lunch_end_minute: int = 30

    # === Mean Reversion Parameters ===
    vwap_deviation_pct: float = 0.10  # 0.10% deviation from VWAP
    require_oversold_overbought: bool = False  # Use RSI confirmation
    rsi_oversold_threshold: float = 35.0
    rsi_overbought_threshold: float = 65.0

    # === Volume Filter ===
    require_low_volume: bool = True  # Prefer low volume periods
    volume_below_threshold: float = 0.8  # Volume below 80% of average

    # === Exit Settings ===
    profit_target_pct: float = 0.10  # 10% quick scalp
    stop_loss_pct: float = 0.08  # Tight stop for choppy markets

    # === Position Sizing ===
    max_trades_per_session: int = 3  # Limit trades during lunch
