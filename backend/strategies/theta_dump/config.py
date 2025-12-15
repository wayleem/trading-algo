"""
Late Day Momentum Strategy configuration (3PM Theta Dump).
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class ThetaDumpConfig(StrategyConfig):
    """
    Late Day Momentum strategy configuration.

    Trades directional momentum after 3PM when theta decay accelerates.

    Key Parameters:
        late_day_start_utc: Hour (UTC) when late day trading begins
        momentum_threshold: Minimum % move to confirm direction
    """

    # === Time Window ===
    late_day_start_utc: int = 20  # 3:00 PM ET = 20:00 UTC
    late_day_start_minute: int = 0

    # === Momentum Detection ===
    momentum_lookback: int = 15  # Bars to measure recent momentum
    momentum_threshold: float = 0.12  # 0.12% move to confirm direction

    # === Trend Confirmation ===
    use_vwap_filter: bool = True  # Price should be on correct side of VWAP
    require_acceleration: bool = True  # Momentum accelerating

    # === Exit Settings ===
    profit_target_pct: float = 0.15  # 15% quick profit
    stop_loss_pct: float = 0.10  # Tight stop

    # === Trade Management ===
    max_trades_per_day: int = 1  # One trade late day
