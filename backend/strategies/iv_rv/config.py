"""
IV vs RV (Implied vs Realized Volatility) Strategy configuration.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class IvRvConfig(StrategyConfig):
    """
    IV vs RV strategy configuration.

    Trades based on disparity between implied and realized volatility.

    Key Parameters:
        rv_lookback: Bars to calculate realized volatility
        iv_rv_threshold: Minimum ratio difference to trigger trade
    """

    # === Volatility Calculation ===
    rv_lookback: int = 50  # Bars to calculate realized volatility
    annualization_factor: float = 252.0  # Trading days per year

    # === Entry Conditions ===
    iv_premium_threshold: float = 1.3  # IV > 1.3x RV = overpriced (sell vol / buy puts on weakness)
    iv_discount_threshold: float = 0.8  # IV < 0.8x RV = underpriced (buy vol / buy calls on strength)

    # === Momentum Filter ===
    require_momentum: bool = True
    momentum_threshold: float = 0.10  # % move to confirm direction
    momentum_lookback: int = 10

    # === Entry Timing ===
    entry_start_minutes: int = 30  # Wait for IV to settle
    entry_cutoff_hour_utc: int = 18

    # === Exit Settings ===
    profit_target_pct: float = 0.20
    stop_loss_pct: float = 0.15

    # === Trade Management ===
    max_trades_per_day: int = 1
