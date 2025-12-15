"""
Theta/Gamma Ratio Strategy configuration.
"""

from dataclasses import dataclass
from strategies.base.config import StrategyConfig


@dataclass
class ThetaGammaConfig(StrategyConfig):
    """
    Theta/Gamma Ratio strategy configuration.

    Trades when theta/gamma ratio favors directional bets.

    Key Parameters:
        min_theta_gamma_ratio: Minimum |theta|/gamma ratio for entry
        max_gamma: Maximum gamma exposure allowed
    """

    # === Greeks Thresholds ===
    # High |theta|/gamma ratio means theta decay is significant relative to gamma risk
    min_theta_gamma_ratio: float = 0.05  # |theta|/gamma > 0.05
    max_gamma: float = 0.10  # Avoid excessive gamma exposure

    # === Entry Conditions ===
    momentum_threshold: float = 0.12  # % move to trigger entry
    momentum_lookback: int = 10
    entry_start_minutes: int = 30  # Wait for Greeks to settle
    entry_cutoff_hour_utc: int = 17  # Earlier cutoff for theta-focused trades

    # === Exit Settings ===
    profit_target_pct: float = 0.15  # Quicker exits for theta plays
    stop_loss_pct: float = 0.12

    # === Trade Management ===
    max_trades_per_day: int = 2
    prefer_high_theta: bool = True  # Prefer options with high theta
