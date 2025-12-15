"""
Base configuration for strategies.

StrategyConfig provides common parameters shared across all strategies.
Strategy-specific configs inherit from this and add their own parameters.
"""

from dataclasses import dataclass, field, asdict
from datetime import date, time
from typing import Optional, Dict, Any


@dataclass
class StrategyConfig:
    """
    Base configuration for all strategies.

    Contains common parameters shared across strategies.
    Strategy-specific configs inherit and add their own parameters.
    """

    # === Core Settings ===
    symbol: str = "SPY"
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    # === Capital and Costs ===
    initial_capital: float = 10000.0
    commission_per_contract: float = 0.65

    # === Realistic Slippage Model ===
    # 0DTE options have wide bid-ask spreads (5-10% on cheap premiums)
    slippage_entry_pct: float = 0.03      # 3% on entry (paying ask vs mid)
    slippage_exit_pct: float = 0.03       # 3% on exit (receiving bid vs mid)
    slippage_stop_extra_pct: float = 0.02 # Extra 2% slippage on stop-loss orders
    min_slippage_dollars: float = 0.02    # Minimum $0.02 slippage per contract

    # === Position Sizing ===
    contract_multiplier: int = 1  # Multiply contracts (10 for SPX simulation via SPY)
    base_contracts: int = 1
    max_contracts: int = 10

    # === Option Settings ===
    option_symbol: str = ""  # Option root symbol (e.g., "SPXW")
    strike_offset: float = 0.5  # Strike offset from ATM
    strike_interval: float = 1.0  # Strike price interval ($1 for SPY, $5 for SPX)
    underlying_multiplier: float = 1.0  # Multiplier for underlying price to strike
    days_to_expiration: int = 0  # 0 = 0DTE, 1 = 1DTE, etc.

    # === Exit Settings ===
    profit_target_pct: Optional[float] = 0.20  # 20% profit target
    profit_target_dollars: float = 0.50  # Fixed dollar profit
    stop_loss_pct: float = 0.25  # 25% stop loss

    # === Time Restrictions ===
    entry_start_time_et: Optional[time] = None  # Earliest entry time (ET)
    entry_end_time_et: Optional[time] = None  # Latest entry time (ET)
    entry_cutoff_hour_utc: int = 19  # Default: 2:00 PM ET = 19:00 UTC

    # === Averaging Down ===
    enable_averaging_down: bool = False
    avg_down_trigger_pct: float = 0.10  # Add contract every -10% from entry
    max_add_ons: int = 3  # Max 3 add-ons (4 contracts total)

    # === Timeframe Settings ===
    primary_timeframe: str = "1Min"  # Primary bar timeframe (1Min for ORB precision)
    candlestick_timeframe: str = "15Min"  # For pattern analysis

    # === RSI Settings (for RSI-based strategies) ===
    rsi_period: int = 14
    rsi_sma_period: int = 10
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    def to_backtest_config(self) -> "BacktestConfig":
        """
        Convert to BacktestConfig for existing infrastructure.

        This bridges the new strategy framework with the existing
        backtest service, allowing reuse of simulation logic.

        Returns:
            BacktestConfig instance with mapped parameters
        """
        from app.services.backtest.config import BacktestConfig

        return BacktestConfig(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            slippage_entry_pct=self.slippage_entry_pct,
            slippage_exit_pct=self.slippage_exit_pct,
            slippage_stop_extra_pct=self.slippage_stop_extra_pct,
            min_slippage_dollars=self.min_slippage_dollars,
            commission_per_contract=self.commission_per_contract,
            contract_multiplier=self.contract_multiplier,
            option_symbol=self.option_symbol,
            strike_offset=self.strike_offset,
            strike_interval=self.strike_interval,
            underlying_multiplier=self.underlying_multiplier,
            profit_target_pct=self.profit_target_pct,
            profit_target_dollars=self.profit_target_dollars,
            stop_loss_pct=self.stop_loss_pct,
            entry_cutoff_hour_utc=self.entry_cutoff_hour_utc,
            avg_down_trigger_pct=self.avg_down_trigger_pct,
            max_add_ons=self.max_add_ons,
            primary_timeframe=self.primary_timeframe,
            candlestick_timeframe=self.candlestick_timeframe,
            rsi_period=self.rsi_period,
            rsi_sma_period=self.rsi_sma_period,
            rsi_oversold=self.rsi_oversold,
            rsi_overbought=self.rsi_overbought,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for JSON storage.

        Handles date/time conversion to ISO format strings.

        Returns:
            Dictionary with all config parameters
        """
        result = asdict(self)

        # Convert date/time objects to strings
        if self.start_date:
            result["start_date"] = self.start_date.isoformat()
        if self.end_date:
            result["end_date"] = self.end_date.isoformat()
        if self.entry_start_time_et:
            result["entry_start_time_et"] = self.entry_start_time_et.isoformat()
        if self.entry_end_time_et:
            result["entry_end_time_et"] = self.entry_end_time_et.isoformat()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        """
        Deserialize from dictionary.

        Handles date/time parsing from ISO format strings.

        Args:
            data: Dictionary with config parameters

        Returns:
            StrategyConfig instance
        """
        from datetime import datetime

        # Convert date strings back to date objects
        if data.get("start_date") and isinstance(data["start_date"], str):
            data["start_date"] = datetime.fromisoformat(data["start_date"]).date()
        if data.get("end_date") and isinstance(data["end_date"], str):
            data["end_date"] = datetime.fromisoformat(data["end_date"]).date()
        if data.get("entry_start_time_et") and isinstance(data["entry_start_time_et"], str):
            data["entry_start_time_et"] = datetime.fromisoformat(data["entry_start_time_et"]).time()
        if data.get("entry_end_time_et") and isinstance(data["entry_end_time_et"], str):
            data["entry_end_time_et"] = datetime.fromisoformat(data["entry_end_time_et"]).time()

        return cls(**data)

    def merge_with(self, overrides: Dict[str, Any]) -> "StrategyConfig":
        """
        Create new config with overrides applied.

        Args:
            overrides: Dictionary of parameters to override

        Returns:
            New StrategyConfig with overrides applied
        """
        current = self.to_dict()
        current.update(overrides)
        return self.from_dict(current)
