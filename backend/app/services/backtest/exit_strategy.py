"""Exit strategy module for backtest simulation.

Provides pluggable exit logic for stop loss and profit targets.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from .config import BacktestConfig
from .models import SimulatedTrade
from .option_pricing import OptionPricingService


class ExitStrategy(ABC):
    """
    Abstract base class for exit strategies.

    Implement this to create custom exit logic (e.g., trailing stops,
    time-based exits, RSI convergence, etc.)
    """

    @abstractmethod
    def check_exit(
        self,
        trade: SimulatedTrade,
        current_price: float,
        config: BacktestConfig,
        current_bar_index: int = 0,
        current_rsi: float = 0.0,
        current_sma: float = 0.0,
    ) -> Tuple[Optional[str], int, float]:
        """
        Check if exit conditions are met.

        Args:
            trade: The current open trade
            current_price: Current option price
            config: Backtest configuration
            current_bar_index: Current bar index (for time-based exits)
            current_rsi: Current RSI value (for RSI-based exits)
            current_sma: Current SMA value

        Returns:
            (exit_reason, contracts_to_close, exit_price)
            - exit_reason: "stop_loss", "profit_target", etc. or None if no exit
            - contracts_to_close: Number of contracts to close
            - exit_price: Price to close at
        """
        pass


class StandardExitStrategy(ExitStrategy):
    """
    Standard exit strategy with stop loss and profit target.

    Supports:
    - Percentage-based stop loss
    - Fixed dollar or percentage profit targets
    - Symbol-aware profit target scaling
    """

    def __init__(self, option_pricing: Optional[OptionPricingService] = None):
        """
        Initialize standard exit strategy.

        Args:
            option_pricing: Optional pricing service for price lookups
        """
        self._option_pricing = option_pricing

    def check_exit(
        self,
        trade: SimulatedTrade,
        current_price: float,
        config: BacktestConfig,
        current_bar_index: int = 0,
        current_rsi: float = 0.0,
        current_sma: float = 0.0,
    ) -> Tuple[Optional[str], int, float]:
        """
        Check standard exit conditions.

        Order of checks:
        1. Stop loss (percentage below avg entry)
        2. Profit target (fixed dollar or percentage)

        Returns:
            (exit_reason, contracts_to_close, current_price)
        """
        # Use avg_entry_price for P&L calc (handles averaging down)
        avg_price = trade.get_effective_entry_price()
        profit_per_contract = current_price - avg_price

        # 1. Stop loss: X% below avg entry price - exits ALL remaining contracts
        stop_loss_price = config.get_stop_loss_price(avg_price)
        if current_price <= stop_loss_price:
            return ("stop_loss", trade.total_contracts, current_price)

        # 2. Profit target: exits ALL contracts
        profit_target_price = config.get_effective_profit_target(avg_price)

        if config.profit_target_pct is not None:
            # Percentage-based target
            if current_price >= profit_target_price:
                return ("profit_target", trade.total_contracts, current_price)
        else:
            # Fixed dollar target
            if profit_per_contract >= config.profit_target_dollars:
                return ("profit_target", trade.total_contracts, current_price)

        return (None, 0, current_price)


class ScaledDollarExitStrategy(ExitStrategy):
    """
    Symbol-aware exit strategy with scaled profit targets.

    Automatically adjusts profit targets based on symbol:
    - SPXW: $0.50 profit target (default)
    - SPY: $0.05 profit target (10x smaller)

    This accounts for the 10x price difference between SPY and SPXW options.
    """

    # Symbol-specific profit targets
    SYMBOL_PROFIT_TARGETS = {
        "SPXW": 0.50,  # $0.50 for SPXW options
        "SPY": 0.05,   # $0.05 for SPY options (10x smaller)
    }

    DEFAULT_PROFIT_TARGET = 0.50

    def __init__(self, option_symbol: str = "SPXW"):
        """
        Initialize scaled exit strategy.

        Args:
            option_symbol: Option root symbol (SPXW, SPY, etc.)
        """
        self._option_symbol = option_symbol
        self._profit_target = self.SYMBOL_PROFIT_TARGETS.get(
            option_symbol.upper(),
            self.DEFAULT_PROFIT_TARGET
        )

    @property
    def profit_target(self) -> float:
        """Get the symbol-specific profit target."""
        return self._profit_target

    def check_exit(
        self,
        trade: SimulatedTrade,
        current_price: float,
        config: BacktestConfig,
        current_bar_index: int = 0,
        current_rsi: float = 0.0,
        current_sma: float = 0.0,
    ) -> Tuple[Optional[str], int, float]:
        """
        Check exit conditions with symbol-scaled profit target.

        Returns:
            (exit_reason, contracts_to_close, current_price)
        """
        avg_price = trade.get_effective_entry_price()
        profit_per_contract = current_price - avg_price

        # 1. Stop loss (same as standard)
        stop_loss_price = config.get_stop_loss_price(avg_price)
        if current_price <= stop_loss_price:
            return ("stop_loss", trade.total_contracts, current_price)

        # 2. Profit target using symbol-specific target
        if profit_per_contract >= self._profit_target:
            return ("profit_target", trade.total_contracts, current_price)

        return (None, 0, current_price)


class PercentageExitStrategy(ExitStrategy):
    """
    Percentage-based profit target exit strategy.

    Uses percentage of entry price for profit target instead of fixed dollars.
    Useful for normalizing returns across different option prices.
    """

    def __init__(self, profit_target_pct: float = 0.15, stop_loss_pct: float = 0.40):
        """
        Initialize percentage exit strategy.

        Args:
            profit_target_pct: Profit target as percentage (0.15 = 15%)
            stop_loss_pct: Stop loss as percentage (0.40 = 40%)
        """
        self._profit_target_pct = profit_target_pct
        self._stop_loss_pct = stop_loss_pct

    def check_exit(
        self,
        trade: SimulatedTrade,
        current_price: float,
        config: BacktestConfig,
        current_bar_index: int = 0,
        current_rsi: float = 0.0,
        current_sma: float = 0.0,
    ) -> Tuple[Optional[str], int, float]:
        """
        Check exit conditions with percentage-based targets.

        Returns:
            (exit_reason, contracts_to_close, current_price)
        """
        avg_price = trade.get_effective_entry_price()

        # 1. Stop loss: percentage below avg entry
        stop_loss_price = avg_price * (1 - self._stop_loss_pct)
        if current_price <= stop_loss_price:
            return ("stop_loss", trade.total_contracts, current_price)

        # 2. Profit target: percentage above avg entry
        profit_target_price = avg_price * (1 + self._profit_target_pct)
        if current_price >= profit_target_price:
            return ("profit_target", trade.total_contracts, current_price)

        return (None, 0, current_price)


class PartialExitStrategy(ExitStrategy):
    """
    Partial exit strategy with tiered profit taking.

    Exit logic:
    1. Stop loss: Exit ALL contracts at stop loss price
    2. Base profit target: Exit 50% of position ("partial_tp")
    3. Extended profit target: Exit remaining 50% if pattern was strong

    For strong patterns (pattern_strength >= threshold):
    - 50% exits at base profit target
    - Remaining 50% runs to extended target (base × multiplier)

    For weak patterns:
    - All contracts exit at base profit target
    """

    def __init__(self, option_pricing: OptionPricingService = None):
        """
        Initialize partial exit strategy.

        Args:
            option_pricing: Optional pricing service for price lookups
        """
        self._option_pricing = option_pricing

    def check_exit(
        self,
        trade: SimulatedTrade,
        current_price: float,
        config: BacktestConfig,
        current_bar_index: int = 0,
        current_rsi: float = 0.0,
        current_sma: float = 0.0,
    ) -> Tuple[Optional[str], int, float]:
        """
        Check partial exit conditions.

        Order of checks:
        1. Stop loss (exits ALL remaining contracts)
        2. Extended target (if partial TP taken and strong pattern)
        3. Base profit target (partial or full)

        Returns:
            (exit_reason, contracts_to_close, current_price)
        """
        avg_price = trade.get_effective_entry_price()

        # 1. Stop loss check - exits ALL remaining contracts
        stop_loss_price = config.get_stop_loss_price(avg_price)
        if current_price <= stop_loss_price:
            return ("stop_loss", trade.remaining_contracts, current_price)

        # Get profit target prices
        base_target = config.get_effective_profit_target(avg_price)
        extended_target = trade.extended_target or config.get_extended_profit_target(avg_price)

        # Check if this is a strong pattern trade that qualifies for extended target
        is_strong_pattern = (
            config.enable_partial_exits and
            trade.pattern_strength >= config.partial_exit_pattern_threshold
        )

        # 2. Check extended target (for remaining contracts after partial TP)
        if trade.partial_tp_taken and is_strong_pattern:
            if current_price >= extended_target:
                return ("extended_tp", trade.remaining_contracts, current_price)
            # Don't exit yet - let it run to extended target
            return (None, 0, current_price)

        # 3. Check base profit target
        if current_price >= base_target:
            if config.enable_partial_exits and not trade.partial_tp_taken:
                # Partial exit: 50% at base target
                contracts_to_exit = trade.remaining_contracts // 2
                if contracts_to_exit > 0:
                    return ("partial_tp", contracts_to_exit, current_price)
                # If only 1 contract, exit all at base target
                return ("profit_target", trade.remaining_contracts, current_price)
            elif trade.partial_tp_taken and not is_strong_pattern:
                # Weak pattern - exit remaining at base target
                return ("profit_target", trade.remaining_contracts, current_price)
            else:
                # No partial exits or already took partial - full exit
                return ("profit_target", trade.remaining_contracts, current_price)

        return (None, 0, current_price)


class BollingerBandExitStrategy(ExitStrategy):
    """
    Bollinger Band-based exit strategy.

    Exit strategies:
    - mean_reversion: Exit when underlying price returns to middle band (SMA)
    - opposite_band: Exit when underlying reaches opposite band

    CALL exits:
    - mean_reversion: Exit when price >= middle band
    - opposite_band: Exit when price >= upper band

    PUT exits:
    - mean_reversion: Exit when price <= middle band
    - opposite_band: Exit when price <= lower band

    Also includes stop loss and profit target protection.
    """

    def __init__(
        self,
        bb_exit_strategy: str = "mean_reversion",
        stop_loss_pct: float = 0.25,
        profit_target_dollars: float = 0.50,
    ):
        """
        Initialize Bollinger Band exit strategy.

        Args:
            bb_exit_strategy: "mean_reversion" or "opposite_band"
            stop_loss_pct: Stop loss percentage
            profit_target_dollars: Fallback profit target if BB exit not triggered
        """
        self._bb_exit_strategy = bb_exit_strategy
        self._stop_loss_pct = stop_loss_pct
        self._profit_target_dollars = profit_target_dollars
        # BB values set per bar
        self._bb_upper: float = 0.0
        self._bb_middle: float = 0.0
        self._bb_lower: float = 0.0
        self._underlying_price: float = 0.0

    def set_bb_values(
        self,
        bb_upper: float,
        bb_middle: float,
        bb_lower: float,
        underlying_price: float,
    ) -> None:
        """
        Set current Bollinger Band values for exit evaluation.

        Must be called before check_exit() for each bar.

        Args:
            bb_upper: Upper Bollinger Band value
            bb_middle: Middle band (SMA) value
            bb_lower: Lower Bollinger Band value
            underlying_price: Current underlying price
        """
        self._bb_upper = bb_upper
        self._bb_middle = bb_middle
        self._bb_lower = bb_lower
        self._underlying_price = underlying_price

    def check_exit(
        self,
        trade: SimulatedTrade,
        current_price: float,
        config: BacktestConfig,
        current_bar_index: int = 0,
        current_rsi: float = 0.0,
        current_sma: float = 0.0,
    ) -> Tuple[Optional[str], int, float]:
        """
        Check Bollinger Band exit conditions.

        Order of checks:
        1. Stop loss (exits ALL contracts)
        2. Profit target (hard limit)
        3. Bollinger Band exit (mean reversion or opposite band)

        Returns:
            (exit_reason, contracts_to_close, current_price)
        """
        avg_price = trade.get_effective_entry_price()

        # 1. Stop loss check first
        stop_loss_price = avg_price * (1 - self._stop_loss_pct)
        if current_price <= stop_loss_price:
            return ("stop_loss", trade.total_contracts, current_price)

        # 2. Profit target check (hard limit)
        if current_price >= avg_price + self._profit_target_dollars:
            return ("profit_target", trade.total_contracts, current_price)

        # 3. Bollinger Band exit
        if self._bb_exit_strategy == "mean_reversion":
            # Exit at middle band
            if trade.is_call and self._underlying_price >= self._bb_middle:
                return ("bb_mean_reversion", trade.total_contracts, current_price)
            if trade.is_put and self._underlying_price <= self._bb_middle:
                return ("bb_mean_reversion", trade.total_contracts, current_price)

        elif self._bb_exit_strategy == "opposite_band":
            # Exit at opposite band
            if trade.is_call and self._underlying_price >= self._bb_upper:
                return ("bb_opposite_band", trade.total_contracts, current_price)
            if trade.is_put and self._underlying_price <= self._bb_lower:
                return ("bb_opposite_band", trade.total_contracts, current_price)

        return (None, 0, current_price)


class RSIConvergenceExitStrategy(ExitStrategy):
    """
    RSI convergence exit strategy.

    Exits when RSI returns to neutral zone:
    - CALL: Exit when RSI >= 40 (from oversold)
    - PUT: Exit when RSI <= 60 (from overbought)

    Also includes stop loss protection.
    """

    def __init__(
        self,
        call_exit_rsi: float = 40.0,
        put_exit_rsi: float = 60.0,
        stop_loss_pct: float = 0.40,
    ):
        """
        Initialize RSI convergence exit strategy.

        Args:
            call_exit_rsi: RSI threshold to exit calls (default 40)
            put_exit_rsi: RSI threshold to exit puts (default 60)
            stop_loss_pct: Stop loss percentage
        """
        self._call_exit_rsi = call_exit_rsi
        self._put_exit_rsi = put_exit_rsi
        self._stop_loss_pct = stop_loss_pct

    def check_exit(
        self,
        trade: SimulatedTrade,
        current_price: float,
        config: BacktestConfig,
        current_bar_index: int = 0,
        current_rsi: float = 0.0,
        current_sma: float = 0.0,
    ) -> Tuple[Optional[str], int, float]:
        """
        Check RSI convergence exit conditions.

        Returns:
            (exit_reason, contracts_to_close, current_price)
        """
        avg_price = trade.get_effective_entry_price()

        # 1. Stop loss check first
        stop_loss_price = avg_price * (1 - self._stop_loss_pct)
        if current_price <= stop_loss_price:
            return ("stop_loss", trade.total_contracts, current_price)

        # 2. RSI convergence exit
        if trade.is_call and current_rsi >= self._call_exit_rsi:
            return ("rsi_convergence", trade.total_contracts, current_price)

        if trade.is_put and current_rsi <= self._put_exit_rsi:
            return ("rsi_convergence", trade.total_contracts, current_price)

        return (None, 0, current_price)


def get_exit_strategy(
    strategy_type: str = "standard",
    option_symbol: str = "SPXW",
    **kwargs
) -> ExitStrategy:
    """
    Factory function to create exit strategies.

    Args:
        strategy_type: Type of strategy ("standard", "scaled", "percentage", "rsi", "partial", "bb")
        option_symbol: Option root symbol (for scaled strategy)
        **kwargs: Additional strategy-specific parameters

    Returns:
        Configured ExitStrategy instance
    """
    strategies = {
        "standard": StandardExitStrategy,
        "scaled": lambda: ScaledDollarExitStrategy(option_symbol),
        "percentage": lambda: PercentageExitStrategy(
            profit_target_pct=kwargs.get("profit_target_pct", 0.15),
            stop_loss_pct=kwargs.get("stop_loss_pct", 0.40),
        ),
        "rsi": lambda: RSIConvergenceExitStrategy(
            call_exit_rsi=kwargs.get("call_exit_rsi", 40.0),
            put_exit_rsi=kwargs.get("put_exit_rsi", 60.0),
            stop_loss_pct=kwargs.get("stop_loss_pct", 0.40),
        ),
        "partial": lambda: PartialExitStrategy(
            option_pricing=kwargs.get("option_pricing"),
        ),
        "bb": lambda: BollingerBandExitStrategy(
            bb_exit_strategy=kwargs.get("bb_exit_strategy", "mean_reversion"),
            stop_loss_pct=kwargs.get("stop_loss_pct", 0.25),
            profit_target_dollars=kwargs.get("profit_target_dollars", 0.50),
        ),
    }

    if strategy_type not in strategies:
        raise ValueError(f"Unknown exit strategy: {strategy_type}")

    factory = strategies[strategy_type]
    return factory() if callable(factory) else factory
