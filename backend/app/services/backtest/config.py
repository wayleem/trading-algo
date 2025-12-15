"""Backtest configuration module."""

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class BacktestConfig:
    """
    Backtest configuration.

    Controls all aspects of the backtest including:
    - Symbol and date range
    - RSI parameters and thresholds
    - Position sizing and risk management
    - Timeframe configuration
    """

    symbol: str = "SPY"
    start_date: date = None
    end_date: date = None
    rsi_period: int = 20             # Optimized from 14 (smoother RSI, less noise)
    rsi_sma_period: int = 12         # Optimized from 14 (faster SMA response)

    # RSI thresholds for entry signals (optimized via Bayesian optimization)
    rsi_oversold: float = 44.0       # Entry: RSI < 44 for calls (optimized from 30)
    rsi_overbought: float = 60.0     # Entry: RSI > 60 for puts (optimized from 70)

    # RSI thresholds for strong signals (3 contracts)
    rsi_strong_oversold: float = 20.0    # Strong: RSI < 20 for calls
    rsi_strong_overbought: float = 80.0  # Strong: RSI > 80 for puts

    # RSI convergence exit thresholds
    rsi_convergence_call_exit: float = 40.0  # Exit CALL when RSI >= 40
    rsi_convergence_put_exit: float = 60.0   # Exit PUT when RSI <= 60

    # Entry time restriction
    entry_cutoff_hour_utc: int = 19  # 2:00 PM ET = 19:00 UTC

    # Profit target configuration
    profit_target_dollars: float = 0.50       # Fixed dollar profit per contract
    profit_target_pct: Optional[float] = None  # Percentage-based target (overrides dollars if set)
    rider_profit_target_dollars: float = 1.00  # $1.00 profit for rider contract

    # Stop loss configuration (optimized via Bayesian optimization)
    stop_loss_pct: float = 0.25              # 25% stop loss from avg entry price (optimized from 40%)

    # Averaging down configuration
    avg_down_trigger_pct: float = 0.10       # Add contract every -10% from original entry
    max_add_ons: int = 3                     # Max 3 add-ons (4 contracts total)

    # Position filters
    max_position_move_pct: float = 0.40      # Skip trading if option moved 40%+ from open

    # Capital and costs
    initial_capital: float = 10000.0
    commission_per_contract: float = 0.65

    # Realistic slippage model for 0DTE options
    slippage_entry_pct: float = 0.03      # 3% on entry (paying ask vs mid)
    slippage_exit_pct: float = 0.03       # 3% on exit (receiving bid vs mid)
    slippage_stop_extra_pct: float = 0.02 # Extra 2% slippage on stop-loss orders
    min_slippage_dollars: float = 0.02    # Minimum $0.02 slippage per contract
    max_hold_bars: int = 3  # Maximum bars to hold (3 mins for 1-min bars)

    # Dual-timeframe configuration (parallel mode - 3min and 5min run independently)
    dual_timeframe_enabled: bool = True       # Enable dual-timeframe (3-min + 5-min parallel)
    primary_timeframe: str = "5Min"           # Primary timeframe (used in single timeframe mode)
    confirmation_timeframe: str = "3Min"      # Secondary timeframe (legacy, not used)
    parallel_mode: bool = False               # Run 3-min and 5-min strategies independently (same as dual_timeframe_enabled)

    # SPX / contract scaling
    contract_multiplier: int = 1              # Multiply contracts (10 for SPX simulation via SPY)
    strike_interval: float = 1.0              # Strike price interval ($1 for SPY, $5 for SPX)
    option_symbol: str = ""                   # Option root symbol (e.g., "SPXW" for SPX options)
    underlying_multiplier: float = 1.0        # Multiplier for underlying price to strike (10.0 for SPY->SPXW)
    strike_offset: float = 0.5                # Strike offset from ATM (optimized from 2.0 - slightly OTM)

    # Candlestick pattern settings (for position sizing, NOT signal filtering)
    candlestick_timeframe: str = "15Min"      # Timeframe for candlestick pattern analysis (default: 15-min for less noise)
    pattern_body_threshold: float = 0.1       # Max body/range ratio for doji detection (10%)
    pattern_shadow_ratio: float = 2.0         # Min shadow/body ratio for hammer/shooting star
    use_pattern_for_averaging: bool = False   # Require pattern confirmation for averaging down

    # Support/Resistance settings
    use_dynamic_exits: bool = False           # Adjust PT/SL based on S/R levels
    sr_lookback_bars: int = 100               # Bars to analyze for S/R detection
    sr_tolerance_pct: float = 0.5             # Price tolerance for S/R level clustering (0.5%)

    # Pattern-based position sizing (optimized via Bayesian optimization)
    use_pattern_position_sizing: bool = False  # Add bonus contracts for strong patterns
    pattern_strength_threshold: float = 0.8    # Min strength (0.0-1.0) for bonus contracts
    pattern_bonus_contracts: int = 4           # Bonus contracts for strong patterns (optimized from 2)

    # Partial exit configuration
    enable_partial_exits: bool = False         # Enable 50% exit at base PT, rest at extended
    extended_target_multiplier: float = 2.0    # Extended target = base PT × this multiplier
    partial_exit_pattern_threshold: float = 0.8  # Min pattern strength for extended target

    # Kelly Criterion for position sizing
    enable_kelly_sizing: bool = False          # Use Kelly formula for sizing
    kelly_fraction: float = 0.25               # Use 25% of full Kelly (conservative)
    kelly_lookback_trades: int = 50            # Number of recent trades for Kelly calculation
    kelly_min_trades: int = 20                 # Min trades before applying Kelly
    kelly_max_multiplier: float = 3.0          # Cap Kelly multiplier at 3x

    # === MACD Configuration ===
    macd_fast_period: int = 12                 # Fast EMA period
    macd_slow_period: int = 26                 # Slow EMA period
    macd_signal_period: int = 9                # Signal line EMA period

    # === Bollinger Bands Configuration ===
    bb_period: int = 20                        # SMA period for middle band
    bb_num_std: float = 2.0                    # Number of standard deviations
    bb_width_threshold: float = 2.0            # Min band width % for volatility filter

    # === Signal Integration Mode ===
    # rsi_only: Original RSI crossover logic (backwards compatible)
    # macd_filter: RSI signals filtered by MACD histogram direction
    # independent: RSI, MACD, BB each generate independent signals
    # hybrid: MACD confirms RSI momentum, BB provides exit levels
    signal_mode: str = "rsi_only"

    # MACD filtering options (for macd_filter and hybrid modes)
    macd_filter_calls: bool = True             # Require positive histogram for calls
    macd_filter_puts: bool = True              # Require negative histogram for puts
    macd_signal_threshold: float = 0.0         # Min histogram magnitude for signal

    # === Bollinger Band Strategies ===
    # Entry: none, touch (enter at band touch), squeeze (enter on breakout)
    bb_entry_strategy: str = "none"
    # Exit: none, mean_reversion (exit at middle band), opposite_band (exit at opposite band)
    bb_exit_strategy: str = "none"
    bb_volatility_filter: bool = False         # Only trade when width >= threshold

    # === Dual RSI Confirmation Mode ===
    # none: Current parallel mode (3-min and 5-min run independently)
    # both_agree: Both timeframes must be oversold/overbought AND cross SMA
    # 5min_trigger: 5-min generates signal, 3-min RSI must be in extreme zone
    # either_triggers: Either timeframe signals, other confirms with zone
    rsi_confirmation_mode: str = "none"
    rsi_confirm_buffer: float = 5.0            # RSI must be within buffer of threshold

    # === ADX Filter Configuration ===
    use_adx_filter: bool = False               # Filter trades by ADX (trend strength)
    adx_period: int = 14                       # ADX calculation period
    adx_threshold: float = 25.0                # ADX < threshold = range-bound (favor ORB)
    adx_filter_mode: str = "below"             # "below" = trade when ADX < threshold, "above" = trade when ADX > threshold

    # === Expected Move Filter Configuration ===
    use_em_filter: bool = False                # Filter trades by Expected Move
    em_max_ratio: float = 0.8                  # Max price move vs EM before filtering (0.8 = 80% of EM)

    def get_effective_profit_target(self, entry_price: float) -> float:
        """
        Calculate effective profit target price.

        If profit_target_pct is set, uses percentage-based target.
        Otherwise, uses fixed dollar target.

        Args:
            entry_price: The entry price of the option

        Returns:
            Target exit price for profit
        """
        if self.profit_target_pct is not None:
            return entry_price * (1 + self.profit_target_pct)
        return entry_price + self.profit_target_dollars

    def get_stop_loss_price(self, avg_entry_price: float) -> float:
        """
        Calculate stop loss price.

        Args:
            avg_entry_price: The average entry price (accounts for averaging down)

        Returns:
            Stop loss trigger price
        """
        return avg_entry_price * (1 - self.stop_loss_pct)

    def get_extended_profit_target(self, entry_price: float) -> float:
        """
        Calculate extended profit target price for partial exits.

        Extended target is the base profit target multiplied by extended_target_multiplier.

        Args:
            entry_price: The entry price of the option

        Returns:
            Extended target exit price for profit (for runner contracts)
        """
        base_target = self.get_effective_profit_target(entry_price)
        profit_amount = base_target - entry_price
        return entry_price + (profit_amount * self.extended_target_multiplier)
