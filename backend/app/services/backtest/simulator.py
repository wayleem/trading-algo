"""Trade simulator module for backtest.

Contains the core simulation loops for executing trades based on signals.
"""

import logging
from datetime import date, datetime
from typing import Callable, Dict, List, Optional

from app.models.schemas import SignalType
from app.services.candlestick_patterns import CandlestickAnalyzer
from app.services.support_resistance import SupportResistanceAnalyzer
from app.services.indicators import calculate_adx_series

from .config import BacktestConfig
from .exit_strategy import ExitStrategy, StandardExitStrategy, BollingerBandExitStrategy
from .models import SimulatedTrade
from .option_pricing import OptionPricingService
from .position_manager import PositionManager

logger = logging.getLogger(__name__)


class TradeSimulator:
    """
    Core trade simulation engine.

    Handles:
    - Single timeframe simulation
    - Dual timeframe confirmation
    - Parallel independent strategies
    - Entry/exit execution
    - Position management (averaging down)
    """

    def __init__(
        self,
        config: BacktestConfig,
        option_pricing: OptionPricingService,
        position_manager: PositionManager,
        exit_strategy: Optional[ExitStrategy] = None,
        option_symbol: str = "SPXW",
    ):
        """
        Initialize trade simulator.

        Args:
            config: Backtest configuration
            option_pricing: Service for option price lookups
            position_manager: Manager for position lifecycle
            exit_strategy: Strategy for determining exits (default: StandardExitStrategy)
            option_symbol: Option root symbol (SPXW, SPY, etc.)
        """
        self._config = config
        self._option_pricing = option_pricing
        self._position_manager = position_manager
        self._exit_strategy = exit_strategy or StandardExitStrategy(option_pricing)
        self._option_symbol = option_symbol

        # Initialize optional analyzers based on config
        self._candlestick_analyzer = None
        if config.use_pattern_for_averaging:
            self._candlestick_analyzer = CandlestickAnalyzer(
                body_threshold=config.pattern_body_threshold,
                shadow_ratio=config.pattern_shadow_ratio,
            )

        self._sr_analyzer = None
        if config.use_dynamic_exits:
            self._sr_analyzer = SupportResistanceAnalyzer(
                lookback_bars=config.sr_lookback_bars,
                tolerance_pct=config.sr_tolerance_pct,
            )

        # Store bars for pattern analysis during averaging down
        self._current_bars: List[dict] = []
        # Store separate pattern bars (e.g., 15-min) if using different timeframe for patterns
        self._pattern_bars: Optional[List[dict]] = None
        self._pattern_interval_minutes: int = self._get_pattern_interval_minutes(config.candlestick_timeframe)

        # Track completed trades for Kelly Criterion calculation
        self._completed_trades: List[SimulatedTrade] = []
        self._current_kelly_multiplier: float = 1.0

    def _get_pattern_interval_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to interval in minutes."""
        timeframe_map = {
            "1Min": 1,
            "3Min": 3,
            "5Min": 5,
            "15Min": 15,
            "30Min": 30,
            "1Hour": 60,
            "1H": 60,
        }
        return timeframe_map.get(timeframe, 15)

    def _get_pattern_bars_for_timestamp(
        self,
        bar_time: datetime,
        count: int = 3,
    ) -> List[dict]:
        """
        Get pattern bars aligned to the given timestamp.

        Args:
            bar_time: The timestamp from the current bar
            count: Number of recent pattern bars to return

        Returns:
            List of recent pattern bars for pattern analysis
        """
        if not self._pattern_bars:
            return []

        # Align to pattern timeframe boundary (floor)
        aligned = bar_time.replace(
            minute=(bar_time.minute // self._pattern_interval_minutes) * self._pattern_interval_minutes,
            second=0,
            microsecond=0
        )

        # Find pattern bars up to and including aligned timestamp
        recent_bars = []
        for bar in self._pattern_bars:
            bar_ts = bar["timestamp"]
            if hasattr(bar_ts, 'replace'):
                bar_ts = bar_ts.replace(second=0, microsecond=0)
            if bar_ts <= aligned:
                recent_bars.append(bar)

        # Return the last `count` bars
        return recent_bars[-count:] if recent_bars else []

    def get_strike_for_trade(
        self,
        underlying_price: float,
        option_type: str,
        strike_offset: float = 2.0,
        strike_interval: float = 1.0,
        underlying_multiplier: float = 1.0,
    ) -> float:
        """
        Calculate strike price for option.

        Args:
            underlying_price: Current underlying price
            option_type: "call" or "put"
            strike_offset: Distance from ATM. Positive = OTM, Negative = ITM
            strike_interval: Strike price interval ($1 for SPY, $5 for SPX)
            underlying_multiplier: Multiplier for underlying (10.0 for SPY->SPXW)

        Returns:
            Strike price rounded to nearest strike interval
        """
        # Scale underlying price for cross-symbol trading (e.g., SPY price -> SPXW strike)
        scaled_price = underlying_price * underlying_multiplier
        # Scale strike offset proportionally (preserves sign for ITM/OTM)
        scaled_offset = strike_offset * underlying_multiplier

        if option_type == "call":
            # Positive offset = OTM (strike above price)
            # Negative offset = ITM (strike below price)
            strike = scaled_price + scaled_offset
        else:
            # Positive offset = OTM (strike below price)
            # Negative offset = ITM (strike above price)
            strike = scaled_price - scaled_offset

        # Round to nearest strike interval
        return round(strike / strike_interval) * strike_interval

    def _calculate_profit_target(self, entry_price: float) -> float:
        """Calculate profit target price based on config."""
        return self._config.get_effective_profit_target(entry_price)

    def _determine_contract_count(
        self,
        signal_type: SignalType,
        current_rsi: float,
        pattern_strength: float = 0.0,
    ) -> int:
        """
        Determine number of contracts based on signal strength, pattern strength, and Kelly.

        Base sizing:
        - Strong RSI signals (RSI < 20 or > 80) get 3 contracts
        - Normal signals get 1 contract

        Pattern bonus (if enabled):
        - Strong patterns (strength >= threshold) add bonus contracts

        Kelly multiplier (if enabled):
        - Scales position based on historical win rate

        Result is multiplied by contract_multiplier.
        """
        config = self._config

        # Base contracts from RSI strength
        if signal_type == SignalType.BUY_CALL:
            is_strong_rsi = current_rsi <= config.rsi_strong_oversold
        else:  # BUY_PUT
            is_strong_rsi = current_rsi >= config.rsi_strong_overbought

        base_contracts = 3 if is_strong_rsi else 1

        # Pattern bonus contracts
        bonus_contracts = 0
        if config.use_pattern_position_sizing and pattern_strength >= config.pattern_strength_threshold:
            bonus_contracts = config.pattern_bonus_contracts

        # Apply contract multiplier
        raw_contracts = (base_contracts + bonus_contracts) * config.contract_multiplier

        # Apply Kelly multiplier if enabled
        if config.enable_kelly_sizing:
            raw_contracts = int(raw_contracts * self._current_kelly_multiplier)

        return max(1, raw_contracts)

    def _calculate_kelly_multiplier(self) -> float:
        """
        Calculate Kelly Criterion position size multiplier.

        Kelly % = W - [(1-W) / R]
        where:
        - W = win rate (probability of winning)
        - R = win/loss ratio (average win / average loss)

        Returns fractional Kelly (kelly_fraction * full Kelly) for conservative sizing.
        """
        config = self._config

        if len(self._completed_trades) < config.kelly_min_trades:
            return 1.0

        # Use recent trades for calculation
        recent_trades = self._completed_trades[-config.kelly_lookback_trades:]

        winners = [t for t in recent_trades if t.pnl is not None and t.pnl > 0]
        losers = [t for t in recent_trades if t.pnl is not None and t.pnl <= 0]

        if not winners or not losers:
            return 1.0

        win_rate = len(winners) / len(recent_trades)
        avg_win = sum(t.pnl for t in winners) / len(winners)
        avg_loss = abs(sum(t.pnl for t in losers) / len(losers))

        if avg_loss == 0:
            return 1.0

        win_loss_ratio = avg_win / avg_loss

        # Full Kelly formula
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Apply fraction for conservative sizing
        fractional_kelly = kelly_pct * config.kelly_fraction

        # Convert to multiplier (1.0 = base, >1 = increase, <1 = decrease)
        # Clamp between 0.5 and kelly_max_multiplier
        multiplier = max(0.5, min(1.0 + fractional_kelly, config.kelly_max_multiplier))

        return multiplier

    async def simulate_trades(
        self,
        bars: List[dict],
        signals: List,
        rsi_values: List[float],
        sma_values: List[float],
        progress_callback: Optional[Callable] = None,
        pattern_bars: Optional[List[dict]] = None,
        pattern_confirmations: Optional[List] = None,
        bb_upper: Optional[List[float]] = None,
        bb_middle: Optional[List[float]] = None,
        bb_lower: Optional[List[float]] = None,
    ) -> List[SimulatedTrade]:
        """
        Simulate trade execution on historical data.

        IMPORTANT: To avoid look-ahead bias, signals detected at bar N are
        executed at bar N+1. This mimics real trading where you see the bar
        close, then enter on the next bar's open.

        Args:
            bars: Price bars with timestamp, open, high, low, close
            signals: Signal objects with signal_type
            rsi_values: RSI values for each bar
            sma_values: SMA values for each bar
            progress_callback: Optional callback for progress updates
            pattern_bars: Optional separate timeframe bars for candlestick pattern analysis
            pattern_confirmations: Optional list of pattern tuples (is_bullish, is_bearish, strength, pattern_name)
            bb_upper: Optional Bollinger Band upper values for exit strategy
            bb_middle: Optional Bollinger Band middle values for exit strategy
            bb_lower: Optional Bollinger Band lower values for exit strategy

        Returns:
            List of completed SimulatedTrade objects
        """
        trades = []
        current_trade: Optional[SimulatedTrade] = None
        current_day: Optional[date] = None
        previous_bar: Optional[dict] = None

        # Pending signal from previous bar (to avoid look-ahead bias)
        pending_signal = None
        pending_signal_rsi: float = 0.0
        pending_signal_bar_index: int = 0
        pending_pattern_info = None

        config = self._config

        # Store bars reference for pattern/S/R analysis
        self._current_bars = bars
        # Store pattern bars if provided (e.g., 15-min bars for pattern analysis)
        self._pattern_bars = pattern_bars
        # Store BB data for exit strategy
        self._bb_upper = bb_upper
        self._bb_middle = bb_middle
        self._bb_lower = bb_lower

        for i, (bar, signal, current_rsi, current_sma) in enumerate(
            zip(bars, signals, rsi_values, sma_values)
        ):
            if progress_callback and i % 1000 == 0:
                progress_callback(i, len(bars))

            bar_time = bar["timestamp"]
            bar_day = bar_time.date() if hasattr(bar_time, "date") else bar_time

            # Reset at start of new day (0DTE - must close at EOD)
            if current_day != bar_day:
                if current_trade and previous_bar is not None:
                    current_trade = await self._close_at_eod(
                        current_trade, previous_bar, trades
                    )
                current_day = bar_day
                # Clear pending signal on new day (can't enter on yesterday's signal)
                pending_signal = None

            # Check exits for current trade
            if current_trade:
                # Set BB values on exit strategy if it's a BB strategy
                if isinstance(self._exit_strategy, BollingerBandExitStrategy):
                    bb_upper_val = self._bb_upper[i] if self._bb_upper and i < len(self._bb_upper) else 0.0
                    bb_middle_val = self._bb_middle[i] if self._bb_middle and i < len(self._bb_middle) else 0.0
                    bb_lower_val = self._bb_lower[i] if self._bb_lower and i < len(self._bb_lower) else 0.0
                    self._exit_strategy.set_bb_values(bb_upper_val, bb_middle_val, bb_lower_val, bar["close"])

                current_trade = await self._process_open_trade(
                    current_trade, bar, i, current_rsi, current_sma, trades
                )

            # Execute pending signal from PREVIOUS bar (avoids look-ahead bias)
            # Entry uses THIS bar's open price, not previous bar's close
            if current_trade is None and pending_signal is not None:
                current_trade = await self._open_new_trade(
                    bar,  # Current bar for entry timing
                    pending_signal,
                    pending_signal_rsi,
                    i,
                    pending_pattern_info,
                    use_open_price=True,  # Use bar's open, not close
                )
                pending_signal = None  # Clear after execution

            # Store NEW signal for execution on NEXT bar (look-ahead prevention)
            if current_trade is None and signal.signal_type != SignalType.NO_SIGNAL:
                pending_signal = signal
                pending_signal_rsi = current_rsi
                pending_signal_bar_index = i
                pending_pattern_info = pattern_confirmations[i] if pattern_confirmations else None

            previous_bar = bar

        # Close any remaining trade at end of data
        if current_trade and bars:
            await self._close_at_eod(current_trade, bars[-1], trades)

        # Track completed trades for Kelly calculation
        self._completed_trades.extend(trades)

        return trades

    async def _close_at_eod(
        self,
        trade: SimulatedTrade,
        last_bar: dict,
        trades: List[SimulatedTrade],
    ) -> None:
        """Close trade at end of day and add to trades list."""
        bar_time = last_bar["timestamp"]
        config = self._config

        # Set settlement underlying price (scaled for SPXW)
        trade.settlement_underlying = last_bar["close"] * config.underlying_multiplier

        # Get exit price
        exit_price = self._option_pricing.get_current_price(
            intraday_data=trade.intraday_prices,
            bar_time=bar_time,
            current_underlying=last_bar["close"],
            entry_underlying=trade.underlying_price,
            entry_option_price=trade.entry_price,
            signal_type=trade.signal_type,
            entry_time=trade.entry_time,
        )

        trade = self._position_manager.close_trade(
            trade=trade,
            exit_time=bar_time,
            exit_price=exit_price,
            exit_reason="end_of_day",
        )
        trades.append(trade)
        return None

    async def _process_open_trade(
        self,
        trade: SimulatedTrade,
        bar: dict,
        bar_index: int,
        current_rsi: float,
        current_sma: float,
        trades: List[SimulatedTrade],
    ) -> Optional[SimulatedTrade]:
        """
        Process an open trade - check exits and averaging down.

        Returns:
            Updated trade if still open, None if closed
        """
        bar_time = bar["timestamp"]
        config = self._config

        # Get current option price
        current_price = self._option_pricing.get_current_price(
            intraday_data=trade.intraday_prices,
            bar_time=bar_time,
            current_underlying=bar["close"],
            entry_underlying=trade.underlying_price,
            entry_option_price=trade.entry_price,
            signal_type=trade.signal_type,
            entry_time=trade.entry_time,
        )

        # Track RSI history during trade
        trade.rsi_history.append((bar_time, current_rsi, current_price, bar["close"]))

        # Check exit conditions
        exit_reason, contracts_to_close, _ = self._exit_strategy.check_exit(
            trade=trade,
            current_price=current_price,
            config=config,
            current_bar_index=bar_index,
            current_rsi=current_rsi,
            current_sma=current_sma,
        )

        # Averaging down check
        if exit_reason is None and current_price is not None:
            trade = self._check_averaging_down(trade, current_price, bar_time, bar_index)

        # Handle partial take profit (if strategy returns it)
        if exit_reason == "partial_tp":
            # Calculate P&L for partial exit
            contracts_exited = contracts_to_close
            partial_pnl = (current_price - trade.get_effective_entry_price()) * contracts_exited * 100
            partial_pnl -= config.commission_per_contract * contracts_exited
            trade.partial_tp_pnl += partial_pnl  # Accumulate partial P&L
            trade.partial_tp_taken = True
            trade.remaining_contracts -= contracts_exited
            return trade

        # Handle full exits
        if exit_reason in ("stop_loss", "rider_tp", "profit_target", "rsi_convergence", "extended_tp", "bb_mean_reversion", "bb_opposite_band"):
            trade = self._position_manager.close_trade(
                trade=trade,
                exit_time=bar_time,
                exit_price=current_price,
                exit_reason=exit_reason,
                contracts_to_close=contracts_to_close,
            )
            trades.append(trade)
            return None

        return trade

    def _check_averaging_down(
        self,
        trade: SimulatedTrade,
        current_price: float,
        bar_time: datetime,
        bar_index: int,
    ) -> SimulatedTrade:
        """Check and execute averaging down if conditions met."""
        config = self._config
        num_addons = len(trade.add_on_entries)

        if num_addons >= config.max_add_ons:
            return trade

        # Trigger at -10%, -20%, -30% from ORIGINAL entry
        threshold_pct = (num_addons + 1) * config.avg_down_trigger_pct
        trigger_price = trade.entry_price * (1 - threshold_pct)

        if current_price <= trigger_price:
            # Check candlestick pattern confirmation if enabled
            if self._candlestick_analyzer:
                position_type = "call" if trade.signal_type == SignalType.BUY_CALL else "put"

                # Use pattern bars (e.g., 15-min) if available, otherwise fall back to current bars
                if self._pattern_bars:
                    recent_bars = self._get_pattern_bars_for_timestamp(bar_time, count=3)
                elif self._current_bars:
                    recent_bars = self._current_bars[max(0, bar_index - 2) : bar_index + 1]
                else:
                    recent_bars = []

                # Require pattern confirmation to average down
                if recent_bars and not self._candlestick_analyzer.should_add_contracts(recent_bars, position_type):
                    return trade  # Skip averaging down without pattern confirmation

            add_price = current_price * (1 + config.slippage_entry_pct)
            add_contracts = config.contract_multiplier

            # Update position
            total_cost = (trade.avg_entry_price * trade.total_contracts) + (
                add_price * add_contracts
            )
            trade.total_contracts += add_contracts
            trade.remaining_contracts += add_contracts
            trade.avg_entry_price = total_cost / trade.total_contracts
            trade.add_on_entries.append((add_price, add_contracts, bar_time))

            # Recalculate stop loss from NEW avg entry
            trade.stop_loss = trade.avg_entry_price * (1 - config.stop_loss_pct)

        return trade

    async def _open_new_trade(
        self,
        bar: dict,
        signal,
        current_rsi: float,
        bar_index: int,
        pattern_info: Optional[tuple] = None,
        use_open_price: bool = False,
    ) -> Optional[SimulatedTrade]:
        """
        Open a new trade based on signal.

        Args:
            bar: Current price bar
            signal: Signal object with signal_type
            current_rsi: Current RSI value
            bar_index: Index of current bar
            pattern_info: Optional tuple of (is_bullish, is_bearish, strength, pattern_name)
            use_open_price: If True, use bar's open price for entry (for next-bar execution)

        Returns:
            SimulatedTrade if opened, None if entry rejected (e.g., past cutoff time)
        """
        config = self._config
        bar_time = bar["timestamp"]
        bar_day = bar_time.date() if hasattr(bar_time, "date") else bar_time

        # Check entry cutoff time (prevent late-day entries when options are worthless)
        if hasattr(bar_time, "hour"):
            bar_hour_utc = bar_time.hour
            # If timezone-aware, convert to UTC first
            if bar_time.tzinfo is not None:
                from datetime import timezone
                bar_hour_utc = bar_time.astimezone(timezone.utc).hour

            if bar_hour_utc >= config.entry_cutoff_hour_utc:
                return None  # Skip entry - too late in the day

        # Use open price for next-bar execution (avoids look-ahead bias)
        # Use close price only for same-bar execution (legacy behavior)
        underlying_price = bar["open"] if use_open_price else bar["close"]

        # Determine option type and strike
        option_type = "call" if signal.signal_type == SignalType.BUY_CALL else "put"
        strike = self.get_strike_for_trade(
            underlying_price,
            option_type,
            strike_offset=config.strike_offset,
            strike_interval=config.strike_interval,
            underlying_multiplier=config.underlying_multiplier,
        )
        expiration = bar_day  # 0DTE
        right = "C" if option_type == "call" else "P"

        # Extract pattern strength and name from pattern_info
        pattern_strength = 0.0
        pattern_name = None
        if pattern_info is not None and len(pattern_info) >= 4:
            is_bullish, is_bearish, strength, name = pattern_info
            # Use strength if pattern confirms the signal direction
            if (option_type == "call" and is_bullish) or (option_type == "put" and is_bearish):
                pattern_strength = strength
                pattern_name = name

        # Pre-fetch ALL intraday data for this option (REQUIRED - no fallback)
        intraday_data = []
        if self._option_pricing._theta_available:
            intraday_data = await self._option_pricing.fetch_intraday_option_data(
                symbol=self._option_symbol,
                expiration=expiration,
                strike=strike,
                right=right,
                target_date=bar_day,
            )

        # REQUIRE real ThetaData - skip trade if unavailable
        if not intraday_data:
            logger.debug(
                f"Skipping trade: No ThetaData for {self._option_symbol} "
                f"{strike}{right} exp={expiration} on {bar_day}"
            )
            return None

        # Get entry price from real data only
        entry_price = self._option_pricing.interpolate_option_price(
            intraday_data, bar_time
        )

        # Skip if interpolation failed (shouldn't happen with data, but safety check)
        if entry_price is None:
            logger.debug(f"Skipping trade: Could not interpolate price at {bar_time}")
            return None

        # Apply slippage
        entry_price *= 1 + config.slippage_entry_pct

        # Update Kelly multiplier before determining contract count
        if config.enable_kelly_sizing:
            self._current_kelly_multiplier = self._calculate_kelly_multiplier()

        # Determine contract count (now includes pattern strength and Kelly)
        contracts = self._determine_contract_count(
            signal.signal_type, current_rsi, pattern_strength
        )

        # Calculate profit target and stop loss (potentially using dynamic S/R levels)
        profit_target_pct = config.profit_target_pct if config.profit_target_pct else 0.10
        stop_loss_pct = config.stop_loss_pct

        if self._sr_analyzer and self._current_bars:
            # Get bars up to current bar for S/R analysis
            historical_bars = self._current_bars[:bar_index + 1]
            # Use the already-computed underlying_price (open or close based on use_open_price)

            # Calculate dynamic profit target based on S/R levels
            profit_target_pct = self._sr_analyzer.calculate_dynamic_profit_target(
                entry_price=entry_price,
                underlying_price=underlying_price,
                position_type=option_type,
                bars=historical_bars,
                base_target_pct=profit_target_pct,
            )

            # Calculate dynamic stop loss based on S/R levels
            stop_loss_pct = self._sr_analyzer.calculate_dynamic_stop_loss(
                entry_price=entry_price,
                underlying_price=underlying_price,
                position_type=option_type,
                bars=historical_bars,
                base_stop_pct=stop_loss_pct,
            )

        # Calculate actual prices from percentages
        if config.profit_target_pct is not None or self._sr_analyzer:
            profit_target = entry_price * (1 + profit_target_pct)
        else:
            profit_target = self._calculate_profit_target(entry_price)

        stop_loss = entry_price * (1 - stop_loss_pct)

        # Calculate extended target for partial exits if enabled
        extended_target = None
        if config.enable_partial_exits:
            extended_target = config.get_extended_profit_target(entry_price)

        return SimulatedTrade(
            entry_time=bar_time,
            entry_price=entry_price,
            signal_type=signal.signal_type,
            underlying_price=underlying_price,
            profit_target=profit_target,
            stop_loss=stop_loss,
            entry_bar_index=bar_index,
            entry_rsi=current_rsi,
            strike=strike,
            expiration=expiration,
            intraday_prices=intraday_data,
            initial_contracts=contracts,
            remaining_contracts=contracts,
            total_contracts=contracts,
            avg_entry_price=entry_price,
            pattern_strength=pattern_strength,
            pattern_name=pattern_name,
            kelly_multiplier=self._current_kelly_multiplier,
            extended_target=extended_target,
        )

    async def simulate_trades_parallel(
        self,
        bars_3min: List[dict],
        signals_3min: List,
        rsi_values_3min: List[float],
        sma_values_3min: List[float],
        bars_5min: List[dict],
        signals_5min: List,
        rsi_values_5min: List[float],
        sma_values_5min: List[float],
        progress_callback: Optional[Callable] = None,
        pattern_bars: Optional[List[dict]] = None,
        pattern_confirmations_3min: Optional[List] = None,
        pattern_confirmations_5min: Optional[List] = None,
        bb_upper_3min: Optional[List[float]] = None,
        bb_middle_3min: Optional[List[float]] = None,
        bb_lower_3min: Optional[List[float]] = None,
        bb_upper_5min: Optional[List[float]] = None,
        bb_middle_5min: Optional[List[float]] = None,
        bb_lower_5min: Optional[List[float]] = None,
    ) -> List[SimulatedTrade]:
        """
        Simulate trades for 3-min and 5-min strategies running INDEPENDENTLY in parallel.
        Each strategy can have its own open position at the same time.

        Pattern confirmations are used for POSITION SIZING only (not signal filtering).
        BB data is passed through for exit strategy use.
        """
        all_trades = []

        # Run 3-min strategy
        logger.info("Running 3-min strategy...")
        trades_3min = await self.simulate_trades(
            bars=bars_3min,
            signals=signals_3min,
            rsi_values=rsi_values_3min,
            sma_values=sma_values_3min,
            progress_callback=None,
            pattern_bars=pattern_bars,
            pattern_confirmations=pattern_confirmations_3min,
            bb_upper=bb_upper_3min,
            bb_middle=bb_middle_3min,
            bb_lower=bb_lower_3min,
        )
        for t in trades_3min:
            t.timeframe = "3Min"
        all_trades.extend(trades_3min)

        # Run 5-min strategy
        logger.info("Running 5-min strategy...")
        trades_5min = await self.simulate_trades(
            bars=bars_5min,
            signals=signals_5min,
            rsi_values=rsi_values_5min,
            sma_values=sma_values_5min,
            progress_callback=progress_callback,
            pattern_bars=pattern_bars,
            pattern_confirmations=pattern_confirmations_5min,
            bb_upper=bb_upper_5min,
            bb_middle=bb_middle_5min,
            bb_lower=bb_lower_5min,
        )
        for t in trades_5min:
            t.timeframe = "5Min"
        all_trades.extend(trades_5min)

        # Sort all trades by entry time
        all_trades.sort(key=lambda t: t.entry_time)

        logger.info(
            f"Parallel mode: {len(trades_3min)} trades from 3-min, "
            f"{len(trades_5min)} trades from 5-min"
        )
        return all_trades

    async def simulate_trades_dual_timeframe(
        self,
        bars_3min: List[dict],
        rsi_values_3min: List[float],
        sma_values_3min: List[float],
        lookup_5min: Dict[datetime, dict],
        signal_lookup_5min: Dict[datetime, SignalType],
        progress_callback: Optional[Callable] = None,
        pattern_bars: Optional[List[dict]] = None,
    ) -> List[SimulatedTrade]:
        """
        Simulate trade execution with dual-timeframe confirmation.

        Entry Logic:
        1. Check if 5-min timeframe has a signal (RSI crosses SMA in extreme zone)
        2. Confirm 3-min RSI is also in extreme zone
        3. Execute on 3-min bar (more granular timing)

        Exit Logic: Same as single-timeframe (profit target, stop loss, EOD)
        """
        trades = []
        current_trade: Optional[SimulatedTrade] = None
        current_day: Optional[date] = None
        previous_bar: Optional[dict] = None

        config = self._config

        # Store bars reference for pattern analysis
        self._current_bars = bars_3min
        self._pattern_bars = pattern_bars

        for i, bar in enumerate(bars_3min):
            if progress_callback and i % 1000 == 0:
                progress_callback(i, len(bars_3min))

            bar_time = bar["timestamp"]
            bar_day = bar_time.date() if hasattr(bar_time, "date") else bar_time
            current_rsi_3min = rsi_values_3min[i]
            current_sma_3min = sma_values_3min[i]

            # Day boundary handling - force close at EOD
            if current_day != bar_day:
                if current_trade and previous_bar is not None:
                    current_trade = await self._close_at_eod(
                        current_trade, previous_bar, trades
                    )
                current_day = bar_day

            # Check exits for current trade
            if current_trade:
                current_trade = await self._process_open_trade(
                    current_trade, bar, i, current_rsi_3min, current_sma_3min, trades
                )

            # Check for new entry with dual-timeframe confirmation
            if current_trade is None:
                signal_5min = self._get_5min_signal_at_time(
                    bar_time, lookup_5min, signal_lookup_5min
                )

                if signal_5min and signal_5min != SignalType.NO_SIGNAL:
                    # Confirm 3-min RSI is in extreme zone
                    confirmed = False
                    if signal_5min == SignalType.BUY_CALL:
                        confirmed = current_rsi_3min <= config.rsi_oversold
                    elif signal_5min == SignalType.BUY_PUT:
                        confirmed = current_rsi_3min >= config.rsi_overbought

                    if confirmed:
                        # Create signal-like object for _open_new_trade
                        class ConfirmedSignal:
                            def __init__(self, sig_type):
                                self.signal_type = sig_type

                        current_trade = await self._open_new_trade(
                            bar, ConfirmedSignal(signal_5min), current_rsi_3min, i
                        )

            previous_bar = bar

        # Close any remaining trade at end of data
        if current_trade and bars_3min:
            await self._close_at_eod(current_trade, bars_3min[-1], trades)

        return trades

    def _get_5min_signal_at_time(
        self,
        bar_time: datetime,
        lookup_5min: Dict[datetime, dict],
        signal_lookup_5min: Dict[datetime, SignalType],
    ) -> Optional[SignalType]:
        """Get 5-min signal aligned to current 3-min bar time."""
        from datetime import timedelta

        # Align to 5-min boundary
        aligned_ts = bar_time.replace(
            minute=(bar_time.minute // 5) * 5, second=0, microsecond=0
        )

        if aligned_ts in signal_lookup_5min:
            return signal_lookup_5min[aligned_ts]

        # Fallback: search backwards for closest 5-min bar
        for offset in range(5, 60, 5):
            check_ts = aligned_ts - timedelta(minutes=offset)
            if check_ts in signal_lookup_5min:
                return signal_lookup_5min[check_ts]

        return None
