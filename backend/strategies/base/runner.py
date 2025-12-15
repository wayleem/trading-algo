"""
Strategy runner - runs strategies with signal generation and trade simulation.

StrategyRunner provides a backtest loop for modular strategies using real
option pricing data from ThetaData and stock data from Alpaca.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional, Callable, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from strategies.base.strategy import BaseStrategy, TradingSignal
    from strategies.base.config import StrategyConfig

logger = logging.getLogger(__name__)


class StrategyRunner:
    """
    Runs strategies with real option pricing from ThetaData.

    Provides:
    - Data fetching via AlpacaClient (stock bars)
    - Real option pricing via ThetaData OptionPricingService
    - Custom signal generation from strategy
    - Entry filtering
    - Trade simulation with accurate P&L calculation

    Usage:
        strategy = MorningFadeStrategy()
        runner = StrategyRunner(strategy)
        result = await runner.run(start_date, end_date)
    """

    def __init__(
        self,
        strategy: "BaseStrategy",
        alpaca_client=None,
    ):
        """
        Initialize strategy runner.

        Args:
            strategy: Strategy to run
            alpaca_client: Optional AlpacaClient for data (creates new if None)
        """
        self._strategy = strategy
        self._alpaca_client = alpaca_client
        self._option_pricing = None
        self._intraday_cache: Dict[tuple, list] = {}  # Cache for intraday option data

    def _get_expiration_date(self, trade_date: date, days_to_expiration: int) -> date:
        """
        Calculate option expiration date from trade date.

        For 0DTE: expiration = trade_date
        For 1DTE: expiration = next trading day
        For N-DTE: expiration = N trading days from trade_date

        Note: This is a simplified version that adds calendar days.
        A production version should use a trading calendar to skip weekends/holidays.
        """
        from datetime import timedelta

        if days_to_expiration == 0:
            return trade_date

        # Simple version: add calendar days and skip weekends
        expiration = trade_date
        days_added = 0

        while days_added < days_to_expiration:
            expiration = expiration + timedelta(days=1)
            # Skip weekends (5=Saturday, 6=Sunday)
            if expiration.weekday() < 5:
                days_added += 1

        return expiration

    async def run(
        self,
        start_date: date,
        end_date: date,
        config: Optional["StrategyConfig"] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Run strategy backtest with real option pricing.

        Steps:
        1. Get strategy config (use provided or default)
        2. Verify ThetaData is available (REQUIRED)
        3. Fetch historical stock data from Alpaca
        4. Generate signals using strategy's signal generator
        5. Apply entry filters
        6. Run trade simulation with real ThetaData option prices
        7. Calculate metrics

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            config: Strategy configuration (uses default if None)
            progress_callback: Optional callback for progress updates

        Returns:
            BacktestResult with trades, metrics, and equity curve
        """
        from app.services.alpaca_client import AlpacaClient
        from app.services.backtest.option_pricing import OptionPricingService
        from app.models.schemas import BacktestResult, BacktestMetrics

        # Get config
        cfg = config or self._strategy.get_default_config()
        cfg.start_date = start_date
        cfg.end_date = end_date

        # Validate config
        errors = self._strategy.validate_config(cfg)
        if errors:
            raise ValueError(f"Invalid config: {'; '.join(errors)}")

        logger.info(f"Running {self._strategy.name}: {start_date} to {end_date}")

        # Initialize clients
        if self._alpaca_client is None:
            self._alpaca_client = AlpacaClient()

        # Initialize option pricing service (uses ThetaData)
        self._option_pricing = OptionPricingService(theta_available=True)
        self._intraday_cache.clear()

        # Check ThetaData availability (REQUIRED - no fallback)
        theta_available = await self._check_theta_availability()
        if not theta_available:
            raise RuntimeError(
                "ThetaData is REQUIRED for accurate backtesting. "
                "Please ensure ThetaTerminal is running on port 25503. "
                "Start with: java -jar ThetaTerminalv3.jar"
            )

        # Fetch historical bars from Alpaca
        bars = await self._fetch_bars(cfg)

        if not bars:
            logger.warning("No historical data available")
            return self._empty_result(cfg)

        logger.info(f"Fetched {len(bars)} bars from Alpaca")

        # Generate signals using strategy's signal generator
        signals = await self._strategy.generate_signals(bars, cfg)
        logger.info(f"Generated {len(signals)} signals")

        # Apply entry filters
        filters = self._strategy.get_entry_filters()
        if filters:
            signals = self._apply_filters(signals, bars, filters, cfg)
            active_signals = sum(1 for s in signals if s.signal_type != "NO_SIGNAL")
            logger.info(f"After filtering: {active_signals} active signals")

        # Run trade simulation with real option pricing
        simulated_trades = await self._simulate_trades(signals, bars, cfg, progress_callback)

        # Convert SimulatedTrade to BacktestTrade format
        trades = self._convert_trades(simulated_trades)

        # Calculate metrics
        metrics = self._calculate_metrics(simulated_trades, cfg)

        # Build result
        return BacktestResult(
            trades=trades,
            metrics=metrics,
            equity_curve=self._build_equity_curve(simulated_trades, cfg.initial_capital),
        )

    async def _simulate_trades(
        self,
        signals: List["TradingSignal"],
        bars: List[dict],
        config: "StrategyConfig",
        progress_callback: Optional[Callable] = None,
    ) -> List:
        """
        Trade simulation with real ThetaData option pricing.

        IMPORTANT: Implements next-bar entry to avoid look-ahead bias.
        Signal at bar N -> Entry at bar N+1 using OPEN price.

        For each signal:
        - Store signal as "pending" (to execute on next bar)
        - Fetch real option prices from ThetaData
        - Enter at real option price + slippage using NEXT bar's open
        - Exit at profit target, stop loss, or end of day using real prices
        - Track P&L
        """
        from app.services.backtest.models import SimulatedTrade
        from app.models.schemas import SignalType

        trades = []
        current_trade = None
        current_day = None
        current_intraday_data = []

        # Pending signal from previous bar (to avoid look-ahead bias)
        pending_signal = None
        pending_signal_bar = None

        total_bars = len(bars)

        for i, (signal, bar) in enumerate(zip(signals, bars)):
            if progress_callback and i % 100 == 0:
                progress_callback(i, total_bars)

            bar_time = bar["timestamp"]
            bar_date = bar_time.date() if hasattr(bar_time, "date") else bar_time

            # Check for new day - close any open position at EOD with intrinsic settlement
            if current_day is not None and bar_date != current_day:
                if current_trade is not None:
                    # Get exit price - use intrinsic value for 0DTE settlement
                    prev_bar = bars[i - 1] if i > 0 else bar
                    days_to_exp = getattr(config, 'days_to_expiration', 0)

                    if days_to_exp == 0:
                        # 0DTE: Settle at intrinsic value (can be $0 for OTM)
                        exit_price = self._calculate_intrinsic_value(
                            current_trade, prev_bar["close"]
                        )
                    else:
                        # Non-0DTE: Use market price
                        exit_price = await self._get_option_price(
                            config=config,
                            trade=current_trade,
                            bar=prev_bar,
                            intraday_data=current_intraday_data,
                            is_entry=False,
                        )

                    # Apply exit slippage
                    exit_slippage = max(
                        exit_price * config.slippage_exit_pct,
                        config.min_slippage_dollars
                    ) if exit_price > 0 else 0
                    exit_price = max(0, exit_price - exit_slippage)

                    current_trade.exit_time = prev_bar["timestamp"]
                    current_trade.exit_price = exit_price
                    current_trade.exit_reason = "end_of_day"
                    current_trade.pnl = self._calculate_trade_pnl(current_trade, config)
                    trades.append(current_trade)
                    current_trade = None
                    current_intraday_data = []

                # Clear pending signal on new day (can't carry overnight)
                pending_signal = None
                pending_signal_bar = None

            current_day = bar_date

            # Execute pending signal from PREVIOUS bar (avoids look-ahead bias)
            # Entry uses THIS bar's open price, not previous bar's close
            if current_trade is None and pending_signal is not None:
                entry_result = await self._execute_pending_entry(
                    pending_signal=pending_signal,
                    pending_bar=pending_signal_bar,
                    current_bar=bar,
                    config=config,
                )

                if entry_result:
                    current_trade = entry_result["trade"]
                    current_intraday_data = entry_result["intraday_data"]
                    logger.debug(
                        f"Entered {current_trade.signal_type.name} @ ${current_trade.entry_price:.2f} "
                        f"(strike={current_trade.strike}, underlying={bar['open']:.2f})"
                    )

                # Clear pending signal after execution attempt
                pending_signal = None
                pending_signal_bar = None

            # Check exit conditions for open position (on every bar)
            if current_trade is not None:
                exit_result = await self._check_exit(
                    current_trade, bar, config, current_intraday_data
                )
                if exit_result:
                    current_trade.exit_time = bar["timestamp"]
                    current_trade.exit_price = exit_result["price"]
                    current_trade.exit_reason = exit_result["reason"]
                    current_trade.pnl = self._calculate_trade_pnl(current_trade, config)
                    trades.append(current_trade)
                    current_trade = None
                    current_intraday_data = []

            # Store signal as pending (execute on NEXT bar to avoid look-ahead bias)
            if signal.signal_type != "NO_SIGNAL" and current_trade is None:
                pending_signal = signal
                pending_signal_bar = bar

        # Close any remaining open trade at last bar with intrinsic settlement
        if current_trade is not None and bars:
            last_bar = bars[-1]
            days_to_exp = getattr(config, 'days_to_expiration', 0)

            if days_to_exp == 0:
                # 0DTE: Settle at intrinsic value
                exit_price = self._calculate_intrinsic_value(
                    current_trade, last_bar["close"]
                )
            else:
                exit_price = await self._get_option_price(
                    config=config,
                    trade=current_trade,
                    bar=last_bar,
                    intraday_data=current_intraday_data,
                    is_entry=False,
                )

            # Apply exit slippage
            exit_slippage = max(
                exit_price * config.slippage_exit_pct,
                config.min_slippage_dollars
            ) if exit_price > 0 else 0
            exit_price = max(0, exit_price - exit_slippage)

            current_trade.exit_time = last_bar["timestamp"]
            current_trade.exit_price = exit_price
            current_trade.exit_reason = "end_of_day"
            current_trade.pnl = self._calculate_trade_pnl(current_trade, config)
            trades.append(current_trade)

        if progress_callback:
            progress_callback(total_bars, total_bars)

        return trades

    async def _execute_pending_entry(
        self,
        pending_signal: "TradingSignal",
        pending_bar: dict,
        current_bar: dict,
        config: "StrategyConfig",
    ) -> Optional[dict]:
        """
        Execute a pending entry signal on the current bar.

        Uses current bar's OPEN price for underlying (not previous bar's close)
        to avoid look-ahead bias.

        Args:
            pending_signal: Signal from previous bar
            pending_bar: Bar where signal was generated
            current_bar: Current bar for execution
            config: Strategy config

        Returns:
            Dict with 'trade' and 'intraday_data', or None if entry failed
        """
        from app.services.backtest.models import SimulatedTrade
        from app.models.schemas import SignalType

        bar_time = current_bar["timestamp"]
        bar_date = bar_time.date() if hasattr(bar_time, "date") else bar_time

        # Use current bar's OPEN price for strike calculation (not signal price)
        underlying_price = current_bar["open"]

        signal_type = (
            SignalType.BUY_CALL
            if pending_signal.signal_type == "BUY_CALL"
            else SignalType.BUY_PUT
        )

        # Calculate strike based on OPEN price
        strike_interval = config.strike_interval
        strike = round(underlying_price / strike_interval) * strike_interval

        # Apply strike offset as number of strike intervals OTM
        # strike_offset=1 means 1 strike OTM, strike_offset=0.5 means nearest OTM strike
        strike_offset = getattr(config, 'strike_offset', 0)
        if strike_offset > 0:
            # Round offset to nearest whole number of intervals
            offset_intervals = round(strike_offset)
            if signal_type == SignalType.BUY_CALL:
                strike += offset_intervals * strike_interval
            else:
                strike -= offset_intervals * strike_interval

        # Ensure strike is on valid interval (no fractional strikes)
        strike = round(strike / strike_interval) * strike_interval

        # Calculate expiration
        days_to_exp = getattr(config, 'days_to_expiration', 0)
        expiration_date = self._get_expiration_date(bar_date, days_to_exp)

        # Fetch intraday option data from ThetaData
        intraday_data = await self._fetch_intraday_option_data(
            config=config,
            expiration=expiration_date,
            strike=strike,
            option_type="call" if signal_type == SignalType.BUY_CALL else "put",
            target_date=bar_date,
        )

        # Get real entry price from ThetaData (REQUIRED - no fallback)
        entry_price = await self._get_option_price_from_data(
            intraday_data=intraday_data,
            bar_time=bar_time,
            is_entry=True,
        )

        if entry_price is None or entry_price <= 0:
            logger.debug(f"No option price available for {bar_date} {strike} - skipping trade")
            return None

        # Apply entry slippage (paying ask vs mid price)
        entry_slippage = max(
            entry_price * config.slippage_entry_pct,
            config.min_slippage_dollars
        )
        entry_price = entry_price + entry_slippage

        # Calculate profit target and stop loss prices
        profit_target_pct = config.profit_target_pct or 0.20
        profit_target_price = entry_price * (1 + profit_target_pct)
        stop_loss_price = entry_price * (1 - config.stop_loss_pct)

        trade = SimulatedTrade(
            entry_time=bar_time,  # Entry at current bar, not signal bar
            signal_type=signal_type,
            underlying_price=underlying_price,  # Use open price
            strike=strike,
            entry_price=entry_price,
            profit_target=profit_target_price,
            stop_loss=stop_loss_price,
            total_contracts=config.base_contracts,
            expiration=expiration_date,
        )

        # Store VWAP info for VWAP-based exit strategies
        if pending_signal.metadata.get("vwap"):
            trade.entry_vwap = pending_signal.metadata["vwap"]

        return {"trade": trade, "intraday_data": intraday_data}

    def _calculate_intrinsic_value(self, trade, settlement_price: float) -> float:
        """
        Calculate intrinsic value for option at settlement.

        For 0DTE options expiring at EOD:
        - CALL: max(0, settlement_price - strike)
        - PUT: max(0, strike - settlement_price)

        OTM options settle at $0 = 100% loss.

        Args:
            trade: SimulatedTrade with strike and signal_type
            settlement_price: Underlying price at settlement

        Returns:
            Intrinsic value (can be $0 for OTM)
        """
        from app.models.schemas import SignalType

        strike = trade.strike

        if trade.signal_type == SignalType.BUY_CALL:
            intrinsic = max(0, settlement_price - strike)
        else:  # PUT
            intrinsic = max(0, strike - settlement_price)

        return intrinsic

    async def _check_theta_availability(self) -> bool:
        """
        Check if ThetaData terminal is available.

        Makes a test request to verify ThetaTerminal is running.
        Any HTTP response (including 404) indicates the server is up.

        Returns:
            True if ThetaData is available, False otherwise
        """
        import httpx
        from app.core.config import settings

        try:
            # Just check if server responds - any response means it's running
            # ThetaTerminal doesn't have a /status endpoint
            url = f"{settings.theta_data_base_url}/"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                # Any HTTP response (including 404) means server is up
                logger.info(f"ThetaData check: server responded with {response.status_code}")
                return True
        except httpx.ConnectError:
            logger.warning("ThetaData terminal not running - connection refused")
            return False
        except Exception as e:
            logger.warning(f"ThetaData availability check failed: {e}")
            return False

    async def _fetch_intraday_option_data(
        self,
        config: "StrategyConfig",
        expiration: date,
        strike: float,
        option_type: str,
        target_date: date,
    ) -> list:
        """
        Fetch intraday option data from ThetaData.

        Uses caching to avoid repeated API calls for the same contract.
        """
        cache_key = (config.symbol, expiration, strike, option_type, target_date)

        if cache_key in self._intraday_cache:
            return self._intraday_cache[cache_key]

        try:
            right = "C" if option_type == "call" else "P"
            data = await self._option_pricing.fetch_intraday_option_data(
                symbol=config.symbol,
                expiration=expiration,
                strike=strike,
                right=right,
                target_date=target_date,
            )
            self._intraday_cache[cache_key] = data
            logger.debug(f"Fetched {len(data)} intraday points for {config.symbol} {strike} {option_type}")
            return data
        except Exception as e:
            logger.warning(f"Failed to fetch option data: {e}")
            return []

    async def _get_option_price_from_data(
        self,
        intraday_data: list,
        bar_time: datetime,
        is_entry: bool = True,
    ) -> Optional[float]:
        """
        Get option price from intraday data at specific time.

        Uses interpolation for accurate pricing between data points.
        """
        if not intraday_data:
            return None

        price = self._option_pricing.interpolate_option_price(intraday_data, bar_time)
        return price

    async def _get_option_price(
        self,
        config: "StrategyConfig",
        trade,
        bar: dict,
        intraday_data: list,
        is_entry: bool = False,
    ) -> Optional[float]:
        """
        Get option price for a trade at given bar.

        Uses real ThetaData prices with interpolation.
        NO FALLBACK - returns None if real data unavailable.

        Args:
            config: Strategy config
            trade: Current trade
            bar: Current bar
            intraday_data: Intraday option data from ThetaData
            is_entry: Whether this is an entry price lookup

        Returns:
            Option price from ThetaData, or None if unavailable
        """
        bar_time = bar["timestamp"]

        # ONLY use real ThetaData - no fallback
        price = await self._get_option_price_from_data(
            intraday_data=intraday_data,
            bar_time=bar_time,
            is_entry=is_entry,
        )

        if price is not None and price > 0:
            return price

        # No fallback - log and return None
        logger.debug(f"No ThetaData price available at {bar_time}")
        return None

    async def _check_exit(
        self,
        trade,
        bar: dict,
        config: "StrategyConfig",
        intraday_data: list,
    ) -> Optional[dict]:
        """Check if exit conditions are met using real option prices.

        IMPORTANT: Uses target/stop prices for exits, not current price.
        This simulates limit orders that fill at the specified price,
        not market orders that fill at whatever the current price is.

        For VWAP fade strategies, uses underlying-based exits:
        - Exit at VWAP reversion (underlying touches VWAP)
        - Exit at partial reversion (e.g., 50% back to VWAP)
        - Stop at max adverse excursion (underlying moves further from VWAP)
        - Time-based deadline (e.g., 10:30 AM)

        Returns None if no exit triggered or if ThetaData unavailable.
        """
        current_option_price = await self._get_option_price(
            config=config,
            trade=trade,
            bar=bar,
            intraday_data=intraday_data,
            is_entry=False,
        )

        # Skip exit check if no ThetaData available (can't determine price)
        if current_option_price is None:
            return None

        entry_price = trade.entry_price
        bar_time = bar["timestamp"]

        # Check for VWAP-based exit (for VWAP fade strategies)
        if hasattr(config, 'use_vwap_touch_exit') and config.use_vwap_touch_exit:
            vwap_exit = self._check_vwap_exit(trade, bar, config, current_option_price)
            if vwap_exit:
                return vwap_exit

        # Profit target - apply exit slippage (receiving bid vs mid)
        if config.profit_target_pct:
            target_price = entry_price * (1 + config.profit_target_pct)
            if current_option_price >= target_price:
                # Apply exit slippage even on limit orders (bid-ask spread)
                exit_slippage = max(
                    target_price * config.slippage_exit_pct,
                    config.min_slippage_dollars
                )
                return {"price": target_price - exit_slippage, "reason": "profit_target"}

        # Stop loss - extra slippage on stops (market order in fast-moving conditions)
        stop_price = entry_price * (1 - config.stop_loss_pct)
        if current_option_price <= stop_price:
            # Stop orders get base exit slippage + extra stop slippage
            total_slip_pct = config.slippage_exit_pct + config.slippage_stop_extra_pct
            exit_slippage = max(
                stop_price * total_slip_pct,
                config.min_slippage_dollars
            )
            return {"price": stop_price - exit_slippage, "reason": "stop_loss"}

        return None

    def _check_vwap_exit(
        self,
        trade,
        bar: dict,
        config: "StrategyConfig",
        current_option_price: float,
    ) -> Optional[dict]:
        """
        Check VWAP-based exit conditions for VWAP fade strategies.

        Exit when underlying reverts toward VWAP, not based on option P&L.
        This matches the validated thesis: 90% reversion to VWAP.

        Args:
            trade: Current trade with entry_vwap and entry_underlying in metadata
            bar: Current bar with close price
            config: Strategy config with VWAP exit parameters
            current_option_price: Current option price for exit

        Returns:
            Exit dict with price and reason, or None if no exit
        """
        bar_time = bar["timestamp"]

        # Get VWAP info from trade metadata
        entry_vwap = getattr(trade, 'entry_vwap', None)
        entry_underlying = trade.underlying_price

        if entry_vwap is None or entry_vwap <= 0:
            return None

        current_underlying = bar["close"]

        # Calculate current deviation from VWAP
        entry_deviation = entry_underlying - entry_vwap  # Negative when below VWAP (call entry)
        current_deviation = current_underlying - entry_vwap

        # 1. Check time-based deadline exit (e.g., 10:30 AM)
        if hasattr(config, 'exit_deadline_hour_utc'):
            deadline_hour = config.exit_deadline_hour_utc
            deadline_minute = getattr(config, 'exit_deadline_minute_utc', 0)
            deadline_mins = deadline_hour * 60 + deadline_minute
            bar_mins = bar_time.hour * 60 + bar_time.minute

            if bar_mins >= deadline_mins:
                exit_slippage = max(
                    current_option_price * config.slippage_exit_pct,
                    config.min_slippage_dollars
                )
                return {
                    "price": current_option_price - exit_slippage,
                    "reason": "vwap_deadline"
                }

        # 2. Check VWAP touch exit (100% reversion)
        if hasattr(config, 'use_vwap_touch_exit') and config.use_vwap_touch_exit:
            # For calls (entry below VWAP): exit when price >= VWAP
            # For puts (entry above VWAP): exit when price <= VWAP
            from app.models.schemas import SignalType

            is_call = trade.signal_type == SignalType.BUY_CALL
            if is_call and current_underlying >= entry_vwap:
                exit_slippage = max(
                    current_option_price * config.slippage_exit_pct,
                    config.min_slippage_dollars
                )
                return {
                    "price": current_option_price - exit_slippage,
                    "reason": "vwap_touch"
                }
            elif not is_call and current_underlying <= entry_vwap:
                exit_slippage = max(
                    current_option_price * config.slippage_exit_pct,
                    config.min_slippage_dollars
                )
                return {
                    "price": current_option_price - exit_slippage,
                    "reason": "vwap_touch"
                }

        # 3. Check partial reversion exit (e.g., 50%)
        if hasattr(config, 'profit_target_reversion_pct'):
            reversion_pct = config.profit_target_reversion_pct / 100.0  # Convert to decimal

            if entry_deviation != 0:
                # Calculate how much of the deviation has been recovered
                reversion_ratio = 1 - (current_deviation / entry_deviation)

                if reversion_ratio >= reversion_pct:
                    exit_slippage = max(
                        current_option_price * config.slippage_exit_pct,
                        config.min_slippage_dollars
                    )
                    return {
                        "price": current_option_price - exit_slippage,
                        "reason": f"vwap_reversion_{int(reversion_pct * 100)}pct"
                    }

        # 4. Check underlying-based stop (max adverse excursion)
        # Stop if underlying moves FURTHER from VWAP (wrong direction)
        if hasattr(config, 'max_deviation_pct'):
            max_adverse_pct = config.max_deviation_pct / 100.0  # e.g., 1.0% max
            current_dev_pct = abs(current_deviation / entry_vwap)

            # Only stop if deviation increased (price went wrong way)
            entry_dev_pct = abs(entry_deviation / entry_vwap)
            if current_dev_pct > max_adverse_pct and current_dev_pct > entry_dev_pct:
                # Price moved further from VWAP - stop out
                total_slip_pct = config.slippage_exit_pct + getattr(config, 'slippage_stop_extra_pct', 0.02)
                exit_slippage = max(
                    current_option_price * total_slip_pct,
                    config.min_slippage_dollars
                )
                return {
                    "price": current_option_price - exit_slippage,
                    "reason": "vwap_stop_adverse"
                }

        return None

    def _calculate_trade_pnl(self, trade, config: "StrategyConfig") -> float:
        """Calculate P&L for a trade."""
        if trade.exit_price is None:
            return 0.0

        price_diff = trade.exit_price - trade.entry_price
        contracts = trade.total_contracts

        # P&L = (exit - entry) * 100 * contracts - commissions
        pnl = price_diff * 100 * contracts
        pnl -= config.commission_per_contract * contracts * 2  # Entry + exit

        return pnl

    def _calculate_metrics(self, trades: List, config: "StrategyConfig"):
        """Calculate performance metrics from trades."""
        from app.models.schemas import BacktestMetrics

        if not trades:
            return BacktestMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
            )

        pnls = [t.pnl for t in trades if t.pnl is not None]

        if not pnls:
            return BacktestMetrics(
                total_trades=len(trades),
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
            )

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) if pnls else 0.0
        avg_pnl = total_pnl / len(pnls) if pnls else 0.0

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Max drawdown
        equity = config.initial_capital
        peak = equity
        max_dd = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe ratio - using DAILY returns, not per-trade returns
        import statistics
        from collections import defaultdict

        # Group P&L by day for proper Sharpe calculation
        daily_pnl = defaultdict(float)
        for trade in trades:
            if trade.exit_time and trade.pnl is not None:
                trade_date = trade.exit_time.date() if hasattr(trade.exit_time, 'date') else trade.exit_time
                daily_pnl[trade_date] += trade.pnl

        if len(daily_pnl) > 1:
            daily_returns = [pnl / config.initial_capital for pnl in daily_pnl.values()]
            avg_daily_return = statistics.mean(daily_returns)
            std_daily_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.001
            # Annualize: multiply avg by 252, multiply std by sqrt(252)
            sharpe = (avg_daily_return / std_daily_return) * (252 ** 0.5) if std_daily_return > 0 else 0.0
        else:
            sharpe = 0.0

        return BacktestMetrics(
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
        )

    def _build_equity_curve(
        self, trades: List, initial_capital: float
    ) -> List[tuple]:
        """Build equity curve from trades as list of (timestamp, equity) tuples."""
        from datetime import datetime as dt, timezone

        start_time = dt(2000, 1, 1, tzinfo=timezone.utc)
        curve = [(start_time, initial_capital)]
        equity = initial_capital

        for trade in trades:
            if trade.pnl is not None and trade.exit_time is not None:
                equity += trade.pnl
                curve.append((trade.exit_time, equity))

        return curve

    def _convert_trades(self, simulated_trades: List) -> List:
        """Convert SimulatedTrade objects to BacktestTrade format."""
        from app.models.schemas import BacktestTrade

        trades = []
        for trade in simulated_trades:
            if trade.exit_time is None or trade.exit_price is None:
                continue

            entry_price = trade.entry_price
            exit_price = trade.exit_price
            pnl = trade.pnl or 0.0
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0

            trades.append(BacktestTrade(
                entry_date=trade.entry_time,
                exit_date=trade.exit_time,
                signal_type=trade.signal_type,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_dollars=pnl,
                pnl_percent=pnl_pct,
                exit_reason=trade.exit_reason or "unknown",
                strike=trade.strike,
            ))

        return trades

    async def _fetch_bars(self, config: "StrategyConfig") -> List[dict]:
        """
        Fetch historical bars using AlpacaClient.

        Fetches in chunks to handle API limits.
        Filters to regular market hours (9:30 AM - 4:00 PM ET) for backtesting accuracy.
        """
        from zoneinfo import ZoneInfo

        start_dt = datetime.combine(config.start_date, datetime.min.time())
        end_dt = datetime.combine(config.end_date, datetime.max.time())

        all_bars = []
        current_start = start_dt

        # Fetch in 5-day chunks
        while current_start < end_dt:
            chunk_end = min(current_start + timedelta(days=5), end_dt)

            try:
                bars = await self._alpaca_client.get_stock_bars(
                    symbol=config.symbol,
                    timeframe=config.primary_timeframe,
                    start=current_start,
                    end=chunk_end,
                    limit=10000,
                )
                all_bars.extend(bars)
            except Exception as e:
                logger.warning(f"Error fetching bars for {current_start}: {e}")

            current_start = chunk_end

        # Filter to regular market hours (9:30 AM - 4:00 PM ET)
        # ThetaData only has options data during regular hours
        et_tz = ZoneInfo("America/New_York")
        market_open_minutes = 9 * 60 + 30  # 9:30 AM = 570 minutes
        market_close_minutes = 16 * 60  # 4:00 PM = 960 minutes

        filtered_bars = []
        for bar in all_bars:
            ts = bar["timestamp"]
            # Convert to ET
            if ts.tzinfo is not None:
                et_time = ts.astimezone(et_tz)
            else:
                et_time = ts

            bar_minutes = et_time.hour * 60 + et_time.minute
            if market_open_minutes <= bar_minutes < market_close_minutes:
                filtered_bars.append(bar)

        logger.info(f"Filtered {len(all_bars)} bars to {len(filtered_bars)} regular market hours bars")
        return filtered_bars

    def _apply_filters(
        self,
        signals: List["TradingSignal"],
        bars: List[dict],
        filters: list,
        config: "StrategyConfig",
    ) -> List["TradingSignal"]:
        """
        Apply entry filters to signals.

        Replaces filtered signals with NO_SIGNAL.
        """
        from strategies.base.strategy import TradingSignal

        filtered_signals = []

        for i, signal in enumerate(signals):
            if signal.signal_type == "NO_SIGNAL":
                filtered_signals.append(signal)
                continue

            bar = bars[i] if i < len(bars) else {}

            allow_entry = True
            rejection_reason = ""

            for f in filters:
                if not f.should_allow_entry(
                    signal.signal_type,
                    bar,
                    signal.timestamp,
                    **signal.metadata,
                ):
                    allow_entry = False
                    rejection_reason = f.get_rejection_reason()
                    break

            if allow_entry:
                filtered_signals.append(signal)
            else:
                filtered_signals.append(
                    TradingSignal(
                        signal_type="NO_SIGNAL",
                        timestamp=signal.timestamp,
                        price=signal.price,
                        reason=f"Filtered: {rejection_reason}",
                        metadata=signal.metadata,
                    )
                )

        return filtered_signals

    def _empty_result(self, config: "StrategyConfig"):
        """Return empty result when no data available."""
        from app.models.schemas import BacktestResult, BacktestMetrics

        return BacktestResult(
            trades=[],
            metrics=BacktestMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                profit_factor=0.0,
            ),
            equity_curve=[],
        )
