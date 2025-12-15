from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, Callable, Dict, Any
import logging
import math

from app.services.alpaca_client import AlpacaClient
from app.services.indicators import calculate_rsi_series, calculate_sma_series
from app.services.signal_generator import SignalGenerator
from app.services.theta_data import ThetaDataClient, ThetaDataError
from app.models.schemas import SignalType, BacktestTrade, BacktestMetrics, BacktestResult
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration."""

    symbol: str = "SPY"
    start_date: date = None
    end_date: date = None
    rsi_period: int = 14
    rsi_sma_period: int = 14

    # RSI thresholds for entry signals
    rsi_oversold: float = 30.0       # Entry: RSI < 30 for calls
    rsi_overbought: float = 70.0     # Entry: RSI > 70 for puts

    # RSI thresholds for strong signals (3 contracts)
    rsi_strong_oversold: float = 20.0    # Strong: RSI < 20 for calls
    rsi_strong_overbought: float = 80.0  # Strong: RSI > 80 for puts

    # RSI convergence exit thresholds
    rsi_convergence_call_exit: float = 40.0  # Exit CALL when RSI >= 40
    rsi_convergence_put_exit: float = 60.0   # Exit PUT when RSI <= 60

    # Entry time restriction
    entry_cutoff_hour_utc: int = 19  # 2:00 PM ET = 19:00 UTC

    profit_target_dollars: float = 0.50       # $0.50 profit per contract
    profit_target_pct: float = None          # Percentage-based profit target (overrides dollars if set)
    rider_profit_target_dollars: float = 1.00 # $1.00 profit for rider contract
    stop_loss_pct: float = 0.40              # 40% stop loss from avg entry price
    avg_down_trigger_pct: float = 0.10       # Add contract every -10% from original entry
    max_add_ons: int = 3                     # Max 3 add-ons (4 contracts total)
    max_position_move_pct: float = 0.40      # Skip trading if option moved 40%+ from open
    initial_capital: float = 10000.0
    slippage_pct: float = 0.01  # 1% slippage on options
    commission_per_contract: float = 0.65
    max_hold_bars: int = 3  # Maximum bars to hold (3 mins for 1-min bars)

    # Dual-timeframe configuration
    dual_timeframe_enabled: bool = True       # Enable dual-timeframe confirmation
    primary_timeframe: str = "5Min"           # Primary timeframe for trigger signal
    confirmation_timeframe: str = "3Min"      # Confirmation timeframe
    parallel_mode: bool = False               # Run 3-min and 5-min strategies independently in parallel

    # SPX / contract scaling
    contract_multiplier: int = 1              # Multiply contracts (10 for SPX simulation via SPY)
    strike_interval: float = 1.0              # Strike price interval ($1 for SPY, $5 for SPX)
    option_symbol: str = ""                   # Option root symbol (e.g., "SPXW" for SPX options, defaults to symbol)
    underlying_multiplier: float = 1.0        # Multiplier for underlying price to option strike (10.0 for SPY->SPXW)
    strike_offset: float = 2.0                # Strike offset from ATM (positive=OTM, negative=ITM, e.g., -2 = $2 ITM)


@dataclass
class SimulatedTrade:
    """Trade during simulation."""

    entry_time: datetime
    entry_price: float
    signal_type: SignalType
    underlying_price: float
    profit_target: float
    stop_loss: float
    entry_bar_index: int = 0  # Bar index when trade was entered
    entry_rsi: float = 0.0  # RSI at entry for tracking signal completion
    strike: float = 0.0  # Option strike price
    expiration: Optional[date] = None  # Option expiration date (0DTE)
    intraday_prices: list = field(default_factory=list)  # Hourly price snapshots from ThetaData
    rsi_history: list = field(default_factory=list)  # [(timestamp, rsi, option_price), ...] for debugging
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    # Contract scaling fields
    initial_contracts: int = 1      # Contracts at entry (1 for normal, 3 for strong signal)
    remaining_contracts: int = 1    # Contracts still open
    partial_tp_taken: bool = False  # True after taking profit on 2 contracts
    partial_tp_pnl: float = 0.0     # P&L realized from partial TP
    # Averaging down fields
    total_contracts: int = 1        # Total contracts including add-ons
    avg_entry_price: float = 0.0    # Weighted average entry price
    add_on_entries: list = field(default_factory=list)  # [(price, contracts, timestamp), ...]
    # Parallel mode tracking
    timeframe: str = ""             # Which timeframe generated this trade (e.g., "3Min", "5Min")
    # Settlement tracking
    settlement_underlying: float = 0.0  # Underlying price at settlement (for EOD expiration)


class BacktestService:
    """
    Historical backtesting engine.

    Simulates RSI strategy on historical data with:
    - Real option pricing via ThetaData (or fallback to simplified model)
    - Slippage and commission
    - Intraday exit logic
    """

    def __init__(
        self,
        alpaca_client: Optional[AlpacaClient] = None,
        theta_client: Optional[ThetaDataClient] = None,
    ):
        self.client = alpaca_client or AlpacaClient()
        self.theta_client = theta_client
        self._theta_available = False
        # Cache for option prices: {(symbol, exp, strike, right, date): price}
        self._option_price_cache: Dict[tuple, float] = {}

    async def run_backtest(
        self,
        config: BacktestConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BacktestResult:
        """
        Run historical backtest.

        Args:
            config: Backtest configuration
            progress_callback: Optional callback for progress updates (current, total)

        Returns:
            BacktestResult with trades, metrics, and equity curve
        """
        # Determine effective option symbol (defaults to config.symbol if not specified)
        option_symbol = config.option_symbol if config.option_symbol else config.symbol

        logger.info(
            f"Starting backtest: {config.symbol} from {config.start_date} to {config.end_date}"
        )
        if option_symbol != config.symbol:
            logger.info(f"Using {option_symbol} options with {config.symbol} underlying")

        # Store option_symbol for use in simulation methods
        self._option_symbol = option_symbol

        # Initialize ThetaData client if not provided
        if self.theta_client is None:
            self.theta_client = ThetaDataClient()

        # Check if ThetaData is available
        self._theta_available = await self._check_theta_availability()
        if self._theta_available:
            logger.info("ThetaData available - using real historical option prices")
        else:
            logger.warning("ThetaData unavailable - using estimated option prices")

        # Fetch historical data (returns tuple for dual-timeframe support)
        bars_data = await self._fetch_historical_bars(config)

        if config.parallel_mode:
            # PARALLEL MODE: Run 3-min and 5-min strategies independently
            bars_5min, bars_3min = bars_data

            if not bars_5min or not bars_3min:
                logger.warning("No historical data available")
                return self._empty_result(config)

            logger.info(f"PARALLEL MODE: Running 3-min and 5-min strategies independently")
            logger.info(f"Fetched {len(bars_5min)} 5-min bars and {len(bars_3min)} 3-min bars")

            signal_generator = SignalGenerator(
                rsi_oversold=config.rsi_oversold,
                rsi_overbought=config.rsi_overbought,
            )

            # Process 3-min strategy
            closes_3min = [bar["close"] for bar in bars_3min]
            timestamps_3min = [bar["timestamp"] for bar in bars_3min]
            rsi_values_3min = calculate_rsi_series(closes_3min, config.rsi_period)
            sma_values_3min = calculate_sma_series(rsi_values_3min, config.rsi_sma_period)
            signals_3min = signal_generator.evaluate_series(
                rsi_values=rsi_values_3min,
                sma_values=sma_values_3min,
                close_prices=closes_3min,
                timestamps=timestamps_3min,
            )

            # Process 5-min strategy
            closes_5min = [bar["close"] for bar in bars_5min]
            timestamps_5min = [bar["timestamp"] for bar in bars_5min]
            rsi_values_5min = calculate_rsi_series(closes_5min, config.rsi_period)
            sma_values_5min = calculate_sma_series(rsi_values_5min, config.rsi_sma_period)
            signals_5min = signal_generator.evaluate_series(
                rsi_values=rsi_values_5min,
                sma_values=sma_values_5min,
                close_prices=closes_5min,
                timestamps=timestamps_5min,
            )

            # Simulate trades for both timeframes independently
            trades = await self._simulate_trades_parallel(
                bars_3min=bars_3min,
                signals_3min=signals_3min,
                rsi_values_3min=rsi_values_3min,
                sma_values_3min=sma_values_3min,
                bars_5min=bars_5min,
                signals_5min=signals_5min,
                rsi_values_5min=rsi_values_5min,
                sma_values_5min=sma_values_5min,
                config=config,
                progress_callback=progress_callback,
            )

        elif config.dual_timeframe_enabled:
            # Dual-timeframe mode: 5-min trigger + 3-min confirmation
            bars_5min, bars_3min = bars_data

            if not bars_5min or not bars_3min:
                logger.warning("No historical data available")
                return self._empty_result(config)

            logger.info(f"Fetched {len(bars_5min)} 5-min bars and {len(bars_3min)} 3-min bars")

            # Calculate indicators for 5-min (PRIMARY - trigger)
            closes_5min = [bar["close"] for bar in bars_5min]
            timestamps_5min = [bar["timestamp"] for bar in bars_5min]
            rsi_values_5min = calculate_rsi_series(closes_5min, config.rsi_period)
            sma_values_5min = calculate_sma_series(rsi_values_5min, config.rsi_sma_period)

            # Build 5-min lookup for fast access
            lookup_5min = self._build_5min_lookup(bars_5min, rsi_values_5min, sma_values_5min)

            # Generate signals using 5-min bars (PRIMARY trigger)
            signal_generator = SignalGenerator(
                rsi_oversold=config.rsi_oversold,
                rsi_overbought=config.rsi_overbought,
            )
            signals_5min = signal_generator.evaluate_series(
                rsi_values=rsi_values_5min,
                sma_values=sma_values_5min,
                close_prices=closes_5min,
                timestamps=timestamps_5min,
            )

            # Build 5-min signal lookup
            signal_lookup_5min = {}
            for sig in signals_5min:
                ts = sig.timestamp
                if hasattr(ts, 'replace'):
                    ts = ts.replace(second=0, microsecond=0)
                signal_lookup_5min[ts] = sig

            # Calculate indicators for 3-min (CONFIRMATION)
            closes_3min = [bar["close"] for bar in bars_3min]
            rsi_values_3min = calculate_rsi_series(closes_3min, config.rsi_period)
            sma_values_3min = calculate_sma_series(rsi_values_3min, config.rsi_sma_period)

            # Simulate trades using dual-timeframe logic
            trades = await self._simulate_trades_dual_timeframe(
                bars_3min=bars_3min,
                rsi_values_3min=rsi_values_3min,
                sma_values_3min=sma_values_3min,
                lookup_5min=lookup_5min,
                signal_lookup_5min=signal_lookup_5min,
                config=config,
                progress_callback=progress_callback,
            )
        else:
            # Legacy single-timeframe mode
            bars = bars_data[0]

            if not bars:
                logger.warning("No historical data available")
                return self._empty_result(config)

            logger.info(f"Fetched {len(bars)} bars")

            # Extract price data
            closes = [bar["close"] for bar in bars]
            timestamps = [bar["timestamp"] for bar in bars]

            # Calculate indicators
            rsi_values = calculate_rsi_series(closes, config.rsi_period)
            sma_values = calculate_sma_series(rsi_values, config.rsi_sma_period)

            # Generate signals
            signal_generator = SignalGenerator(
                rsi_oversold=config.rsi_oversold,
                rsi_overbought=config.rsi_overbought,
            )

            signals = signal_generator.evaluate_series(
                rsi_values=rsi_values,
                sma_values=sma_values,
                close_prices=closes,
                timestamps=timestamps,
            )

            # Simulate trades
            trades = await self._simulate_trades(
                bars=bars,
                signals=signals,
                rsi_values=rsi_values,
                sma_values=sma_values,
                config=config,
                progress_callback=progress_callback,
            )

        # Calculate metrics
        metrics = self._calculate_metrics(trades, config)

        # Build equity curve
        equity_curve = self._build_equity_curve(trades, config.initial_capital)

        logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"Win rate: {metrics.win_rate:.1%}, Total P&L: ${metrics.total_pnl:.2f}"
        )

        return BacktestResult(
            trades=[self._to_backtest_trade(t) for t in trades],
            metrics=metrics,
            equity_curve=equity_curve,
        )

    async def _fetch_historical_bars(
        self, config: BacktestConfig
    ) -> tuple[list[dict], Optional[list[dict]]]:
        """
        Fetch historical bars for backtesting.

        Returns:
            Tuple of (primary_bars, confirmation_bars)
            - If dual_timeframe_enabled: (5-min bars, 3-min bars)
            - If disabled: (3-min bars, None)
        """
        import asyncio

        start_dt = datetime.combine(config.start_date, datetime.min.time())
        end_dt = datetime.combine(config.end_date, datetime.max.time())

        async def fetch_bars_for_timeframe(timeframe: str) -> list[dict]:
            """Fetch bars for a specific timeframe in chunks."""
            all_bars = []
            current_start = start_dt
            while current_start < end_dt:
                chunk_end = min(current_start + timedelta(days=5), end_dt)
                bars = await self.client.get_stock_bars(
                    symbol=config.symbol,
                    timeframe=timeframe,
                    start=current_start,
                    end=chunk_end,
                    limit=10000,
                )
                all_bars.extend(bars)
                current_start = chunk_end
            return all_bars

        if config.parallel_mode or config.dual_timeframe_enabled:
            # Fetch both timeframes in parallel
            primary_bars, confirmation_bars = await asyncio.gather(
                fetch_bars_for_timeframe("5Min"),  # 5-min
                fetch_bars_for_timeframe("3Min")   # 3-min
            )
            return (primary_bars, confirmation_bars)
        else:
            # Single timeframe mode - use configured primary timeframe
            bars = await fetch_bars_for_timeframe(config.primary_timeframe)
            return (bars, None)

    async def _check_theta_availability(self) -> bool:
        """Check if ThetaData (ThetaTerminal) is available."""
        try:
            # Try to make a simple request to ThetaTerminal to check if it's running
            # We don't use get_expirations because it filters to future dates only
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{settings.theta_data_base_url}/v3/option/list/expirations",
                    params={"symbol": "SPY", "format": "json"},
                )
                if response.status_code == 200:
                    data = response.json()
                    return "response" in data and len(data["response"]) > 0
            return False
        except Exception as e:
            logger.debug(f"ThetaData not available: {e}")
            return False

    def _calculate_profit_target(
        self, entry_price: float, config: BacktestConfig
    ) -> float:
        """
        Calculate profit target price based on config.

        If profit_target_pct is set, use percentage-based target.
        Otherwise, use fixed dollar target.
        """
        if config.profit_target_pct is not None:
            return entry_price * (1 + config.profit_target_pct)
        else:
            return entry_price + config.profit_target_dollars

    def _get_strike_for_trade(
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
                           e.g., +2 = $2 OTM, -2 = $2 ITM
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
        # Round to nearest strike interval (e.g., $1 for SPY, $5 for SPX)
        return round(strike / strike_interval) * strike_interval

    def _check_option_moved_too_much(
        self,
        intraday_prices: list,
        current_time: datetime,
        max_move_pct: float,
    ) -> bool:
        """
        Check if option has moved +/-40% from market open (9:30 AM ET).

        Args:
            intraday_prices: List of (timestamp, open, high, low, close, volume) tuples
            current_time: Current bar timestamp
            max_move_pct: Maximum allowed move (e.g., 0.40 for 40%)

        Returns:
            True if option moved too much (should skip trading), False otherwise
        """
        if not intraday_prices:
            return False

        # Find 9:30 AM ET opening price (14:30 UTC)
        open_price = None
        for ts, o, h, l, c, v in intraday_prices:
            if hasattr(ts, 'hour'):
                # Check for 9:30 AM ET = 14:30 UTC
                if ts.hour == 14 and ts.minute == 30:
                    open_price = o
                    break
            elif isinstance(ts, str):
                # Parse ISO timestamp
                try:
                    from datetime import datetime as dt
                    parsed_ts = dt.fromisoformat(ts.replace('Z', '+00:00'))
                    if parsed_ts.hour == 14 and parsed_ts.minute == 30:
                        open_price = o
                        break
                except:
                    continue

        if open_price is None or open_price <= 0:
            return False

        # Get current price from intraday data
        current_price = None
        for ts, o, h, l, c, v in intraday_prices:
            price_ts = ts
            if isinstance(ts, str):
                try:
                    from datetime import datetime as dt
                    price_ts = dt.fromisoformat(ts.replace('Z', '+00:00'))
                except:
                    continue
            if price_ts <= current_time:
                current_price = c  # Use close of most recent bar at or before current time

        if current_price is None:
            return False

        # Calculate percentage move from open
        move_pct = abs(current_price - open_price) / open_price

        return move_pct >= max_move_pct

    def _is_downtrend(self, bars: list[dict], current_index: int, lookback: int) -> bool:
        """
        Check if SPY is in downtrend (lower highs AND lower lows).
        Compares peaks/troughs in first half vs second half of lookback period.
        """
        if current_index < lookback:
            return False

        # Split lookback into first half (older) and second half (recent)
        mid = lookback // 2
        first_half_start = current_index - lookback
        first_half_end = current_index - mid
        second_half_start = current_index - mid
        second_half_end = current_index + 1

        # Get highs and lows for each half
        first_highs = [bars[i].get("high", bars[i]["close"]) for i in range(first_half_start, first_half_end)]
        first_lows = [bars[i].get("low", bars[i]["close"]) for i in range(first_half_start, first_half_end)]
        second_highs = [bars[i].get("high", bars[i]["close"]) for i in range(second_half_start, second_half_end)]
        second_lows = [bars[i].get("low", bars[i]["close"]) for i in range(second_half_start, second_half_end)]

        if not first_highs or not second_highs:
            return False

        # Downtrend: recent high < old high AND recent low < old low
        lower_high = max(second_highs) < max(first_highs)
        lower_low = min(second_lows) < min(first_lows)

        return lower_high and lower_low

    def _is_uptrend(self, bars: list[dict], current_index: int, lookback: int) -> bool:
        """
        Check if SPY is in uptrend (higher highs AND higher lows).
        Compares peaks/troughs in first half vs second half of lookback period.
        """
        if current_index < lookback:
            return False

        # Split lookback into first half (older) and second half (recent)
        mid = lookback // 2
        first_half_start = current_index - lookback
        first_half_end = current_index - mid
        second_half_start = current_index - mid
        second_half_end = current_index + 1

        # Get highs and lows for each half
        first_highs = [bars[i].get("high", bars[i]["close"]) for i in range(first_half_start, first_half_end)]
        first_lows = [bars[i].get("low", bars[i]["close"]) for i in range(first_half_start, first_half_end)]
        second_highs = [bars[i].get("high", bars[i]["close"]) for i in range(second_half_start, second_half_end)]
        second_lows = [bars[i].get("low", bars[i]["close"]) for i in range(second_half_start, second_half_end)]

        if not first_highs or not second_highs:
            return False

        # Uptrend: recent high > old high AND recent low > old low
        higher_high = max(second_highs) > max(first_highs)
        higher_low = min(second_lows) > min(first_lows)

        return higher_high and higher_low

    def _build_5min_lookup(
        self,
        bars_5min: list[dict],
        rsi_values_5min: list[float],
        sma_values_5min: list[float],
    ) -> dict:
        """
        Build lookup dict: 5-min bar timestamp -> (rsi, sma, bar_data).

        This allows O(1) lookup of 5-min indicators given a timestamp.
        """
        lookup = {}
        for i, bar in enumerate(bars_5min):
            ts = bar["timestamp"]
            # Normalize to minute precision (remove seconds/microseconds)
            if hasattr(ts, 'replace'):
                ts = ts.replace(second=0, microsecond=0)
            lookup[ts] = (rsi_values_5min[i], sma_values_5min[i], bar)
        return lookup

    def _get_aligned_5min_timestamp(self, timestamp: datetime) -> datetime:
        """Get the aligned 5-min bar timestamp for a given timestamp."""
        ts = timestamp
        if hasattr(timestamp, 'replace'):
            ts = timestamp.replace(second=0, microsecond=0)
        minute = ts.minute
        five_min_boundary = (minute // 5) * 5
        return ts.replace(minute=five_min_boundary)

    def _get_5min_data_for_timestamp(
        self,
        timestamp: datetime,
        lookup_5min: dict,
    ) -> Optional[tuple[float, float, dict]]:
        """
        Get the most recent completed 5-min bar data for a given timestamp.

        Args:
            timestamp: Current bar timestamp (from 3-min bars)
            lookup_5min: Pre-built lookup dict

        Returns:
            (rsi_5min, sma_5min, bar_5min) or None if no valid bar found
        """
        aligned_ts = self._get_aligned_5min_timestamp(timestamp)

        # Look up in the pre-built dict
        if aligned_ts in lookup_5min:
            return lookup_5min[aligned_ts]

        # Fallback: search backwards for closest 5-min bar (handles gaps)
        for offset in range(5, 60, 5):
            check_ts = aligned_ts - timedelta(minutes=offset)
            if check_ts in lookup_5min:
                return lookup_5min[check_ts]

        return None

    async def _get_real_option_price(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        option_type: str,
        target_date: date,
        entry_time: bool = True,
    ) -> Optional[float]:
        """
        Get real historical option price from ThetaData.

        Uses mid price (bid+ask)/2 for accurate simulation.
        For 0DTE options, gets the first available price of the day (entry) or
        last price (exit).

        Args:
            symbol: Underlying symbol (e.g., "SPY")
            expiration: Option expiration date
            strike: Strike price
            option_type: "call" or "put"
            target_date: Date to get price for
            entry_time: If True, get first price of day; if False, get last price

        Returns:
            Mid price or None if unavailable
        """
        if not self._theta_available:
            return None

        # Check cache first
        right = "C" if option_type == "call" else "P"
        time_key = "entry" if entry_time else "exit"
        cache_key = (symbol, expiration, strike, right, target_date, time_key)
        if cache_key in self._option_price_cache:
            return self._option_price_cache[cache_key]

        try:
            # Fetch all hourly data for the day
            data = await self._fetch_intraday_option_data(
                symbol=symbol,
                expiration=expiration,
                strike=strike,
                right=right,
                target_date=target_date,
            )

            if data:
                # Get first or last valid price
                if entry_time:
                    # Get first valid price (market open)
                    for point in data:
                        if point.get("midpoint", 0) > 0:
                            price = float(point["midpoint"])
                            self._option_price_cache[cache_key] = price
                            return price
                else:
                    # Get last valid price (EOD)
                    for point in reversed(data):
                        if point.get("midpoint", 0) > 0:
                            price = float(point["midpoint"])
                            self._option_price_cache[cache_key] = price
                            return price

        except ThetaDataError as e:
            logger.debug(f"ThetaData error for {symbol} {strike} {option_type}: {e}")
        except Exception as e:
            logger.debug(f"Error getting option price: {e}")

        return None

    async def _fetch_intraday_option_data(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
        target_date: date,
    ) -> list[dict]:
        """
        Fetch all intraday 1-minute OHLC data for an option contract.

        Uses ThetaData v3 OHLC endpoint with 1-minute intervals.

        Returns list of data points with timestamp, midpoint, open, high, low, close.
        """
        import httpx

        try:
            # v3 API uses strike in dollars with decimal (e.g., "580.000")
            # and right as "call" or "put"
            right_str = "call" if right.upper() == "C" else "put"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{settings.theta_data_base_url}/v3/option/history/ohlc",
                    params={
                        "symbol": symbol.upper(),
                        "expiration": expiration.strftime("%Y%m%d"),
                        "strike": f"{strike:.3f}",  # Strike in dollars with 3 decimals
                        "right": right_str,
                        "date": target_date.strftime("%Y%m%d"),
                        "interval": "1m",  # 1 minute bars
                        "format": "json",
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    # v3 response format: {"response": [{"contract": {...}, "data": [{...}, ...]}, ...]}
                    if "response" in data and data["response"]:
                        results = []
                        # The response contains contract info + data array
                        contract_response = data["response"][0] if data["response"] else {}
                        ohlc_data = contract_response.get("data", [])

                        for row in ohlc_data:
                            if not row:
                                continue

                            # Parse timestamp (format: "2024-10-01T09:30:00")
                            ts_str = row.get("timestamp", "")
                            if not ts_str:
                                continue

                            try:
                                bar_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            except ValueError:
                                continue

                            open_price = row.get("open", 0)
                            high_price = row.get("high", 0)
                            low_price = row.get("low", 0)
                            close_price = row.get("close", 0)

                            # Skip rows with no price data
                            if not any([open_price, high_price, low_price, close_price]):
                                continue

                            # Calculate midpoint from high/low or use close
                            midpoint = (high_price + low_price) / 2 if high_price and low_price else close_price

                            results.append({
                                "timestamp": bar_time,
                                "midpoint": midpoint,
                                "open": open_price,
                                "high": high_price,
                                "low": low_price,
                                "close": close_price,
                            })
                        return results

        except Exception as e:
            logger.debug(f"Error fetching intraday option data: {e}")

        return []

    def _interpolate_option_price(
        self,
        intraday_data: list[dict],
        bar_time: datetime,
    ) -> Optional[float]:
        """
        Interpolate option price at bar_time from hourly snapshots.

        Uses linear interpolation between the two surrounding hourly data points.
        Returns closest available price if at boundary.

        Args:
            intraday_data: List of hourly snapshots with timestamp and midpoint
            bar_time: The time to get price for (assumed UTC from Alpaca)

        Returns:
            Interpolated mid price or None if no data available
        """
        if not intraday_data:
            return None

        # Parse timestamps and filter valid data points
        valid_points = []
        for point in intraday_data:
            mid = point.get("midpoint", 0)
            if mid <= 0:
                continue

            ts_str = point.get("timestamp", "")
            if not ts_str:
                continue

            try:
                # Parse timestamp (format: "2024-10-15T10:30:00")
                # ThetaData returns times in ET (Eastern Time), no timezone info
                if isinstance(ts_str, str):
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    ts = ts_str
                valid_points.append((ts, mid))
            except (ValueError, TypeError):
                continue

        if not valid_points:
            return None

        # Sort by timestamp
        valid_points.sort(key=lambda x: x[0])

        # Convert bar_time from UTC to ET for comparison with ThetaData timestamps
        # ThetaData returns timestamps in ET without timezone info
        # Alpaca bars come in UTC
        from zoneinfo import ZoneInfo
        if bar_time.tzinfo is not None:
            # Convert UTC to ET
            et_tz = ZoneInfo("America/New_York")
            bar_time_et = bar_time.astimezone(et_tz).replace(tzinfo=None)
        else:
            # Already naive, assume it's in the right timezone
            bar_time_et = bar_time

        # Make bar_time timezone-naive for comparison
        bar_time_naive = bar_time_et

        # Find surrounding points for interpolation
        before = None
        after = None

        for ts, mid in valid_points:
            ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
            if ts_naive <= bar_time_naive:
                before = (ts_naive, mid)
            elif after is None:
                after = (ts_naive, mid)
                break

        # Return closest or interpolated price
        if before is None and after is None:
            return None
        elif before is None:
            return after[1]
        elif after is None:
            return before[1]
        else:
            # Linear interpolation
            before_ts, before_price = before
            after_ts, after_price = after

            total_seconds = (after_ts - before_ts).total_seconds()
            if total_seconds <= 0:
                return before_price

            elapsed_seconds = (bar_time_naive - before_ts).total_seconds()
            fraction = elapsed_seconds / total_seconds

            interpolated = before_price + fraction * (after_price - before_price)
            return max(0.01, interpolated)  # Price can't go below $0.01

    async def _simulate_trades(
        self,
        bars: list[dict],
        signals: list,
        rsi_values: list[float],
        sma_values: list[float],
        config: BacktestConfig,
        progress_callback: Optional[Callable] = None,
    ) -> list[SimulatedTrade]:
        """Simulate trade execution on historical data."""
        trades = []
        current_trade: Optional[SimulatedTrade] = None
        current_day: Optional[date] = None
        previous_bar: Optional[dict] = None  # Track previous bar for EOD closes

        for i, (bar, signal, current_rsi, current_sma) in enumerate(zip(bars, signals, rsi_values, sma_values)):
            if progress_callback and i % 1000 == 0:
                progress_callback(i, len(bars))

            bar_time = bar["timestamp"]
            bar_day = bar_time.date() if hasattr(bar_time, "date") else bar_time

            # Reset at start of new day (for 0DTE simulation)
            if current_day != bar_day:
                if current_trade and previous_bar is not None:
                    # Force close at end of previous day using PREVIOUS bar (last bar of that day)
                    prev_bar_time = previous_bar["timestamp"]
                    # Set settlement underlying price (scaled for SPXW)
                    current_trade.settlement_underlying = previous_bar["close"] * config.underlying_multiplier
                    # Try interpolated price first, fallback to estimate
                    exit_price = None
                    if current_trade.intraday_prices:
                        exit_price = self._interpolate_option_price(
                            current_trade.intraday_prices, prev_bar_time
                        )
                    if exit_price is None:
                        exit_price = self._estimate_option_price(
                            previous_bar["close"],
                            current_trade.underlying_price,
                            current_trade.entry_price,
                            current_trade.signal_type,
                        )
                    current_trade = self._close_trade(
                        trade=current_trade,
                        exit_time=prev_bar_time,  # Use previous bar's time
                        exit_price=exit_price,
                        exit_reason="end_of_day",
                        config=config,
                    )
                    trades.append(current_trade)
                    current_trade = None
                current_day = bar_day

            # Check exits for current trade
            if current_trade:
                # Track RSI history during trade for debugging
                option_price_for_history = None
                if current_trade.intraday_prices:
                    option_price_for_history = self._interpolate_option_price(
                        current_trade.intraday_prices, bar_time
                    )
                # Store: (timestamp, rsi, option_price, spy_price)
                current_trade.rsi_history.append((bar_time, current_rsi, option_price_for_history, bar["close"]))

                exit_reason, contracts_to_close, current_price = self._check_exit(
                    current_trade, bar, config, i, current_rsi, current_sma
                )

                # Averaging down: every -10% drop from ORIGINAL entry price
                if exit_reason is None and current_price is not None:
                    num_addons = len(current_trade.add_on_entries)
                    if num_addons < config.max_add_ons:
                        # Trigger at -10%, -20%, -30% from ORIGINAL entry
                        threshold_pct = (num_addons + 1) * config.avg_down_trigger_pct
                        trigger_price = current_trade.entry_price * (1 - threshold_pct)

                        if current_price <= trigger_price:
                            add_price = current_price * (1 + config.slippage_pct)
                            add_contracts = config.contract_multiplier  # Scale add-ons by multiplier
                            total_cost = (current_trade.avg_entry_price * current_trade.total_contracts) + (add_price * add_contracts)
                            current_trade.total_contracts += add_contracts
                            current_trade.remaining_contracts += add_contracts
                            current_trade.avg_entry_price = total_cost / current_trade.total_contracts
                            current_trade.add_on_entries.append((add_price, add_contracts, bar_time))
                            # Recalculate stop loss from NEW avg entry
                            current_trade.stop_loss = current_trade.avg_entry_price * (1 - config.stop_loss_pct)

                if exit_reason == "partial_tp":
                    # Close 2 contracts at +$0.50, keep 1 rider
                    partial_pnl = (current_price - current_trade.entry_price) * 2 * 100
                    partial_pnl -= config.commission_per_contract * 2  # Exit commission for 2
                    current_trade.partial_tp_pnl = partial_pnl
                    current_trade.partial_tp_taken = True
                    current_trade.remaining_contracts = 1
                    # Trade stays open - rider continues

                elif exit_reason in ("stop_loss", "rider_tp", "profit_target"):
                    # Full exit - close remaining contracts
                    current_trade = self._close_trade(
                        trade=current_trade,
                        exit_time=bar_time,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        config=config,
                        contracts_to_close=contracts_to_close,
                    )
                    trades.append(current_trade)
                    current_trade = None

            # Check for new entry (only if no position)
            if current_trade is None and signal.signal_type != SignalType.NO_SIGNAL:
                # Determine option type and strike
                option_type = "call" if signal.signal_type == SignalType.BUY_CALL else "put"
                strike = self._get_strike_for_trade(
                    bar["close"], option_type,
                    strike_offset=config.strike_offset,
                    strike_interval=config.strike_interval,
                    underlying_multiplier=config.underlying_multiplier,
                )
                expiration = bar_day  # 0DTE
                right = "C" if option_type == "call" else "P"

                # Pre-fetch ALL intraday data for this option (for exit price interpolation)
                intraday_data = []
                if self._theta_available:
                    intraday_data = await self._fetch_intraday_option_data(
                        symbol=self._option_symbol,
                        expiration=expiration,
                        strike=strike,
                        right=right,
                        target_date=bar_day,
                    )

                # Get entry price from intraday data (first valid price)
                entry_price = None
                if intraday_data:
                    entry_price = self._interpolate_option_price(intraday_data, bar_time)

                # Fallback to estimated price if ThetaData unavailable
                if entry_price is None:
                    entry_price = self._estimate_initial_option_price(
                        signal.signal_type, bar["close"]
                    )

                # Apply slippage
                entry_price *= 1 + config.slippage_pct

                # Determine contract count: 3 for strong signals, 1 for normal
                # Then multiply by contract_multiplier (10 for SPX simulation)
                if signal.signal_type == SignalType.BUY_CALL:
                    is_strong = current_rsi <= config.rsi_strong_oversold  # RSI < 20
                else:  # BUY_PUT
                    is_strong = current_rsi >= config.rsi_strong_overbought  # RSI > 80
                base_contracts = 3 if is_strong else 1
                contracts = base_contracts * config.contract_multiplier

                current_trade = SimulatedTrade(
                    entry_time=bar_time,
                    entry_price=entry_price,
                    signal_type=signal.signal_type,
                    underlying_price=bar["close"],
                    profit_target=self._calculate_profit_target(entry_price, config),
                    stop_loss=entry_price * (1 - config.stop_loss_pct),
                    entry_bar_index=i,  # Track entry bar for timeout check
                    entry_rsi=current_rsi,  # Track RSI at entry for signal completion
                    strike=strike,
                    expiration=expiration,
                    intraday_prices=intraday_data,  # Store for exit price interpolation
                    initial_contracts=contracts,
                    remaining_contracts=contracts,
                    total_contracts=contracts,  # Start with initial contracts
                    avg_entry_price=entry_price,  # Initial avg = entry price
                )

            # Track previous bar for EOD closes
            previous_bar = bar

        # Close any remaining trade
        if current_trade and bars:
            last_bar = bars[-1]
            # Set settlement underlying price (scaled for SPXW)
            current_trade.settlement_underlying = last_bar["close"] * config.underlying_multiplier
            # Try interpolated price first, fallback to estimate
            option_price = None
            if current_trade.intraday_prices:
                option_price = self._interpolate_option_price(
                    current_trade.intraday_prices, last_bar["timestamp"]
                )
            if option_price is None:
                option_price = self._estimate_option_price(
                    last_bar["close"],
                    current_trade.underlying_price,
                    current_trade.entry_price,
                    current_trade.signal_type,
                )
            current_trade = self._close_trade(
                trade=current_trade,
                exit_time=last_bar["timestamp"],
                exit_price=option_price,
                exit_reason="end_of_day",
                config=config,
            )
            trades.append(current_trade)

        return trades

    async def _simulate_trades_parallel(
        self,
        bars_3min: list[dict],
        signals_3min: list,
        rsi_values_3min: list[float],
        sma_values_3min: list[float],
        bars_5min: list[dict],
        signals_5min: list,
        rsi_values_5min: list[float],
        sma_values_5min: list[float],
        config: BacktestConfig,
        progress_callback: Optional[Callable] = None,
    ) -> list[SimulatedTrade]:
        """
        Simulate trades for 3-min and 5-min strategies running INDEPENDENTLY in parallel.
        Each strategy can have its own open position at the same time.
        """
        all_trades = []

        # Run 3-min strategy
        logger.info("Running 3-min strategy...")
        trades_3min = await self._simulate_trades_for_timeframe(
            bars=bars_3min,
            signals=signals_3min,
            rsi_values=rsi_values_3min,
            sma_values=sma_values_3min,
            config=config,
            timeframe_label="3Min",
            progress_callback=None,  # We'll do combined progress
        )
        for t in trades_3min:
            t.timeframe = "3Min"
        all_trades.extend(trades_3min)

        # Run 5-min strategy
        logger.info("Running 5-min strategy...")
        trades_5min = await self._simulate_trades_for_timeframe(
            bars=bars_5min,
            signals=signals_5min,
            rsi_values=rsi_values_5min,
            sma_values=sma_values_5min,
            config=config,
            timeframe_label="5Min",
            progress_callback=progress_callback,
        )
        for t in trades_5min:
            t.timeframe = "5Min"
        all_trades.extend(trades_5min)

        # Sort all trades by entry time
        all_trades.sort(key=lambda t: t.entry_time)

        logger.info(f"Parallel mode: {len(trades_3min)} trades from 3-min, {len(trades_5min)} trades from 5-min")
        return all_trades

    async def _simulate_trades_for_timeframe(
        self,
        bars: list[dict],
        signals: list,
        rsi_values: list[float],
        sma_values: list[float],
        config: BacktestConfig,
        timeframe_label: str,
        progress_callback: Optional[Callable] = None,
    ) -> list[SimulatedTrade]:
        """Simulate trades for a single timeframe (helper for parallel mode)."""
        trades = []
        current_trade: Optional[SimulatedTrade] = None
        current_day: Optional[date] = None
        previous_bar: Optional[dict] = None

        for i, (bar, signal, current_rsi, current_sma) in enumerate(zip(bars, signals, rsi_values, sma_values)):
            if progress_callback and i % 1000 == 0:
                progress_callback(i, len(bars))

            bar_time = bar["timestamp"]
            bar_day = bar_time.date() if hasattr(bar_time, "date") else bar_time

            # Reset at start of new day (for 0DTE simulation)
            if current_day != bar_day:
                if current_trade and previous_bar is not None:
                    prev_bar_time = previous_bar["timestamp"]
                    # Set settlement underlying price (scaled for SPXW)
                    current_trade.settlement_underlying = previous_bar["close"] * config.underlying_multiplier
                    exit_price = None
                    if current_trade.intraday_prices:
                        exit_price = self._interpolate_option_price(
                            current_trade.intraday_prices, prev_bar_time
                        )
                    if exit_price is None:
                        exit_price = self._estimate_option_price(
                            previous_bar["close"],
                            current_trade.underlying_price,
                            current_trade.entry_price,
                            current_trade.signal_type,
                        )
                    current_trade = self._close_trade(
                        trade=current_trade,
                        exit_time=prev_bar_time,
                        exit_price=exit_price,
                        exit_reason="end_of_day",
                        config=config,
                    )
                    trades.append(current_trade)
                    current_trade = None
                current_day = bar_day

            # Check exits for current trade
            if current_trade:
                option_price_for_history = None
                if current_trade.intraday_prices:
                    option_price_for_history = self._interpolate_option_price(
                        current_trade.intraday_prices, bar_time
                    )
                current_trade.rsi_history.append((bar_time, current_rsi, option_price_for_history, bar["close"]))

                exit_reason, contracts_to_close, current_price = self._check_exit(
                    current_trade, bar, config, i, current_rsi, current_sma
                )

                # Averaging down
                if exit_reason is None and current_price is not None:
                    num_addons = len(current_trade.add_on_entries)
                    if num_addons < config.max_add_ons:
                        threshold_pct = (num_addons + 1) * config.avg_down_trigger_pct
                        trigger_price = current_trade.entry_price * (1 - threshold_pct)

                        if current_price <= trigger_price:
                            add_price = current_price * (1 + config.slippage_pct)
                            add_contracts = config.contract_multiplier  # Scale add-ons by multiplier
                            total_cost = (current_trade.avg_entry_price * current_trade.total_contracts) + (add_price * add_contracts)
                            current_trade.total_contracts += add_contracts
                            current_trade.remaining_contracts += add_contracts
                            current_trade.avg_entry_price = total_cost / current_trade.total_contracts
                            current_trade.add_on_entries.append((add_price, add_contracts, bar_time))
                            current_trade.stop_loss = current_trade.avg_entry_price * (1 - config.stop_loss_pct)

                if exit_reason == "partial_tp":
                    partial_pnl = (current_price - current_trade.entry_price) * 2 * 100
                    partial_pnl -= config.commission_per_contract * 2
                    current_trade.partial_tp_pnl = partial_pnl
                    current_trade.partial_tp_taken = True
                    current_trade.remaining_contracts = 1

                elif exit_reason in ("stop_loss", "rider_tp", "profit_target"):
                    current_trade = self._close_trade(
                        trade=current_trade,
                        exit_time=bar_time,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        config=config,
                        contracts_to_close=contracts_to_close,
                    )
                    trades.append(current_trade)
                    current_trade = None

            # Check for new entry (only if no position for THIS timeframe)
            if current_trade is None and signal.signal_type != SignalType.NO_SIGNAL:
                option_type = "call" if signal.signal_type == SignalType.BUY_CALL else "put"
                strike = self._get_strike_for_trade(
                    bar["close"], option_type,
                    strike_offset=config.strike_offset,
                    strike_interval=config.strike_interval,
                    underlying_multiplier=config.underlying_multiplier,
                )
                expiration = bar_day
                right = "C" if option_type == "call" else "P"

                intraday_data = []
                if self._theta_available:
                    intraday_data = await self._fetch_intraday_option_data(
                        symbol=self._option_symbol,
                        expiration=expiration,
                        strike=strike,
                        right=right,
                        target_date=bar_day,
                    )

                entry_price = None
                if intraday_data:
                    entry_price = self._interpolate_option_price(intraday_data, bar_time)

                if entry_price is None:
                    entry_price = self._estimate_initial_option_price(
                        signal.signal_type, bar["close"]
                    )

                entry_price *= 1 + config.slippage_pct

                if signal.signal_type == SignalType.BUY_CALL:
                    is_strong = current_rsi <= config.rsi_strong_oversold
                else:
                    is_strong = current_rsi >= config.rsi_strong_overbought
                base_contracts = 3 if is_strong else 1
                contracts = base_contracts * config.contract_multiplier

                current_trade = SimulatedTrade(
                    entry_time=bar_time,
                    entry_price=entry_price,
                    signal_type=signal.signal_type,
                    underlying_price=bar["close"],
                    profit_target=self._calculate_profit_target(entry_price, config),
                    stop_loss=entry_price * (1 - config.stop_loss_pct),
                    entry_bar_index=i,
                    entry_rsi=current_rsi,
                    strike=strike,
                    expiration=expiration,
                    intraday_prices=intraday_data,
                    initial_contracts=contracts,
                    remaining_contracts=contracts,
                    total_contracts=contracts,
                    avg_entry_price=entry_price,
                    timeframe=timeframe_label,
                )

            previous_bar = bar

        # Close any remaining trade
        if current_trade and bars:
            last_bar = bars[-1]
            # Set settlement underlying price (scaled for SPXW)
            current_trade.settlement_underlying = last_bar["close"] * config.underlying_multiplier
            option_price = None
            if current_trade.intraday_prices:
                option_price = self._interpolate_option_price(
                    current_trade.intraday_prices, last_bar["timestamp"]
                )
            if option_price is None:
                option_price = self._estimate_option_price(
                    last_bar["close"],
                    current_trade.underlying_price,
                    current_trade.entry_price,
                    current_trade.signal_type,
                )
            current_trade = self._close_trade(
                trade=current_trade,
                exit_time=last_bar["timestamp"],
                exit_price=option_price,
                exit_reason="end_of_day",
                config=config,
            )
            trades.append(current_trade)

        return trades

    async def _simulate_trades_dual_timeframe(
        self,
        bars_3min: list[dict],
        rsi_values_3min: list[float],
        sma_values_3min: list[float],
        lookup_5min: dict,
        signal_lookup_5min: dict,
        config: BacktestConfig,
        progress_callback: Optional[Callable] = None,
    ) -> list[SimulatedTrade]:
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
                    prev_bar_time = previous_bar["timestamp"]
                    # Set settlement underlying price (scaled for SPXW)
                    current_trade.settlement_underlying = previous_bar["close"] * config.underlying_multiplier
                    exit_price = None
                    if current_trade.intraday_prices:
                        exit_price = self._interpolate_option_price(
                            current_trade.intraday_prices, prev_bar_time
                        )
                    if exit_price is None:
                        exit_price = self._estimate_option_price(
                            previous_bar["close"],
                            current_trade.underlying_price,
                            current_trade.entry_price,
                            current_trade.signal_type,
                        )
                    current_trade = self._close_trade(
                        trade=current_trade,
                        exit_time=prev_bar_time,
                        exit_price=exit_price,
                        exit_reason="end_of_day",
                        config=config,
                    )
                    trades.append(current_trade)
                    current_trade = None
                current_day = bar_day

            # Exit checking (using 3-min bars for granular exit timing)
            if current_trade:
                # Track RSI history
                option_price_for_history = None
                if current_trade.intraday_prices:
                    option_price_for_history = self._interpolate_option_price(
                        current_trade.intraday_prices, bar_time
                    )
                current_trade.rsi_history.append((bar_time, current_rsi_3min, option_price_for_history, bar["close"]))

                exit_reason, contracts_to_close, current_price = self._check_exit(
                    current_trade, bar, config, i, current_rsi_3min, current_sma_3min
                )

                # Averaging down: every -10% drop from ORIGINAL entry price
                if exit_reason is None and current_price is not None:
                    num_addons = len(current_trade.add_on_entries)
                    if num_addons < config.max_add_ons:
                        # Trigger at -10%, -20%, -30% from ORIGINAL entry
                        threshold_pct = (num_addons + 1) * config.avg_down_trigger_pct
                        trigger_price = current_trade.entry_price * (1 - threshold_pct)

                        if current_price <= trigger_price:
                            add_price = current_price * (1 + config.slippage_pct)
                            add_contracts = config.contract_multiplier  # Scale add-ons by multiplier
                            total_cost = (current_trade.avg_entry_price * current_trade.total_contracts) + (add_price * add_contracts)
                            current_trade.total_contracts += add_contracts
                            current_trade.remaining_contracts += add_contracts
                            current_trade.avg_entry_price = total_cost / current_trade.total_contracts
                            current_trade.add_on_entries.append((add_price, add_contracts, bar_time))
                            # Recalculate stop loss from NEW avg entry
                            current_trade.stop_loss = current_trade.avg_entry_price * (1 - config.stop_loss_pct)

                if exit_reason in ("stop_loss", "profit_target"):
                    current_trade = self._close_trade(
                        trade=current_trade,
                        exit_time=bar_time,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        config=config,
                        contracts_to_close=contracts_to_close,
                    )
                    trades.append(current_trade)
                    current_trade = None

            # Entry checking with DUAL-TIMEFRAME CONFIRMATION
            if current_trade is None:
                # Get 5-min signal for this timestamp
                aligned_5min_ts = self._get_aligned_5min_timestamp(bar_time)
                signal_5min = signal_lookup_5min.get(aligned_5min_ts)

                # CONDITION 1: 5-min must have a signal (crossover in extreme zone)
                if signal_5min is None or signal_5min.signal_type == SignalType.NO_SIGNAL:
                    continue

                # CONDITION 2: 3-min RSI must also be in extreme zone (confirmation)
                is_confirmed = False
                if signal_5min.signal_type == SignalType.BUY_CALL:
                    is_confirmed = current_rsi_3min < config.rsi_oversold
                elif signal_5min.signal_type == SignalType.BUY_PUT:
                    is_confirmed = current_rsi_3min > config.rsi_overbought

                if not is_confirmed:
                    continue  # Skip - confirmation failed

                # BOTH CONDITIONS MET - proceed with entry
                signal_type = signal_5min.signal_type
                option_type = "call" if signal_type == SignalType.BUY_CALL else "put"
                strike = self._get_strike_for_trade(
                    bar["close"], option_type,
                    strike_offset=config.strike_offset,
                    strike_interval=config.strike_interval,
                    underlying_multiplier=config.underlying_multiplier,
                )
                expiration = bar_day
                right = "C" if option_type == "call" else "P"

                # Fetch intraday option data
                intraday_data = []
                if self._theta_available:
                    intraday_data = await self._fetch_intraday_option_data(
                        symbol=self._option_symbol,
                        expiration=expiration,
                        strike=strike,
                        right=right,
                        target_date=bar_day,
                    )

                # Get entry price
                entry_price = None
                if intraday_data:
                    entry_price = self._interpolate_option_price(intraday_data, bar_time)
                if entry_price is None:
                    entry_price = self._estimate_initial_option_price(signal_type, bar["close"])
                entry_price *= 1 + config.slippage_pct

                # Determine contract count (strong signal = 3 contracts)
                # Then multiply by contract_multiplier (10 for SPX simulation)
                if signal_type == SignalType.BUY_CALL:
                    is_strong = current_rsi_3min <= config.rsi_strong_oversold
                else:
                    is_strong = current_rsi_3min >= config.rsi_strong_overbought
                base_contracts = 3 if is_strong else 1
                contracts = base_contracts * config.contract_multiplier

                current_trade = SimulatedTrade(
                    entry_time=bar_time,
                    entry_price=entry_price,
                    signal_type=signal_type,
                    underlying_price=bar["close"],
                    profit_target=self._calculate_profit_target(entry_price, config),
                    stop_loss=entry_price * (1 - config.stop_loss_pct),
                    entry_bar_index=i,
                    entry_rsi=current_rsi_3min,
                    strike=strike,
                    expiration=expiration,
                    intraday_prices=intraday_data,
                    initial_contracts=contracts,
                    remaining_contracts=contracts,
                    total_contracts=contracts,
                    avg_entry_price=entry_price,
                )

            previous_bar = bar

        # Close any remaining trade
        if current_trade and bars_3min:
            last_bar = bars_3min[-1]
            # Set settlement underlying price (scaled for SPXW)
            current_trade.settlement_underlying = last_bar["close"] * config.underlying_multiplier
            option_price = None
            if current_trade.intraday_prices:
                option_price = self._interpolate_option_price(
                    current_trade.intraday_prices, last_bar["timestamp"]
                )
            if option_price is None:
                option_price = self._estimate_option_price(
                    last_bar["close"],
                    current_trade.underlying_price,
                    current_trade.entry_price,
                    current_trade.signal_type,
                )
            current_trade = self._close_trade(
                trade=current_trade,
                exit_time=last_bar["timestamp"],
                exit_price=option_price,
                exit_reason="end_of_day",
                config=config,
            )
            trades.append(current_trade)

        return trades

    def _estimate_initial_option_price(
        self, signal_type: SignalType, underlying_price: float
    ) -> float:
        """
        Estimate initial OTM option price (simplified model).

        For 0DTE $20 OTM options, price is lower than ATM due to no intrinsic value.
        This is a rough approximation - real backtest would use historical options data.
        """
        # OTM options are cheaper - roughly 0.3-0.5% of underlying for $20 OTM
        base_pct = 0.004  # 0.4% of underlying (~$2.40 for SPY at $600)
        return underlying_price * base_pct

    def _estimate_option_price(
        self,
        current_underlying: float,
        entry_underlying: float,
        entry_option_price: float,
        signal_type: SignalType,
    ) -> float:
        """
        Estimate option price change based on underlying movement.

        Uses simplified delta approximation for $20 OTM options (~0.3 delta).
        """
        underlying_change = current_underlying - entry_underlying

        # $20 OTM options have roughly 0.3 delta (lower than ATM's 0.5)
        delta = 0.3 if signal_type == SignalType.BUY_CALL else -0.3

        # Option price change = delta * underlying change
        option_change = delta * underlying_change

        # New option price (can't go below 0.01)
        new_price = max(0.01, entry_option_price + option_change)

        return new_price

    def _check_exit(
        self, trade: SimulatedTrade, bar: dict, config: BacktestConfig, current_bar_index: int, current_rsi: float, current_sma: float
    ) -> tuple[Optional[str], int, float]:
        """
        Check if exit conditions are met.

        Returns:
            (exit_reason, contracts_to_close, current_price)
            - "partial_tp": Close 2 contracts, keep 1 rider (for 3-contract trades)
            - "rider_tp": Close rider at +$1.00 target
            - "profit_target": Normal TP at +$0.50
            - "stop_loss": 30% stop loss on remaining contracts
            - None: No exit
        """
        bar_time = bar["timestamp"]

        # Try to get real interpolated price from ThetaData
        current_price = None
        if trade.intraday_prices:
            current_price = self._interpolate_option_price(trade.intraday_prices, bar_time)

        # Fallback to estimated price
        if current_price is None:
            current_price = self._estimate_option_price(
                bar["close"],
                trade.underlying_price,
                trade.entry_price,
                trade.signal_type,
            )

        # Use avg_entry_price for P&L calc (handles averaging down)
        avg_price = trade.avg_entry_price if trade.avg_entry_price > 0 else trade.entry_price
        profit_per_contract = current_price - avg_price

        # 1. Stop loss: X% below avg entry price - exits ALL remaining contracts
        stop_loss_price = avg_price * (1 - config.stop_loss_pct)
        if current_price <= stop_loss_price:
            return ("stop_loss", trade.total_contracts, current_price)

        # 2. Profit target: exits ALL contracts
        # Use percentage target if set, otherwise use fixed dollar target
        if config.profit_target_pct is not None:
            profit_target_price = avg_price * (1 + config.profit_target_pct)
            if current_price >= profit_target_price:
                return ("profit_target", trade.total_contracts, current_price)
        else:
            if profit_per_contract >= config.profit_target_dollars:
                return ("profit_target", trade.total_contracts, current_price)

        return (None, 0, current_price)

    def _calculate_settlement_value(
        self,
        trade: SimulatedTrade,
    ) -> float:
        """
        Calculate option settlement value at expiration.

        For calls: max(0, settlement_price - strike)
        For puts: max(0, strike - settlement_price)

        Uses trade.settlement_underlying (final underlying price at EOD).
        Works for both physical (SPY) and cash (SPXW) settlement.

        Returns:
            Intrinsic value per share (multiply by 100 for contract value)
        """
        settlement_price = trade.settlement_underlying
        if settlement_price <= 0:
            # Fallback to entry underlying if settlement not set
            settlement_price = trade.underlying_price

        if trade.signal_type == SignalType.BUY_CALL:
            intrinsic = settlement_price - trade.strike
        else:  # PUT
            intrinsic = trade.strike - settlement_price

        return max(0, intrinsic)

    def _close_trade(
        self,
        trade: SimulatedTrade,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        config: BacktestConfig,
        contracts_to_close: int = None,
    ) -> SimulatedTrade:
        """Close a trade and calculate P&L."""
        # For EOD exits, use settlement value (intrinsic) instead of last traded price
        if exit_reason == "end_of_day":
            settlement_value = self._calculate_settlement_value(trade)
            exit_price = settlement_value
            # No slippage on settlement - it's automatic
        else:
            # Apply slippage on normal exits
            exit_price *= 1 - config.slippage_pct

        if contracts_to_close is None:
            contracts_to_close = trade.total_contracts  # Use total_contracts for averaging down

        # Use avg_entry_price for P&L calc (handles averaging down)
        avg_price = trade.avg_entry_price if trade.avg_entry_price > 0 else trade.entry_price

        # Calculate P&L for contracts being closed (options are 100 shares per contract)
        gross_pnl = (exit_price - avg_price) * contracts_to_close * 100

        # Commission: entry for all entries (initial + add-ons), exit for all contracts
        total_entries = 1 + len(trade.add_on_entries)  # Initial entry + all add-on entries
        entry_commission = config.commission_per_contract * total_entries
        exit_commission = config.commission_per_contract * contracts_to_close
        net_pnl = gross_pnl - entry_commission - exit_commission

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.pnl = net_pnl
        trade.remaining_contracts = 0

        return trade

    def _calculate_metrics(
        self, trades: list[SimulatedTrade], config: BacktestConfig
    ) -> BacktestMetrics:
        """Calculate performance metrics."""
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
                total_contracts=0,
                winning_contracts=0,
                losing_contracts=0,
                contract_win_rate=0.0,
            )

        pnls = [t.pnl for t in trades if t.pnl is not None]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls) if pnls else 0

        # Win rate (trade-level)
        win_rate = len(winning) / len(pnls) if pnls else 0

        # Contract-level metrics
        # Count each entry (initial + add-ons) as independent
        total_contracts = 0
        winning_contracts = 0
        losing_contracts = 0

        for trade in trades:
            if trade.pnl is None or trade.exit_price is None:
                continue

            # Initial entry
            total_contracts += 1
            if trade.exit_price >= trade.entry_price:
                winning_contracts += 1
            else:
                losing_contracts += 1

            # Each add-on entry
            for addon_price, addon_contracts, addon_time in trade.add_on_entries:
                total_contracts += 1
                if trade.exit_price >= addon_price:
                    winning_contracts += 1
                else:
                    losing_contracts += 1

        contract_win_rate = winning_contracts / total_contracts if total_contracts > 0 else 0.0

        # Max drawdown
        equity = config.initial_capital
        peak = equity
        max_dd = 0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Profit factor
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (using daily returns for proper annualization)
        sharpe = self._calculate_sharpe_ratio_daily(trades, config)

        return BacktestMetrics(
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            total_contracts=total_contracts,
            winning_contracts=winning_contracts,
            losing_contracts=losing_contracts,
            contract_win_rate=contract_win_rate,
        )

    def _calculate_sharpe_ratio_daily(
        self, trades: list[SimulatedTrade], config: BacktestConfig
    ) -> float:
        """
        Calculate annualized Sharpe ratio using daily returns.

        Aggregates trade P&Ls by day to avoid frequency inflation.
        Uses sample standard deviation (n-1 denominator) for statistical correctness.
        """
        if not trades:
            return 0.0

        # Aggregate P&Ls by day
        from collections import defaultdict
        daily_pnls: dict[date, float] = defaultdict(float)
        for trade in trades:
            if trade.exit_time and trade.pnl is not None:
                day = trade.exit_time.date()
                daily_pnls[day] += trade.pnl

        if len(daily_pnls) <= 1:
            return 0.0

        # Convert to daily returns (relative to capital)
        daily_returns = [pnl / config.initial_capital for pnl in daily_pnls.values()]

        n = len(daily_returns)
        avg_return = sum(daily_returns) / n

        # Sample standard deviation (n-1 denominator, Bessel's correction)
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / (n - 1)
        std_return = math.sqrt(variance)

        if std_return <= 0:
            return 0.0

        # Annualize with sqrt(252) - correct for daily returns
        return (avg_return / std_return) * math.sqrt(252)

    def _build_equity_curve(
        self, trades: list[SimulatedTrade], initial_capital: float
    ) -> list[tuple[datetime, float]]:
        """Build equity curve from trades."""
        curve = []
        equity = initial_capital

        for trade in trades:
            if trade.exit_time and trade.pnl is not None:
                equity += trade.pnl
                curve.append((trade.exit_time, equity))

        return curve

    def _to_backtest_trade(self, trade: SimulatedTrade) -> BacktestTrade:
        """Convert SimulatedTrade to BacktestTrade schema."""
        pnl_pct = (
            (trade.exit_price - trade.entry_price) / trade.entry_price
            if trade.entry_price > 0
            else 0
        )

        return BacktestTrade(
            entry_date=trade.entry_time,
            exit_date=trade.exit_time or trade.entry_time,
            signal_type=trade.signal_type,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price or trade.entry_price,
            pnl_dollars=trade.pnl or 0,
            pnl_percent=pnl_pct,
            exit_reason=trade.exit_reason or "unknown",
            entry_rsi=trade.entry_rsi,
            rsi_history=trade.rsi_history,
            timeframe=trade.timeframe,
            strike=trade.strike,
            settlement_underlying=trade.settlement_underlying,
        )

    def _empty_result(self, config: BacktestConfig) -> BacktestResult:
        """Return empty result when no data available."""
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
