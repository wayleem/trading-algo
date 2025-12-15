"""Backtest service orchestrator.

Main entry point that coordinates data fetching, simulation, and metrics.
"""

import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from app.services.alpaca_client import AlpacaClient
from app.services.theta_data import ThetaDataClient
from app.core.config import settings
from app.models.schemas import BacktestResult, SignalType
from app.services.indicators import (
    calculate_rsi_series,
    calculate_sma_series,
    calculate_macd_series,
    calculate_bollinger_bands_series,
    calculate_band_width_series,
)
from app.services.signal_generator import SignalGenerator, EnhancedSignalGenerator
from app.services.candlestick_patterns import CandlestickAnalyzer

from .config import BacktestConfig
from .exit_strategy import ExitStrategy, StandardExitStrategy, PartialExitStrategy, BollingerBandExitStrategy, get_exit_strategy
from .metrics import MetricsCalculator
from .models import SimulatedTrade
from .option_pricing import OptionPricingService
from .position_manager import PositionManager
from .simulator import TradeSimulator

logger = logging.getLogger(__name__)


class BacktestService:
    """
    Historical backtesting engine.

    Simulates RSI strategy on historical data with:
    - Real option pricing via ThetaData (or fallback to simplified model)
    - Slippage and commission
    - Intraday exit logic
    - Multiple timeframe modes (single, dual, parallel)
    """

    def __init__(
        self,
        alpaca_client: Optional[AlpacaClient] = None,
        theta_client: Optional[ThetaDataClient] = None,
    ):
        """
        Initialize backtest service.

        Args:
            alpaca_client: Client for fetching historical bars
            theta_client: Client for option data (optional)
        """
        self.client = alpaca_client or AlpacaClient()
        self.theta_client = theta_client
        self._theta_available = False
        self._option_symbol = "SPXW"

    async def run_backtest(
        self,
        config: BacktestConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        exit_strategy: Optional[ExitStrategy] = None,
    ) -> BacktestResult:
        """
        Run historical backtest.

        Args:
            config: Backtest configuration
            progress_callback: Optional callback for progress updates (current, total)
            exit_strategy: Optional custom exit strategy (default: StandardExitStrategy)

        Returns:
            BacktestResult with trades, metrics, and equity curve
        """
        # Determine effective option symbol
        option_symbol = config.option_symbol if config.option_symbol else config.symbol
        self._option_symbol = option_symbol

        logger.info(
            f"Starting backtest: {config.symbol} from {config.start_date} to {config.end_date}"
        )
        if option_symbol != config.symbol:
            logger.info(f"Using {option_symbol} options with {config.symbol} underlying")

        # Initialize ThetaData client if not provided
        if self.theta_client is None:
            self.theta_client = ThetaDataClient()

        # Check if ThetaData is available (REQUIRED - no fallback)
        self._theta_available = await self._check_theta_availability()
        if self._theta_available:
            logger.info("ThetaData available - using real historical option prices")
        else:
            raise RuntimeError(
                "ThetaData is REQUIRED for accurate backtesting. "
                "Please ensure ThetaTerminal is running on port 25503. "
                "Start with: java -jar ThetaTerminalv3.jar"
            )

        # Initialize components
        option_pricing = OptionPricingService(theta_available=self._theta_available)
        position_manager = PositionManager(config)
        metrics_calculator = MetricsCalculator(config)

        # Use provided exit strategy or default based on config
        if exit_strategy is None:
            if config.bb_exit_strategy != "none":
                exit_strategy = BollingerBandExitStrategy(
                    bb_exit_strategy=config.bb_exit_strategy,
                    stop_loss_pct=config.stop_loss_pct,
                    profit_target_dollars=config.profit_target_dollars,
                )
            elif config.enable_partial_exits:
                exit_strategy = PartialExitStrategy(option_pricing)
            else:
                exit_strategy = StandardExitStrategy(option_pricing)

        simulator = TradeSimulator(
            config=config,
            option_pricing=option_pricing,
            position_manager=position_manager,
            exit_strategy=exit_strategy,
            option_symbol=option_symbol,
        )

        # Fetch historical data
        bars_data = await self._fetch_historical_bars(config)

        # Run appropriate simulation mode
        if config.parallel_mode or config.dual_timeframe_enabled:
            # Note: dual_timeframe_enabled and parallel_mode both run 3-min + 5-min independently
            # There is no "confirmation" mode - timeframes always run in parallel
            trades = await self._run_parallel_mode(
                bars_data, config, simulator, progress_callback
            )
        else:
            trades = await self._run_single_timeframe_mode(
                bars_data, config, simulator, progress_callback
            )

        # Build and return result
        result = metrics_calculator.build_result(trades)

        logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"Win rate: {result.metrics.win_rate:.1%}, Total P&L: ${result.metrics.total_pnl:.2f}"
        )

        return result

    async def _run_parallel_mode(
        self,
        bars_data: tuple,
        config: BacktestConfig,
        simulator: TradeSimulator,
        progress_callback: Optional[Callable],
    ) -> List[SimulatedTrade]:
        """Run 3-min and 5-min strategies independently in parallel."""
        bars_5min, bars_3min, pattern_bars = bars_data

        if not bars_5min or not bars_3min:
            logger.warning("No historical data available")
            return []

        logger.info("PARALLEL MODE: Running 3-min and 5-min strategies independently")
        logger.info(f"Fetched {len(bars_5min)} 5-min bars and {len(bars_3min)} 3-min bars")
        logger.info(f"Signal mode: {config.signal_mode}, RSI confirmation: {config.rsi_confirmation_mode}")

        # Setup candlestick pattern lookup for POSITION SIZING (not signal filtering)
        pattern_lookup = None
        pattern_interval = 15  # Default
        if config.use_pattern_position_sizing and pattern_bars:
            candlestick_analyzer = CandlestickAnalyzer(
                body_threshold=config.pattern_body_threshold,
                shadow_ratio=config.pattern_shadow_ratio,
            )
            logger.info(f"Building pattern lookup from {len(pattern_bars)} {config.candlestick_timeframe} bars for position sizing")
            pattern_lookup = self._build_pattern_lookup(pattern_bars, candlestick_analyzer)
            pattern_interval = self._get_pattern_interval_minutes(config.candlestick_timeframe)

        # Process 3-min strategy
        closes_3min = [bar["close"] for bar in bars_3min]
        timestamps_3min = [bar["timestamp"] for bar in bars_3min]
        rsi_values_3min = calculate_rsi_series(closes_3min, config.rsi_period)
        sma_values_3min = calculate_sma_series(rsi_values_3min, config.rsi_sma_period)

        # Process 5-min strategy
        closes_5min = [bar["close"] for bar in bars_5min]
        timestamps_5min = [bar["timestamp"] for bar in bars_5min]
        rsi_values_5min = calculate_rsi_series(closes_5min, config.rsi_period)
        sma_values_5min = calculate_sma_series(rsi_values_5min, config.rsi_sma_period)

        # Calculate MACD and BB if needed
        macd_histogram_3min = None
        macd_histogram_5min = None
        bb_upper_3min = bb_middle_3min = bb_lower_3min = bb_width_3min = None
        bb_upper_5min = bb_middle_5min = bb_lower_5min = bb_width_5min = None

        if config.signal_mode in ("macd_filter", "independent", "hybrid"):
            logger.info(f"Calculating MACD (fast={config.macd_fast_period}, slow={config.macd_slow_period}, signal={config.macd_signal_period})")
            _, _, macd_histogram_3min = calculate_macd_series(
                closes_3min, config.macd_fast_period, config.macd_slow_period, config.macd_signal_period
            )
            _, _, macd_histogram_5min = calculate_macd_series(
                closes_5min, config.macd_fast_period, config.macd_slow_period, config.macd_signal_period
            )

        if config.bb_entry_strategy != "none" or config.bb_exit_strategy != "none" or config.bb_volatility_filter:
            logger.info(f"Calculating Bollinger Bands (period={config.bb_period}, std={config.bb_num_std})")
            bb_upper_3min, bb_middle_3min, bb_lower_3min = calculate_bollinger_bands_series(
                closes_3min, config.bb_period, config.bb_num_std
            )
            bb_width_3min = calculate_band_width_series(bb_upper_3min, bb_lower_3min, bb_middle_3min)
            bb_upper_5min, bb_middle_5min, bb_lower_5min = calculate_bollinger_bands_series(
                closes_5min, config.bb_period, config.bb_num_std
            )
            bb_width_5min = calculate_band_width_series(bb_upper_5min, bb_lower_5min, bb_middle_5min)

        # Choose signal generator based on mode
        use_enhanced = (
            config.signal_mode != "rsi_only" or
            config.rsi_confirmation_mode != "none"
        )

        if use_enhanced:
            signal_generator = EnhancedSignalGenerator(
                rsi_oversold=config.rsi_oversold,
                rsi_overbought=config.rsi_overbought,
                signal_mode=config.signal_mode,
                rsi_confirmation_mode=config.rsi_confirmation_mode,
                rsi_confirm_buffer=config.rsi_confirm_buffer,
                macd_filter_calls=config.macd_filter_calls,
                macd_filter_puts=config.macd_filter_puts,
                macd_signal_threshold=config.macd_signal_threshold,
                bb_entry_strategy=config.bb_entry_strategy,
                bb_volatility_filter=config.bb_volatility_filter,
                bb_width_threshold=config.bb_width_threshold,
            )

            # For dual RSI confirmation, use 5-min as primary and 3-min as secondary
            if config.rsi_confirmation_mode != "none":
                logger.info(f"Using dual RSI confirmation mode: {config.rsi_confirmation_mode}")
                # Generate unified signals using cross-timeframe confirmation
                signals_5min = signal_generator.evaluate_series(
                    rsi_values=rsi_values_5min,
                    sma_values=sma_values_5min,
                    close_prices=closes_5min,
                    timestamps=timestamps_5min,
                    macd_histogram=macd_histogram_5min,
                    bb_upper=bb_upper_5min,
                    bb_lower=bb_lower_5min,
                    bb_middle=bb_middle_5min,
                    bb_width=bb_width_5min,
                    rsi_values_secondary=rsi_values_3min,
                    sma_values_secondary=sma_values_3min,
                    timestamps_secondary=timestamps_3min,
                )
                # For 3-min, use 5-min as secondary
                signals_3min = signal_generator.evaluate_series(
                    rsi_values=rsi_values_3min,
                    sma_values=sma_values_3min,
                    close_prices=closes_3min,
                    timestamps=timestamps_3min,
                    macd_histogram=macd_histogram_3min,
                    bb_upper=bb_upper_3min,
                    bb_lower=bb_lower_3min,
                    bb_middle=bb_middle_3min,
                    bb_width=bb_width_3min,
                    rsi_values_secondary=rsi_values_5min,
                    sma_values_secondary=sma_values_5min,
                    timestamps_secondary=timestamps_5min,
                )
            else:
                # No cross-timeframe confirmation, generate signals independently
                signals_3min = signal_generator.evaluate_series(
                    rsi_values=rsi_values_3min,
                    sma_values=sma_values_3min,
                    close_prices=closes_3min,
                    timestamps=timestamps_3min,
                    macd_histogram=macd_histogram_3min,
                    bb_upper=bb_upper_3min,
                    bb_lower=bb_lower_3min,
                    bb_middle=bb_middle_3min,
                    bb_width=bb_width_3min,
                )
                signals_5min = signal_generator.evaluate_series(
                    rsi_values=rsi_values_5min,
                    sma_values=sma_values_5min,
                    close_prices=closes_5min,
                    timestamps=timestamps_5min,
                    macd_histogram=macd_histogram_5min,
                    bb_upper=bb_upper_5min,
                    bb_lower=bb_lower_5min,
                    bb_middle=bb_middle_5min,
                    bb_width=bb_width_5min,
                )
        else:
            # Original RSI-only generator
            signal_generator = SignalGenerator(
                rsi_oversold=config.rsi_oversold,
                rsi_overbought=config.rsi_overbought,
            )
            signals_3min = signal_generator.evaluate_series(
                rsi_values=rsi_values_3min,
                sma_values=sma_values_3min,
                close_prices=closes_3min,
                timestamps=timestamps_3min,
                pattern_confirmations=None,
            )
            signals_5min = signal_generator.evaluate_series(
                rsi_values=rsi_values_5min,
                sma_values=sma_values_5min,
                close_prices=closes_5min,
                timestamps=timestamps_5min,
                pattern_confirmations=None,
            )

        # Build pattern confirmations for position sizing (passed to simulator)
        pattern_confirmations_3min = None
        pattern_confirmations_5min = None
        if pattern_lookup:
            pattern_confirmations_3min = [
                self._get_pattern_for_timestamp(ts, pattern_lookup, pattern_interval)
                for ts in timestamps_3min
            ]
            pattern_confirmations_5min = [
                self._get_pattern_for_timestamp(ts, pattern_lookup, pattern_interval)
                for ts in timestamps_5min
            ]

        # Simulate trades for both timeframes
        return await simulator.simulate_trades_parallel(
            bars_3min=bars_3min,
            signals_3min=signals_3min,
            rsi_values_3min=rsi_values_3min,
            sma_values_3min=sma_values_3min,
            bars_5min=bars_5min,
            signals_5min=signals_5min,
            rsi_values_5min=rsi_values_5min,
            sma_values_5min=sma_values_5min,
            progress_callback=progress_callback,
            pattern_bars=pattern_bars,
            pattern_confirmations_3min=pattern_confirmations_3min,
            pattern_confirmations_5min=pattern_confirmations_5min,
            bb_upper_3min=bb_upper_3min,
            bb_middle_3min=bb_middle_3min,
            bb_lower_3min=bb_lower_3min,
            bb_upper_5min=bb_upper_5min,
            bb_middle_5min=bb_middle_5min,
            bb_lower_5min=bb_lower_5min,
        )

    async def _run_dual_timeframe_mode(
        self,
        bars_data: tuple,
        config: BacktestConfig,
        simulator: TradeSimulator,
        progress_callback: Optional[Callable],
    ) -> List[SimulatedTrade]:
        """Run dual-timeframe mode: 5-min trigger + 3-min confirmation."""
        bars_5min, bars_3min, pattern_bars = bars_data

        if not bars_5min or not bars_3min:
            logger.warning("No historical data available")
            return []

        logger.info(f"Fetched {len(bars_5min)} 5-min bars and {len(bars_3min)} 3-min bars")

        # Setup candlestick analyzer and pattern lookup if enabled
        pattern_lookup = None
        pattern_interval = 15  # Default
        if config.use_candlestick_filter:
            candlestick_analyzer = CandlestickAnalyzer(
                body_threshold=config.pattern_body_threshold,
                shadow_ratio=config.pattern_shadow_ratio,
            )
            if pattern_bars:
                logger.info(f"Building pattern lookup from {len(pattern_bars)} {config.candlestick_timeframe} bars")
                pattern_lookup = self._build_pattern_lookup(pattern_bars, candlestick_analyzer)
                pattern_interval = self._get_pattern_interval_minutes(config.candlestick_timeframe)

        # Calculate indicators for 5-min (PRIMARY - trigger)
        closes_5min = [bar["close"] for bar in bars_5min]
        timestamps_5min = [bar["timestamp"] for bar in bars_5min]
        rsi_values_5min = calculate_rsi_series(closes_5min, config.rsi_period)
        sma_values_5min = calculate_sma_series(rsi_values_5min, config.rsi_sma_period)

        # Build 5-min lookup for fast access
        lookup_5min = self._build_5min_lookup(bars_5min, rsi_values_5min, sma_values_5min)

        # Generate pattern confirmations for 5-min bars using pattern lookup
        pattern_confirmations_5min = None
        if pattern_lookup:
            pattern_confirmations_5min = [
                self._get_pattern_for_timestamp(ts, pattern_lookup, pattern_interval)
                for ts in timestamps_5min
            ]

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
            pattern_confirmations=pattern_confirmations_5min,
        )

        # Build 5-min signal lookup
        signal_lookup_5min = {}
        for sig in signals_5min:
            ts = sig.timestamp
            if hasattr(ts, 'replace'):
                ts = ts.replace(second=0, microsecond=0)
            signal_lookup_5min[ts] = sig.signal_type

        # Calculate indicators for 3-min (CONFIRMATION)
        closes_3min = [bar["close"] for bar in bars_3min]
        rsi_values_3min = calculate_rsi_series(closes_3min, config.rsi_period)
        sma_values_3min = calculate_sma_series(rsi_values_3min, config.rsi_sma_period)

        # Simulate trades using dual-timeframe logic
        return await simulator.simulate_trades_dual_timeframe(
            bars_3min=bars_3min,
            rsi_values_3min=rsi_values_3min,
            sma_values_3min=sma_values_3min,
            lookup_5min=lookup_5min,
            signal_lookup_5min=signal_lookup_5min,
            progress_callback=progress_callback,
            pattern_bars=pattern_bars,
        )

    async def _run_single_timeframe_mode(
        self,
        bars_data: tuple,
        config: BacktestConfig,
        simulator: TradeSimulator,
        progress_callback: Optional[Callable],
    ) -> List[SimulatedTrade]:
        """Run single timeframe mode."""
        bars = bars_data[0]
        pattern_bars = bars_data[2] if len(bars_data) > 2 else None

        if not bars:
            logger.warning("No historical data available")
            return []

        logger.info(f"Fetched {len(bars)} bars")

        # Extract price data
        closes = [bar["close"] for bar in bars]
        timestamps = [bar["timestamp"] for bar in bars]

        # Calculate indicators
        rsi_values = calculate_rsi_series(closes, config.rsi_period)
        sma_values = calculate_sma_series(rsi_values, config.rsi_sma_period)

        # Generate signals (NO pattern filtering - RSI crossover only)
        signal_generator = SignalGenerator(
            rsi_oversold=config.rsi_oversold,
            rsi_overbought=config.rsi_overbought,
        )
        signals = signal_generator.evaluate_series(
            rsi_values=rsi_values,
            sma_values=sma_values,
            close_prices=closes,
            timestamps=timestamps,
            pattern_confirmations=None,  # Patterns don't filter signals
        )

        # Build pattern confirmations for POSITION SIZING only (if enabled)
        pattern_confirmations = None
        if config.use_pattern_position_sizing:
            candlestick_analyzer = CandlestickAnalyzer(
                body_threshold=config.pattern_body_threshold,
                shadow_ratio=config.pattern_shadow_ratio,
            )
            if pattern_bars:
                # Use separate timeframe bars for pattern analysis
                logger.info(f"Building pattern lookup from {len(pattern_bars)} {config.candlestick_timeframe} bars for position sizing")
                pattern_lookup = self._build_pattern_lookup(pattern_bars, candlestick_analyzer)
                pattern_interval = self._get_pattern_interval_minutes(config.candlestick_timeframe)
                pattern_confirmations = [
                    self._get_pattern_for_timestamp(ts, pattern_lookup, pattern_interval)
                    for ts in timestamps
                ]
            else:
                # Fallback to same-timeframe pattern analysis
                pattern_confirmations = []
                for i in range(len(bars)):
                    # Get recent bars for pattern analysis (need up to 3 bars for multi-candle patterns)
                    recent_bars = bars[max(0, i - 2) : i + 1]
                    # Get both confirmations with strength
                    is_bullish, bullish_strength, bullish_name = candlestick_analyzer.get_confirmation_with_strength(
                        recent_bars, "call"
                    )
                    is_bearish, bearish_strength, bearish_name = candlestick_analyzer.get_confirmation_with_strength(
                        recent_bars, "put"
                    )
                    # Use the stronger pattern's info
                    if bullish_strength >= bearish_strength:
                        strength = bullish_strength
                        pattern_name = bullish_name
                    else:
                        strength = bearish_strength
                        pattern_name = bearish_name
                    pattern_confirmations.append((is_bullish, is_bearish, strength, pattern_name))

        # Simulate trades (pattern_confirmations used for position sizing only)
        return await simulator.simulate_trades(
            bars=bars,
            signals=signals,
            rsi_values=rsi_values,
            sma_values=sma_values,
            progress_callback=progress_callback,
            pattern_bars=pattern_bars,
            pattern_confirmations=pattern_confirmations,
        )

    async def _fetch_historical_bars(
        self, config: BacktestConfig
    ) -> Tuple[List[dict], Optional[List[dict]], Optional[List[dict]]]:
        """
        Fetch historical bars for backtesting.

        Returns:
            Tuple of (primary_bars, confirmation_bars, pattern_bars)
            - If dual_timeframe_enabled: (5-min bars, 3-min bars, pattern_bars or None)
            - If disabled: (primary bars, None, pattern_bars or None)
            - pattern_bars is only fetched when use_candlestick_filter is enabled
        """
        import asyncio

        start_dt = datetime.combine(config.start_date, datetime.min.time())
        end_dt = datetime.combine(config.end_date, datetime.max.time())

        async def fetch_bars_for_timeframe(timeframe: str) -> List[dict]:
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

        # Build list of timeframes to fetch
        if config.parallel_mode or config.dual_timeframe_enabled:
            fetches = [
                fetch_bars_for_timeframe("5Min"),  # 5-min
                fetch_bars_for_timeframe("3Min"),  # 3-min
            ]
            # Add pattern timeframe fetch if pattern position sizing is enabled
            if config.use_pattern_position_sizing:
                fetches.append(fetch_bars_for_timeframe(config.candlestick_timeframe))
                logger.info(f"Fetching {config.candlestick_timeframe} bars for pattern-based position sizing")

            results = await asyncio.gather(*fetches)

            primary_bars = results[0]
            confirmation_bars = results[1]
            pattern_bars = results[2] if config.use_pattern_position_sizing else None
            return (primary_bars, confirmation_bars, pattern_bars)
        else:
            # Single timeframe mode
            fetches = [fetch_bars_for_timeframe(config.primary_timeframe)]

            # Add pattern timeframe fetch if pattern position sizing is enabled
            if config.use_pattern_position_sizing:
                fetches.append(fetch_bars_for_timeframe(config.candlestick_timeframe))
                logger.info(f"Fetching {config.candlestick_timeframe} bars for pattern-based position sizing")

            results = await asyncio.gather(*fetches)

            primary_bars = results[0]
            pattern_bars = results[1] if config.use_pattern_position_sizing else None
            return (primary_bars, None, pattern_bars)

    async def _check_theta_availability(self) -> bool:
        """Check if ThetaData (ThetaTerminal) is available."""
        try:
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

    def _build_5min_lookup(
        self,
        bars: List[dict],
        rsi_values: List[float],
        sma_values: List[float],
    ) -> Dict[datetime, dict]:
        """Build lookup dict for 5-min bars indexed by timestamp."""
        lookup = {}
        for bar, rsi, sma in zip(bars, rsi_values, sma_values):
            ts = bar["timestamp"]
            if hasattr(ts, 'replace'):
                ts = ts.replace(second=0, microsecond=0)
            lookup[ts] = {
                "bar": bar,
                "rsi": rsi,
                "sma": sma,
            }
        return lookup

    def _build_pattern_lookup(
        self,
        bars: List[dict],
        candlestick_analyzer: CandlestickAnalyzer,
    ) -> Dict[datetime, Tuple[bool, bool, float, Optional[str]]]:
        """
        Build lookup dict for candlestick patterns indexed by timestamp.

        Args:
            bars: List of OHLC bars (e.g., 15-min bars)
            candlestick_analyzer: CandlestickAnalyzer instance

        Returns:
            Dict mapping timestamp → (is_bullish, is_bearish, strength, pattern_name)
            - is_bullish: True if bullish pattern detected
            - is_bearish: True if bearish pattern detected
            - strength: Pattern strength 0.0-1.0 (max of bullish/bearish strength)
            - pattern_name: Name of the detected pattern or None
        """
        lookup = {}
        for i, bar in enumerate(bars):
            # Get recent bars for pattern analysis (need up to 3 bars for multi-candle patterns)
            recent_bars = bars[max(0, i - 2) : i + 1]

            # Get bullish and bearish confirmation with strength
            is_bullish, bullish_strength, bullish_name = candlestick_analyzer.get_confirmation_with_strength(
                recent_bars, "call"
            )
            is_bearish, bearish_strength, bearish_name = candlestick_analyzer.get_confirmation_with_strength(
                recent_bars, "put"
            )

            # Use the stronger pattern's info
            if bullish_strength >= bearish_strength:
                strength = bullish_strength
                pattern_name = bullish_name
            else:
                strength = bearish_strength
                pattern_name = bearish_name

            ts = bar["timestamp"]
            if hasattr(ts, 'replace'):
                ts = ts.replace(second=0, microsecond=0)
            lookup[ts] = (is_bullish, is_bearish, strength, pattern_name)

        return lookup

    def _get_pattern_for_timestamp(
        self,
        timestamp: datetime,
        pattern_lookup: Dict[datetime, Tuple[bool, bool, float, Optional[str]]],
        pattern_interval_minutes: int = 15,
    ) -> Optional[Tuple[bool, bool, float, Optional[str]]]:
        """
        Get pattern confirmation for a given timestamp by aligning to pattern timeframe.

        Args:
            timestamp: The timestamp from RSI bars (e.g., 3-min or 5-min)
            pattern_lookup: Dict of pattern confirmations indexed by pattern bar timestamp
            pattern_interval_minutes: The interval of the pattern bars (default 15 for 15Min)

        Returns:
            Tuple of (is_bullish, is_bearish, strength, pattern_name) or None if not found
        """
        if not pattern_lookup:
            return None

        # Align to pattern timeframe boundary (floor)
        aligned = timestamp.replace(
            minute=(timestamp.minute // pattern_interval_minutes) * pattern_interval_minutes,
            second=0,
            microsecond=0
        )

        return pattern_lookup.get(aligned, None)

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
