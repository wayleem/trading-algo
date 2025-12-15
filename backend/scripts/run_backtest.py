#!/usr/bin/env python3
"""
CLI script to run historical backtest.

Usage:
    python run_backtest.py --start 2024-01-01 --end 2024-12-01
    python run_backtest.py --start 2024-01-01 --end 2024-12-01 --profit-target 0.30
"""

import argparse
import asyncio
import subprocess
import sys
import time
import httpx
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.backtest import BacktestService, BacktestConfig
from app.services.alpaca_client import AlpacaClient


def start_theta_terminal():
    """Start ThetaTerminal if not already running."""
    theta_jar = Path(__file__).parent.parent / "ThetaTerminalv3.jar"

    if not theta_jar.exists():
        print(f"Warning: ThetaTerminal not found at {theta_jar}")
        return None

    # Check if already running
    try:
        response = httpx.get("http://localhost:25503/v3/option/list/expirations?symbol=SPY&format=json", timeout=2.0)
        if response.status_code == 200:
            print("ThetaTerminal already running")
            return None
    except:
        pass

    # Start ThetaTerminal
    print("Starting ThetaTerminal...")
    process = subprocess.Popen(
        ["java", "-jar", str(theta_jar)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for it to be ready
    for i in range(30):  # Wait up to 30 seconds
        time.sleep(1)
        try:
            response = httpx.get("http://localhost:25503/v3/option/list/expirations?symbol=SPY&format=json", timeout=2.0)
            if response.status_code == 200:
                print("ThetaTerminal started successfully")
                return process
        except:
            pass
        print(f"Waiting for ThetaTerminal... ({i+1}/30)")

    print("Warning: ThetaTerminal did not start in time")
    return process


def check_symbol_has_options(symbol: str) -> bool:
    """Check if ThetaData has option data for the given symbol."""
    try:
        response = httpx.get(
            f"http://localhost:25503/v3/option/list/expirations?symbol={symbol}&format=json",
            timeout=5.0
        )
        if response.status_code == 200:
            data = response.json()
            return "response" in data and len(data.get("response", [])) > 0
        return False
    except:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Run RSI strategy backtest")

    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Symbol to backtest (default: SPY, use SPX for S&P 500 index options)",
    )
    parser.add_argument(
        "--rsi-period",
        type=int,
        default=20,
        help="RSI period (default: 20, optimized from 14)",
    )
    parser.add_argument(
        "--sma-period",
        type=int,
        default=12,
        help="RSI SMA period (default: 12, optimized from 14)",
    )
    parser.add_argument(
        "--rsi-oversold",
        type=float,
        default=44.0,
        help="RSI oversold threshold for CALL entries (default: 44, optimized from 30)",
    )
    parser.add_argument(
        "--rsi-overbought",
        type=float,
        default=60.0,
        help="RSI overbought threshold for PUT entries (default: 60, optimized from 70)",
    )
    parser.add_argument(
        "--profit-target",
        type=float,
        default=0.50,
        help="Profit target in dollars per contract (default: 0.50)",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.25,
        help="Stop loss percentage as decimal (default: 0.25 for 25%%, optimized from 0.40)",
    )
    parser.add_argument(
        "--max-hold-bars",
        type=int,
        default=3,
        help="Maximum bars to hold position (default: 3)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="dual",
        choices=["3min", "5min", "dual", "parallel"],
        help="Timeframe: 3min, 5min, dual (confirmation), or parallel (independent) (default: dual)",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=1,
        help="Contract multiplier (default: 1, use 10 for SPX-like exposure with SPY)",
    )
    parser.add_argument(
        "--strike-offset",
        type=float,
        default=0.5,
        help="Strike offset from ATM (default: 0.5, optimized from 2). Positive=OTM, Negative=ITM",
    )
    parser.add_argument(
        "--profit-target-pct",
        type=float,
        default=None,
        help="Percentage-based profit target (e.g., 0.15 for 15%%). Overrides --profit-target if set.",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0,
        help="Slippage per side as decimal (e.g., 0.01 = 1%%). Applied to both entry and exit. Default: 0 (no slippage)",
    )

    # Candlestick pattern arguments (for position sizing only, NOT signal filtering)
    parser.add_argument(
        "--candlestick-timeframe",
        type=str,
        default="15Min",
        help="Timeframe for candlestick pattern analysis (default: 15Min). Options: 1Min, 3Min, 5Min, 15Min, 30Min",
    )
    parser.add_argument(
        "--use-pattern-averaging",
        action="store_true",
        help="Use candlestick patterns to confirm averaging down",
    )
    parser.add_argument(
        "--pattern-body-threshold",
        type=float,
        default=0.1,
        help="Max body/range ratio for doji detection (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--pattern-shadow-ratio",
        type=float,
        default=2.0,
        help="Min shadow/body ratio for hammer/shooting star (default: 2.0)",
    )

    # Support/Resistance arguments
    parser.add_argument(
        "--use-dynamic-exits",
        action="store_true",
        help="Adjust profit targets and stop losses based on S/R levels",
    )
    parser.add_argument(
        "--sr-lookback",
        type=int,
        default=100,
        help="Number of bars to look back for S/R calculation (default: 100)",
    )

    # Pattern-based position sizing arguments
    parser.add_argument(
        "--use-pattern-sizing",
        action="store_true",
        help="Add bonus contracts for strong candlestick patterns (strength >= threshold)",
    )
    parser.add_argument(
        "--pattern-strength-threshold",
        type=float,
        default=0.8,
        help="Minimum pattern strength (0.0-1.0) to add bonus contracts (default: 0.8)",
    )
    parser.add_argument(
        "--pattern-bonus-contracts",
        type=int,
        default=4,
        help="Number of bonus contracts for strong patterns (default: 4, optimized from 2)",
    )

    # Partial exit arguments
    parser.add_argument(
        "--enable-partial-exits",
        action="store_true",
        help="Enable partial exits: 50%% at base target, rest at extended target for strong patterns",
    )
    parser.add_argument(
        "--extended-target-multiplier",
        type=float,
        default=2.0,
        help="Extended target = base profit × multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--partial-exit-threshold",
        type=float,
        default=0.8,
        help="Pattern strength threshold for extended target (default: 0.8)",
    )

    # Kelly Criterion arguments
    parser.add_argument(
        "--enable-kelly",
        action="store_true",
        help="Enable Kelly Criterion position sizing based on historical win rate",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fraction of full Kelly to use (default: 0.25 = 25%% Kelly, conservative)",
    )
    parser.add_argument(
        "--kelly-lookback",
        type=int,
        default=50,
        help="Number of recent trades to use for Kelly calculation (default: 50)",
    )
    parser.add_argument(
        "--kelly-min-trades",
        type=int,
        default=20,
        help="Minimum trades required before applying Kelly (default: 20)",
    )
    parser.add_argument(
        "--kelly-max-multiplier",
        type=float,
        default=3.0,
        help="Maximum Kelly multiplier cap (default: 3.0)",
    )

    # === MACD Arguments ===
    parser.add_argument(
        "--macd-fast",
        type=int,
        default=12,
        help="MACD fast EMA period (default: 12)",
    )
    parser.add_argument(
        "--macd-slow",
        type=int,
        default=26,
        help="MACD slow EMA period (default: 26)",
    )
    parser.add_argument(
        "--macd-signal",
        type=int,
        default=9,
        help="MACD signal line EMA period (default: 9)",
    )

    # === Bollinger Bands Arguments ===
    parser.add_argument(
        "--bb-period",
        type=int,
        default=20,
        help="Bollinger Bands SMA period (default: 20)",
    )
    parser.add_argument(
        "--bb-std",
        type=float,
        default=2.0,
        help="Bollinger Bands standard deviation multiplier (default: 2.0)",
    )
    parser.add_argument(
        "--bb-width-threshold",
        type=float,
        default=2.0,
        help="Minimum band width %% for volatility filter (default: 2.0)",
    )

    # === Signal Integration Mode ===
    parser.add_argument(
        "--signal-mode",
        type=str,
        default="rsi_only",
        choices=["rsi_only", "macd_filter", "independent", "hybrid"],
        help="Signal mode: rsi_only (default), macd_filter, independent, or hybrid",
    )

    # === Bollinger Band Strategies ===
    parser.add_argument(
        "--bb-entry",
        type=str,
        default="none",
        choices=["none", "touch", "squeeze"],
        help="BB entry strategy: none (default), touch (enter at band touch), squeeze",
    )
    parser.add_argument(
        "--bb-exit",
        type=str,
        default="none",
        choices=["none", "mean_reversion", "opposite_band"],
        help="BB exit strategy: none (default), mean_reversion (middle band), opposite_band",
    )
    parser.add_argument(
        "--bb-volatility-filter",
        action="store_true",
        help="Only trade when BB width >= threshold",
    )

    # === Dual RSI Confirmation Mode ===
    parser.add_argument(
        "--rsi-confirm",
        type=str,
        default="none",
        choices=["none", "both_agree", "5min_trigger", "either_triggers"],
        help="RSI confirmation mode: none (default/parallel), both_agree, 5min_trigger, either_triggers",
    )
    parser.add_argument(
        "--rsi-confirm-buffer",
        type=float,
        default=5.0,
        help="RSI zone confirmation buffer (default: 5.0 points)",
    )

    # === Entry Cutoff Time ===
    parser.add_argument(
        "--entry-cutoff-hour",
        type=int,
        default=19,
        help="UTC hour after which no new entries (17=12PM ET, 18=1PM ET, 19=2PM ET, 20=3PM ET). Default: 19",
    )

    # === ADX Filter ===
    parser.add_argument(
        "--use-adx-filter",
        action="store_true",
        help="Filter trades by ADX (trend strength). ADX < threshold = range-bound market",
    )
    parser.add_argument(
        "--adx-threshold",
        type=float,
        default=25.0,
        help="ADX threshold for filtering (default: 25). Trade when ADX < threshold",
    )
    parser.add_argument(
        "--adx-period",
        type=int,
        default=14,
        help="ADX calculation period (default: 14)",
    )

    # === Expected Move Filter ===
    parser.add_argument(
        "--use-em-filter",
        action="store_true",
        help="Filter trades by Expected Move (price vs EM ratio)",
    )
    parser.add_argument(
        "--em-max-ratio",
        type=float,
        default=0.8,
        help="Max price move vs EM before filtering (default: 0.8 = 80%% of EM)",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Start ThetaTerminal for real option prices
    theta_process = start_theta_terminal()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        print("Use format: YYYY-MM-DD")
        sys.exit(1)

    # Handle symbol and SPX detection
    symbol = args.symbol.upper()
    contract_multiplier = args.multiplier
    strike_interval = 1.0  # Default for SPY
    profit_target = args.profit_target
    strike_offset = args.strike_offset  # Strike offset from ATM
    spx_simulation = False

    # SPX mode: use SPY underlying bars, but trade actual SPXW options
    option_symbol = ""  # Default: same as underlying symbol
    underlying_multiplier = 1.0  # Default: no scaling

    if symbol == "SPX":
        # SPX is an index - Alpaca doesn't provide SPX underlying data
        # Use SPY price bars for RSI, but trade actual SPXW options
        print("SPX requested - using SPY underlying with SPXW options")
        print("(SPY bars for RSI calculation, SPXW options for trading)")
        option_symbol = "SPXW"  # Trade weekly SPX options
        underlying_multiplier = 10.0  # SPX ≈ 10x SPY, so scale strikes
        strike_interval = 5.0  # SPX uses $5 strike intervals
        symbol = "SPY"  # Use SPY for underlying price data
        spx_simulation = True

    # Configure timeframe
    parallel_mode = False
    if args.timeframe == "parallel":
        dual_enabled = False
        parallel_mode = True
        primary_tf = "5Min"
        confirm_tf = "3Min"
        tf_display = "PARALLEL (3-min + 5-min running independently)"
    elif args.timeframe == "dual":
        dual_enabled = True
        primary_tf = "5Min"
        confirm_tf = "3Min"
        tf_display = "Dual (5-min trigger + 3-min confirm)"
    elif args.timeframe == "5min":
        dual_enabled = False
        primary_tf = "5Min"
        confirm_tf = "5Min"
        tf_display = "5-min single timeframe"
    else:  # 3min
        dual_enabled = False
        primary_tf = "3Min"
        confirm_tf = "3Min"
        tf_display = "3-min single timeframe"

    # Determine ITM/OTM display
    if strike_offset > 0:
        strike_type = f"${abs(strike_offset):.0f} OTM"
    elif strike_offset < 0:
        strike_type = f"${abs(strike_offset):.0f} ITM"
    else:
        strike_type = "ATM"

    print("=" * 60)
    print("Marcus RSI Strategy Backtest")
    print("=" * 60)
    if spx_simulation:
        print(f"Symbol:         SPXW options (using SPY for RSI)")
        print(f"Strike Scaling: SPY price x10 -> SPXW strikes")
        # Display scaled strike offset for SPXW
        scaled_offset = strike_offset * underlying_multiplier
        if scaled_offset > 0:
            strike_type = f"${abs(scaled_offset):.0f} OTM"
        elif scaled_offset < 0:
            strike_type = f"${abs(scaled_offset):.0f} ITM"
        else:
            strike_type = "ATM"
    else:
        print(f"Symbol:         {symbol}")
    print(f"Strike:         {strike_type}")
    print(f"Period:         {start_date} to {end_date}")
    print(f"Timeframe:      {tf_display}")
    print(f"RSI Period:     {args.rsi_period}")
    print(f"RSI SMA Period: {args.sma_period}")
    if args.profit_target_pct:
        print(f"Profit Target:  +{args.profit_target_pct:.0%} (percentage-based)")
    else:
        print(f"Profit Target:  +${profit_target:.2f}/contract")
    print(f"Stop Loss:      {args.stop_loss_pct:.0%} below avg entry")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    if args.use_pattern_averaging:
        print(f"Pattern Avg:    Enabled")
    if args.use_dynamic_exits:
        print(f"Dynamic Exits:  S/R-based (lookback: {args.sr_lookback} bars)")
    if args.use_pattern_sizing:
        print(f"Pattern Sizing: +{args.pattern_bonus_contracts} contracts for strength >= {args.pattern_strength_threshold}")
    if args.enable_partial_exits:
        print(f"Partial Exits:  50% at base PT, 50% at {args.extended_target_multiplier}x extended")
    if args.enable_kelly:
        print(f"Kelly Sizing:   {args.kelly_fraction:.0%} Kelly, lookback={args.kelly_lookback}, max={args.kelly_max_multiplier}x")
    if args.signal_mode != "rsi_only":
        print(f"Signal Mode:    {args.signal_mode}")
        print(f"MACD:           fast={args.macd_fast}, slow={args.macd_slow}, signal={args.macd_signal}")
    if args.bb_entry != "none" or args.bb_exit != "none" or args.bb_volatility_filter:
        print(f"Bollinger:      period={args.bb_period}, std={args.bb_std}")
        if args.bb_entry != "none":
            print(f"  BB Entry:     {args.bb_entry}")
        if args.bb_exit != "none":
            print(f"  BB Exit:      {args.bb_exit}")
        if args.bb_volatility_filter:
            print(f"  BB Filter:    width >= {args.bb_width_threshold}%")
    if args.rsi_confirm != "none":
        print(f"RSI Confirm:    {args.rsi_confirm} (buffer: {args.rsi_confirm_buffer})")
    # Display entry cutoff time in ET
    cutoff_et_map = {17: "12:00 PM", 18: "1:00 PM", 19: "2:00 PM", 20: "3:00 PM", 21: "4:00 PM"}
    cutoff_et = cutoff_et_map.get(args.entry_cutoff_hour, f"{args.entry_cutoff_hour}:00 UTC")
    print(f"Entry Cutoff:   {cutoff_et} ET (no new entries after)")
    print(f"Strategy:       RSI Mean Reversion")
    # ADX Filter display
    if args.use_adx_filter:
        print(f"ADX Filter:     Enabled (trade when ADX < {args.adx_threshold}, period={args.adx_period})")
    # Expected Move Filter display
    if args.use_em_filter:
        print(f"EM Filter:      Enabled (max {args.em_max_ratio:.0%} of Expected Move)")
    print("=" * 60)
    print()

    # Create config
    config = BacktestConfig(
        symbol=symbol,  # May have been changed to SPY for SPX simulation
        start_date=start_date,
        end_date=end_date,
        rsi_period=args.rsi_period,
        rsi_sma_period=args.sma_period,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        profit_target_dollars=profit_target,  # Scaled if using multiplier
        profit_target_pct=args.profit_target_pct,  # Percentage-based target (overrides dollars if set)
        stop_loss_pct=args.stop_loss_pct,  # Stop loss percentage from CLI
        max_hold_bars=args.max_hold_bars,
        initial_capital=args.initial_capital,
        dual_timeframe_enabled=dual_enabled,
        primary_timeframe=primary_tf,
        confirmation_timeframe=confirm_tf,
        parallel_mode=parallel_mode,
        contract_multiplier=contract_multiplier,
        strike_interval=strike_interval,
        option_symbol=option_symbol,
        underlying_multiplier=underlying_multiplier,
        strike_offset=strike_offset,
        slippage_entry_pct=args.slippage,
        slippage_exit_pct=args.slippage,
        # Candlestick pattern settings (for position sizing only, NOT signal filtering)
        candlestick_timeframe=args.candlestick_timeframe,
        pattern_body_threshold=args.pattern_body_threshold,
        pattern_shadow_ratio=args.pattern_shadow_ratio,
        use_pattern_for_averaging=args.use_pattern_averaging,
        # Support/Resistance settings
        use_dynamic_exits=args.use_dynamic_exits,
        sr_lookback_bars=args.sr_lookback,
        # Pattern-based position sizing
        use_pattern_position_sizing=args.use_pattern_sizing,
        pattern_strength_threshold=args.pattern_strength_threshold,
        pattern_bonus_contracts=args.pattern_bonus_contracts,
        # Partial exit settings
        enable_partial_exits=args.enable_partial_exits,
        extended_target_multiplier=args.extended_target_multiplier,
        partial_exit_pattern_threshold=args.partial_exit_threshold,
        # Kelly Criterion settings
        enable_kelly_sizing=args.enable_kelly,
        kelly_fraction=args.kelly_fraction,
        kelly_lookback_trades=args.kelly_lookback,
        kelly_min_trades=args.kelly_min_trades,
        kelly_max_multiplier=args.kelly_max_multiplier,
        # MACD settings
        macd_fast_period=args.macd_fast,
        macd_slow_period=args.macd_slow,
        macd_signal_period=args.macd_signal,
        # Bollinger Bands settings
        bb_period=args.bb_period,
        bb_num_std=args.bb_std,
        bb_width_threshold=args.bb_width_threshold,
        # Signal mode
        signal_mode=args.signal_mode,
        # BB strategies
        bb_entry_strategy=args.bb_entry,
        bb_exit_strategy=args.bb_exit,
        bb_volatility_filter=args.bb_volatility_filter,
        # Dual RSI confirmation
        rsi_confirmation_mode=args.rsi_confirm,
        rsi_confirm_buffer=args.rsi_confirm_buffer,
        # Entry cutoff time
        entry_cutoff_hour_utc=args.entry_cutoff_hour,
        # ADX Filter settings
        use_adx_filter=args.use_adx_filter,
        adx_period=args.adx_period,
        adx_threshold=args.adx_threshold,
        # Expected Move Filter settings
        use_em_filter=args.use_em_filter,
        em_max_ratio=args.em_max_ratio,
    )

    # Run backtest
    print("Running backtest...")
    client = AlpacaClient()
    service = BacktestService(client)

    def progress_callback(current, total):
        pct = current / total * 100 if total > 0 else 0
        print(f"\rProgress: {pct:.1f}% ({current}/{total} bars)", end="", flush=True)

    result = await service.run_backtest(config, progress_callback)
    print("\n")

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    metrics = result.metrics
    print("Performance Metrics (Trade-Level):")
    print(f"  Total Trades:     {metrics.total_trades}")
    print(f"  Winning Trades:   {metrics.winning_trades}")
    print(f"  Losing Trades:    {metrics.losing_trades}")
    print(f"  Win Rate:         {metrics.win_rate:.1%}")
    print()
    print("Performance Metrics (Contract-Level - each entry counted independently):")
    print(f"  Total Contracts:  {metrics.total_contracts}")
    print(f"  Winning Entries:  {metrics.winning_contracts}")
    print(f"  Losing Entries:   {metrics.losing_contracts}")
    print(f"  Entry Win Rate:   {metrics.contract_win_rate:.1%}")
    print()
    print(f"  Total P&L:        ${metrics.total_pnl:,.2f}")
    print(f"  Average P&L:      ${metrics.avg_pnl:,.2f}")
    print(f"  Max Drawdown:     {metrics.max_drawdown:.1%}")
    print(f"  Profit Factor:    {metrics.profit_factor:.2f}")
    print(f"  Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
    print()

    # Final equity
    if result.equity_curve:
        final_equity = result.equity_curve[-1][1]
        total_return = (final_equity - args.initial_capital) / args.initial_capital
        print(f"  Final Equity:     ${final_equity:,.2f}")
        print(f"  Total Return:     {total_return:.1%}")
    print()

    # Trade breakdown by exit reason
    if result.trades:
        print("Trade Breakdown by Exit Reason:")
        reasons = {}
        for trade in result.trades:
            reason = trade.exit_reason
            if reason not in reasons:
                reasons[reason] = {"count": 0, "pnl": 0}
            reasons[reason]["count"] += 1
            reasons[reason]["pnl"] += trade.pnl_dollars

        for reason, data in sorted(reasons.items()):
            print(f"  {reason}: {data['count']} trades, ${data['pnl']:,.2f} P&L")

    # Pattern strength analysis (if using pattern features)
    trades_with_patterns = [t for t in result.trades if hasattr(t, 'pattern_strength') and t.pattern_strength > 0]
    if trades_with_patterns:
        print()
        print("Pattern Strength Analysis:")
        strong_patterns = [t for t in trades_with_patterns if t.pattern_strength >= 0.8]
        weak_patterns = [t for t in trades_with_patterns if t.pattern_strength < 0.8]

        if strong_patterns:
            strong_wins = sum(1 for t in strong_patterns if t.pnl_dollars > 0)
            strong_pnl = sum(t.pnl_dollars for t in strong_patterns)
            print(f"  Strong patterns (≥0.8): {len(strong_patterns)} trades, {strong_wins/len(strong_patterns):.1%} win rate, ${strong_pnl:,.2f} P&L")

        if weak_patterns:
            weak_wins = sum(1 for t in weak_patterns if t.pnl_dollars > 0)
            weak_pnl = sum(t.pnl_dollars for t in weak_patterns)
            print(f"  Weak patterns (<0.8):   {len(weak_patterns)} trades, {weak_wins/len(weak_patterns):.1%} win rate, ${weak_pnl:,.2f} P&L")

        # Show pattern breakdown
        patterns = {}
        for t in trades_with_patterns:
            name = t.pattern_name or "Unknown"
            if name not in patterns:
                patterns[name] = {"count": 0, "wins": 0, "pnl": 0}
            patterns[name]["count"] += 1
            if t.pnl_dollars > 0:
                patterns[name]["wins"] += 1
            patterns[name]["pnl"] += t.pnl_dollars

        if patterns:
            print("  By Pattern Type:")
            for name, data in sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True)[:5]:
                win_rate = data["wins"] / data["count"] if data["count"] > 0 else 0
                print(f"    {name}: {data['count']} trades, {win_rate:.1%} win rate, ${data['pnl']:,.2f} P&L")

    # Kelly multiplier analysis (if using Kelly sizing)
    trades_with_kelly = [t for t in result.trades if hasattr(t, 'kelly_multiplier') and t.kelly_multiplier != 1.0]
    if trades_with_kelly:
        print()
        print("Kelly Criterion Analysis:")
        kelly_values = [t.kelly_multiplier for t in trades_with_kelly]
        avg_kelly = sum(kelly_values) / len(kelly_values)
        min_kelly = min(kelly_values)
        max_kelly = max(kelly_values)
        print(f"  Avg multiplier: {avg_kelly:.2f}x (range: {min_kelly:.2f}x - {max_kelly:.2f}x)")
        print(f"  Trades affected: {len(trades_with_kelly)}")

    # Show EOD settlement details (expired positions)
    eod_trades = [t for t in result.trades if t.exit_reason == "end_of_day"]
    if eod_trades:
        print()
        print("EOD Settlement Details (expired positions):")
        for t in eod_trades:
            if t.strike > 0:
                itm_otm = "ITM" if t.exit_price > 0 else "OTM (worthless)"
                print(f"  {t.entry_date.date()}: {t.signal_type.value} strike ${t.strike:.0f} -> {itm_otm}, settled ${t.exit_price:.2f}")

    print()

    # Timeframe breakdown (for parallel mode)
    if result.trades and any(t.timeframe for t in result.trades):
        print("Trade Breakdown by Timeframe:")
        timeframes = {}
        for trade in result.trades:
            tf = trade.timeframe or "unknown"
            if tf not in timeframes:
                timeframes[tf] = {"count": 0, "pnl": 0, "wins": 0}
            timeframes[tf]["count"] += 1
            timeframes[tf]["pnl"] += trade.pnl_dollars
            if trade.pnl_dollars > 0:
                timeframes[tf]["wins"] += 1

        for tf, data in sorted(timeframes.items()):
            win_rate = data["wins"] / data["count"] * 100 if data["count"] > 0 else 0
            print(f"  {tf}: {data['count']} trades, {win_rate:.1f}% win rate, ${data['pnl']:,.2f} P&L")

        print()

    # Show profit target trades with entry prices and % gain
    pt_trades = [t for t in result.trades if t.exit_reason == "profit_target"]
    if pt_trades:
        print("Profit Target Trade Details:")
        for trade in pt_trades:
            pct_gain = (0.50 / trade.entry_price) * 100 if trade.entry_price > 0 else 0
            print(f"  Entry: ${trade.entry_price:.2f} -> +$0.50 = {pct_gain:.1f}% gain")

    # Calculate average holding period for wins vs losses
    print()
    print("Average Holding Period:")
    winners = [t for t in result.trades if t.pnl_dollars > 0]
    losers = [t for t in result.trades if t.pnl_dollars <= 0]

    if winners:
        avg_win_mins = sum((t.exit_date - t.entry_date).total_seconds() / 60 for t in winners) / len(winners)
        print(f"  Winners ({len(winners)} trades): {avg_win_mins:.1f} minutes avg")
    if losers:
        avg_loss_mins = sum((t.exit_date - t.entry_date).total_seconds() / 60 for t in losers) / len(losers)
        print(f"  Losers ({len(losers)} trades):  {avg_loss_mins:.1f} minutes avg")

    # Calculate average price movement every 5 minutes for option contracts
    print()
    print("Average Option Price Movement (5-min intervals):")
    all_movements = {5: [], 10: [], 15: [], 20: [], 30: []}

    for trade in result.trades:
        if not trade.rsi_history:
            continue
        entry_price = trade.entry_price
        entry_time = trade.entry_date

        # Find price at each 5-min interval after entry using rsi_history
        # rsi_history format: [(timestamp, rsi, option_price), ...]
        for mins in all_movements.keys():
            target_time = entry_time + timedelta(minutes=mins)
            # Find closest bar to target time
            closest_price = None
            min_diff = float('inf')
            for item in trade.rsi_history:
                bar_time, rsi_val, opt_price = item[0], item[1], item[2] if len(item) > 2 else None
                if opt_price is None:
                    continue
                if isinstance(bar_time, str):
                    bar_time = datetime.fromisoformat(bar_time.replace('Z', '+00:00'))
                if bar_time.tzinfo is None and entry_time.tzinfo is not None:
                    bar_time = bar_time.replace(tzinfo=entry_time.tzinfo)
                diff = abs((bar_time - target_time).total_seconds())
                if diff < min_diff and diff < 120:  # Within 2 min of target
                    min_diff = diff
                    closest_price = opt_price

            if closest_price is not None:
                movement = closest_price - entry_price
                all_movements[mins].append(movement)

    for mins, movements in all_movements.items():
        if movements:
            avg_move = sum(movements) / len(movements)
            avg_pct = (avg_move / (sum(t.entry_price for t in result.trades) / len(result.trades))) * 100
            print(f"  {mins:2d} min: ${avg_move:+.2f} avg ({avg_pct:+.1f}%) - {len(movements)} samples")

    # Analyze price movement from when RSI hits 30/70 (not entry, but the crossover point)
    print()
    print("Price Movement After RSI Crosses 30/70:")
    print("(Tracking from when RSI first enters extreme zone)")
    rsi_cross_movements = {5: [], 10: [], 15: [], 20: [], 30: [], 60: []}

    for trade in result.trades:
        if not trade.rsi_history or len(trade.rsi_history) < 2:
            continue

        # Find when RSI first crossed into extreme zone (30 for calls, 70 for puts)
        is_call = trade.signal_type.value == "buy_call"
        threshold = 30 if is_call else 70
        cross_time = None
        cross_price = None

        for i, item in enumerate(trade.rsi_history):
            bar_time, rsi_val = item[0], item[1]
            opt_price = item[2] if len(item) > 2 else None
            if opt_price is None:
                continue

            # Check if RSI crossed threshold
            if is_call and rsi_val < threshold:
                cross_time = bar_time
                cross_price = opt_price
                break
            elif not is_call and rsi_val > threshold:
                cross_time = bar_time
                cross_price = opt_price
                break

        if cross_time is None or cross_price is None:
            continue

        if isinstance(cross_time, str):
            cross_time = datetime.fromisoformat(cross_time.replace('Z', '+00:00'))

        # Find prices at intervals after the RSI cross
        for mins in rsi_cross_movements.keys():
            target_time = cross_time + timedelta(minutes=mins)
            closest_price = None
            min_diff = float('inf')

            for item in trade.rsi_history:
                bar_time, rsi_val, opt_price = item[0], item[1], item[2] if len(item) > 2 else None
                if opt_price is None:
                    continue
                if isinstance(bar_time, str):
                    bar_time = datetime.fromisoformat(bar_time.replace('Z', '+00:00'))

                diff = abs((bar_time - target_time).total_seconds())
                if diff < min_diff and diff < 120:
                    min_diff = diff
                    closest_price = opt_price

            if closest_price is not None:
                movement = closest_price - cross_price
                rsi_cross_movements[mins].append((movement, cross_price))

    for mins, data in rsi_cross_movements.items():
        if data:
            movements = [d[0] for d in data]
            prices = [d[1] for d in data]
            avg_move = sum(movements) / len(movements)
            avg_entry = sum(prices) / len(prices)
            avg_pct = (avg_move / avg_entry) * 100 if avg_entry > 0 else 0
            print(f"  {mins:2d} min: ${avg_move:+.2f} avg ({avg_pct:+.1f}%) - {len(data)} samples")

    # Analyze SPY price movement after RSI crosses 30/70
    print()
    print("SPY Price Movement After RSI Crosses 30/70:")
    spy_movements = {5: [], 10: [], 15: [], 20: [], 30: [], 60: []}

    for trade in result.trades:
        if not trade.rsi_history or len(trade.rsi_history) < 2:
            continue

        is_call = trade.signal_type.value == "buy_call"
        threshold = 30 if is_call else 70
        cross_time = None
        cross_idx = None

        # Find when RSI first crossed threshold in rsi_history
        for i, item in enumerate(trade.rsi_history):
            rsi_val = item[1]
            if is_call and rsi_val < threshold:
                cross_time = item[0]
                cross_idx = i
                break
            elif not is_call and rsi_val > threshold:
                cross_time = item[0]
                cross_idx = i
                break

        if cross_time is None:
            continue

        if isinstance(cross_time, str):
            cross_time = datetime.fromisoformat(cross_time.replace('Z', '+00:00'))

        # We need SPY prices - check if rsi_history has underlying price (4th element)
        # If not, we need to get it from elsewhere
        # For now, check the structure
        if len(trade.rsi_history[0]) > 3:
            cross_spy = trade.rsi_history[cross_idx][3]  # underlying price

            for mins in spy_movements.keys():
                target_time = cross_time + timedelta(minutes=mins)
                closest_spy = None
                min_diff = float('inf')

                for item in trade.rsi_history:
                    if len(item) <= 3:
                        continue
                    bar_time = item[0]
                    spy_price = item[3]
                    if isinstance(bar_time, str):
                        bar_time = datetime.fromisoformat(bar_time.replace('Z', '+00:00'))

                    diff = abs((bar_time - target_time).total_seconds())
                    if diff < min_diff and diff < 120:
                        min_diff = diff
                        closest_spy = spy_price

                if closest_spy is not None:
                    movement = closest_spy - cross_spy
                    # For calls, we want SPY to go UP; for puts, we want SPY to go DOWN
                    # Show directional movement (positive = favorable)
                    if not is_call:
                        movement = -movement  # Invert for puts so positive = favorable
                    spy_movements[mins].append((movement, cross_spy, is_call))

    if any(spy_movements.values()):
        print("(Positive = favorable direction for the trade)")
        for mins, data in spy_movements.items():
            if data:
                movements = [d[0] for d in data]
                avg_move = sum(movements) / len(movements)
                print(f"  {mins:2d} min: ${avg_move:+.2f} avg - {len(data)} samples")
    else:
        # rsi_history doesn't have SPY prices, need to add them
        print("  (SPY prices not tracked in rsi_history - adding support...)")

    print()
    print("=" * 60)

    # Show RSI analysis for end-of-day trades
    eod_trades = [t for t in result.trades if t.exit_reason == "end_of_day"]
    if eod_trades:
        print("\n" + "=" * 60)
        print("RSI ANALYSIS FOR END-OF-DAY TRADES")
        print("=" * 60)
        for trade in eod_trades[:5]:  # Show first 5 EOD trades
            print(f"\n--- {trade.signal_type.value} at {trade.entry_date} ---")
            print(f"Entry: RSI={trade.entry_rsi:.1f}, Price=${trade.entry_price:.2f}")
            print(f"Exit:  Price=${trade.exit_price:.2f}, P&L=${trade.pnl_dollars:.2f}")
            prices_available = sum(1 for _, _, p, *_ in trade.rsi_history if p is not None)
            print(f"Price data points: {prices_available}/{len(trade.rsi_history)}")
            # Show RSI range during trade
            rsi_values = [r for _, r, *_ in trade.rsi_history]
            if rsi_values:
                min_rsi, max_rsi = min(rsi_values), max(rsi_values)
                print(f"RSI range: {min_rsi:.1f} - {max_rsi:.1f}")
                # For PUT trades, did RSI go below 60? For CALL trades, did RSI go above 40?
                if "put" in trade.signal_type.value:
                    reached_target = any(r <= 60 for r in rsi_values)
                    print(f"RSI reached <=60 (PUT target): {'YES' if reached_target else 'NO'}")
                else:
                    reached_target = any(r >= 40 for r in rsi_values)
                    print(f"RSI reached >=40 (CALL target): {'YES' if reached_target else 'NO'}")
        if len(eod_trades) > 5:
            print(f"\n... and {len(eod_trades) - 5} more EOD trades")


if __name__ == "__main__":
    asyncio.run(main())
