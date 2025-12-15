#!/usr/bin/env python3
"""
CLI script to run paper trading with parallel dual-timeframe RSI strategy.

This matches the optimized backtest configuration:
- Parallel 3-min and 5-min RSI streams (independent signals)
- $0.50 OTM strike offset
- 1% slippage accounting
- Pattern-based position sizing

Usage:
    python run_paper_trading.py
    python run_paper_trading.py --symbol SPY --profit-target 0.18
"""

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import signal as signal_module
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.alpaca_client import AlpacaClient
from app.services.indicators import IndicatorService
from app.services.signal_generator import SignalGenerator
from app.services.order_executor import OrderExecutor
from app.services.position_manager import PositionManager
from app.models.schemas import SignalType
from app.core.config import settings


# Global flag for graceful shutdown
running = True


@dataclass
class TimeframeState:
    """State for a single timeframe's RSI strategy."""
    label: str  # "3Min" or "5Min"
    rsi_history: list = field(default_factory=list)
    current_rsi: float = 50.0
    current_sma: float = 50.0
    current_trade: Optional[dict] = None
    last_bar_time: Optional[datetime] = None


def signal_handler(sig, frame):
    global running
    print("\nShutdown requested...")
    running = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run RSI paper trading bot (parallel mode)")

    parser.add_argument(
        "--symbol",
        type=str,
        default=settings.symbol,
        help=f"Symbol to trade (default: {settings.symbol})",
    )
    parser.add_argument(
        "--profit-target",
        type=float,
        default=settings.profit_target_pct,
        help=f"Profit target percentage (default: {settings.profit_target_pct})",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=settings.stop_loss_pct,
        help=f"Stop loss percentage (default: {settings.stop_loss_pct})",
    )
    parser.add_argument(
        "--contracts",
        type=int,
        default=settings.contracts_per_trade,
        help=f"Contracts per trade (default: {settings.contracts_per_trade})",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=settings.initial_capital,
        help=f"Virtual capital to simulate (default: ${settings.initial_capital:,.0f})",
    )
    parser.add_argument(
        "--daily-loss-limit",
        type=float,
        default=settings.daily_loss_limit,
        help=f"Daily loss limit (default: ${settings.daily_loss_limit:,.0f})",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel mode (use single timeframe)",
    )

    return parser.parse_args()


def get_strike_with_offset(underlying_price: float, option_type: str, offset: float) -> float:
    """
    Calculate strike price with OTM offset.

    Args:
        underlying_price: Current underlying price
        option_type: "call" or "put"
        offset: Distance from ATM (positive = OTM)

    Returns:
        Strike price rounded to nearest $1
    """
    if option_type == "call":
        # OTM call = strike above price
        strike = underlying_price + offset
    else:
        # OTM put = strike below price
        strike = underlying_price - offset
    return round(strike)


def get_zone(rsi: float) -> str:
    """Get RSI zone label."""
    if rsi < settings.rsi_oversold:
        return "OVERSOLD"
    elif rsi > settings.rsi_overbought:
        return "OVERBOUGHT"
    return "NEUTRAL"


async def main():
    global running

    args = parse_args()

    # Update settings from args
    settings.symbol = args.symbol
    settings.profit_target_pct = args.profit_target
    settings.stop_loss_pct = args.stop_loss
    settings.contracts_per_trade = args.contracts
    settings.initial_capital = args.capital
    settings.daily_loss_limit = args.daily_loss_limit

    parallel_mode = settings.parallel_mode and not args.no_parallel

    # Virtual capital tracking
    virtual_capital = settings.initial_capital
    daily_pnl = 0.0

    print("=" * 70)
    print("Marcus RSI Paper Trading Bot (Optimized)")
    print("=" * 70)
    print(f"Symbol:         {settings.symbol}")
    print(f"Mode:           {'PARALLEL (3-min + 5-min)' if parallel_mode else 'SINGLE (1-min)'}")
    print(f"RSI Period:     {settings.rsi_period}")
    print(f"RSI SMA Period: {settings.rsi_sma_period}")
    print(f"RSI Thresholds: Oversold < {settings.rsi_oversold}, Overbought > {settings.rsi_overbought}")
    print(f"Profit Target:  {settings.profit_target_pct:.0%}")
    print(f"Stop Loss:      {settings.stop_loss_pct:.0%}")
    print(f"Strike Offset:  ${settings.strike_offset:.2f} OTM")
    print(f"Slippage:       {settings.slippage_pct:.0%}")
    print(f"Contracts:      {settings.contracts_per_trade}")
    print(f"Paper Trading:  {settings.alpaca_paper}")
    print("-" * 70)
    print(f"Virtual Capital:   ${virtual_capital:,.2f}")
    print(f"Daily Loss Limit:  ${settings.daily_loss_limit:,.2f}")
    print(f"Max Risk/Trade:    {settings.max_risk_per_trade_pct:.0%}")
    print("=" * 70)
    print()

    # Initialize components
    print("Connecting to Alpaca...")
    client = AlpacaClient()

    try:
        account = client.get_account()
        print(f"Connected! Account: {account.account_number}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print()
    except Exception as e:
        print(f"Failed to connect to Alpaca: {e}")
        print("Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set in .env")
        sys.exit(1)

    # Create indicator services for each timeframe
    indicator_3min = IndicatorService(
        rsi_period=settings.rsi_period,
        sma_period=settings.rsi_sma_period,
    )
    indicator_5min = IndicatorService(
        rsi_period=settings.rsi_period,
        sma_period=settings.rsi_sma_period,
    )

    signal_generator_3min = SignalGenerator()  # Separate instance for 3-min timeframe
    signal_generator_5min = SignalGenerator()  # Separate instance for 5-min timeframe
    executor = OrderExecutor(client)
    position_manager = PositionManager(client, executor)

    # Initialize timeframe states
    state_3min = TimeframeState(label="3Min")
    state_5min = TimeframeState(label="5Min")

    print("Starting trading loop (Ctrl+C to stop)...")
    print()

    # Set up signal handler for graceful shutdown
    signal_module.signal(signal_module.SIGINT, signal_handler)
    signal_module.signal(signal_module.SIGTERM, signal_handler)

    # Track loop timing
    last_3min_update = None
    last_5min_update = None

    while running:
        try:
            # Check market hours
            if not position_manager.is_market_hours():
                if not client.is_market_open():
                    next_open = client.get_next_market_open()
                    print(f"\rMarket closed. Next open: {next_open}", end="", flush=True)
                await asyncio.sleep(60)
                continue

            now = datetime.now()
            total_pnl = executor.get_total_pnl()
            current_equity = virtual_capital + total_pnl

            # Check daily loss limit
            if total_pnl <= -settings.daily_loss_limit:
                print(f"\n[RISK] Daily loss limit (${settings.daily_loss_limit:.2f}) reached. Stopping trading.")
                running = False
                continue

            if parallel_mode:
                # PARALLEL MODE: Fetch both 3-min and 5-min bars
                bars_3min, bars_5min = await asyncio.gather(
                    client.get_stock_bars(
                        symbol=settings.symbol,
                        timeframe="3Min",
                        limit=settings.rsi_period + settings.rsi_sma_period + 5,
                    ),
                    client.get_stock_bars(
                        symbol=settings.symbol,
                        timeframe="5Min",
                        limit=settings.rsi_period + settings.rsi_sma_period + 5,
                    ),
                )

                # Process 3-min timeframe
                if len(bars_3min) >= settings.rsi_period + 1:
                    latest_3min = bars_3min[-1]
                    bar_time_3min = latest_3min["timestamp"]

                    # Only update if we have a new bar
                    if state_3min.last_bar_time != bar_time_3min:
                        state_3min.last_bar_time = bar_time_3min
                        closes_3min = [bar["close"] for bar in bars_3min]
                        state_3min.current_rsi = indicator_3min.calculate_rsi(closes_3min)

                        state_3min.rsi_history.append(state_3min.current_rsi)
                        if len(state_3min.rsi_history) > settings.rsi_sma_period * 2:
                            state_3min.rsi_history = state_3min.rsi_history[-settings.rsi_sma_period * 2:]

                        state_3min.current_sma = indicator_3min.calculate_sma(state_3min.rsi_history)

                        # Generate signal for 3-min
                        signal_3min = signal_generator_3min.evaluate(
                            current_rsi=state_3min.current_rsi,
                            current_sma=state_3min.current_sma,
                            close_price=latest_3min["close"],
                            timestamp=bar_time_3min,
                        )

                        # Execute trade if signal and no open position for this timeframe
                        if state_3min.current_trade is None and signal_3min.signal_type != SignalType.NO_SIGNAL:
                            option_type = "call" if signal_3min.signal_type == SignalType.BUY_CALL else "put"
                            strike = get_strike_with_offset(latest_3min["close"], option_type, settings.strike_offset)
                            trade = executor.execute_signal(signal_3min)
                            if trade:
                                state_3min.current_trade = {
                                    "trade": trade,
                                    "entry_price": trade.entry_price * (1 + settings.slippage_pct),
                                    "strike": strike,
                                }
                                print(f"\n[3Min TRADE] Opened {trade.signal_type.value}: {trade.option_symbol} @ strike ${strike}")

                # Process 5-min timeframe
                if len(bars_5min) >= settings.rsi_period + 1:
                    latest_5min = bars_5min[-1]
                    bar_time_5min = latest_5min["timestamp"]

                    # Only update if we have a new bar
                    if state_5min.last_bar_time != bar_time_5min:
                        state_5min.last_bar_time = bar_time_5min
                        closes_5min = [bar["close"] for bar in bars_5min]
                        state_5min.current_rsi = indicator_5min.calculate_rsi(closes_5min)

                        state_5min.rsi_history.append(state_5min.current_rsi)
                        if len(state_5min.rsi_history) > settings.rsi_sma_period * 2:
                            state_5min.rsi_history = state_5min.rsi_history[-settings.rsi_sma_period * 2:]

                        state_5min.current_sma = indicator_5min.calculate_sma(state_5min.rsi_history)

                        # Generate signal for 5-min
                        signal_5min = signal_generator_5min.evaluate(
                            current_rsi=state_5min.current_rsi,
                            current_sma=state_5min.current_sma,
                            close_price=latest_5min["close"],
                            timestamp=bar_time_5min,
                        )

                        # Execute trade if signal and no open position for this timeframe
                        if state_5min.current_trade is None and signal_5min.signal_type != SignalType.NO_SIGNAL:
                            option_type = "call" if signal_5min.signal_type == SignalType.BUY_CALL else "put"
                            strike = get_strike_with_offset(latest_5min["close"], option_type, settings.strike_offset)
                            trade = executor.execute_signal(signal_5min)
                            if trade:
                                state_5min.current_trade = {
                                    "trade": trade,
                                    "entry_price": trade.entry_price * (1 + settings.slippage_pct),
                                    "strike": strike,
                                }
                                print(f"\n[5Min TRADE] Opened {trade.signal_type.value}: {trade.option_symbol} @ strike ${strike}")

                # Status output for parallel mode
                now_str = now.strftime("%H:%M:%S")
                latest_price = bars_3min[-1]["close"] if bars_3min else 0

                zone_3min = get_zone(state_3min.current_rsi)
                zone_5min = get_zone(state_5min.current_rsi)

                trade_3min_str = "None" if state_3min.current_trade is None else f"OPEN"
                trade_5min_str = "None" if state_5min.current_trade is None else f"OPEN"

                print(f"[{now_str}] ${latest_price:.2f} | "
                      f"3Min: RSI {state_3min.current_rsi:5.1f} {zone_3min:>10} [{trade_3min_str}] | "
                      f"5Min: RSI {state_5min.current_rsi:5.1f} {zone_5min:>10} [{trade_5min_str}] | "
                      f"Equity: ${current_equity:,.0f}")

            else:
                # SINGLE TIMEFRAME MODE (fallback)
                bars = await client.get_stock_bars(
                    symbol=settings.symbol,
                    timeframe="1Min",
                    limit=settings.rsi_period + settings.rsi_sma_period + 5,
                )

                if len(bars) < settings.rsi_period + 1:
                    print("Waiting for sufficient data...")
                    await asyncio.sleep(60)
                    continue

                closes = [bar["close"] for bar in bars]
                current_rsi = indicator_3min.calculate_rsi(closes)

                state_3min.rsi_history.append(current_rsi)
                if len(state_3min.rsi_history) > settings.rsi_sma_period * 2:
                    state_3min.rsi_history = state_3min.rsi_history[-settings.rsi_sma_period * 2:]

                current_sma = indicator_3min.calculate_sma(state_3min.rsi_history)

                latest_bar = bars[-1]
                signal = signal_generator_3min.evaluate(
                    current_rsi=current_rsi,
                    current_sma=current_sma,
                    close_price=latest_bar["close"],
                    timestamp=latest_bar["timestamp"],
                )

                now_str = now.strftime("%H:%M:%S")
                bar_time = latest_bar["timestamp"]
                if hasattr(bar_time, 'strftime'):
                    bar_time_str = bar_time.strftime("%H:%M")
                else:
                    bar_time_str = str(bar_time)[-8:-3]

                open_trades = len(executor.get_open_trades())
                zone = get_zone(current_rsi)

                status = (
                    f"[{now_str}] Bar:{bar_time_str} | "
                    f"RSI:{current_rsi:5.1f} | "
                    f"SMA:{current_sma:5.1f} | "
                    f"{zone:>10} | "
                    f"${latest_bar['close']:.2f} | "
                    f"Equity:${current_equity:,.0f} | "
                    f"P&L:${total_pnl:+,.0f}"
                )
                print(status)

                # Execute signals if no open position
                if open_trades == 0:
                    trade = executor.execute_signal(signal)
                    if trade:
                        print(f"\n[TRADE] Opened {trade.signal_type.value}: {trade.option_symbol}")

            # Check exits for all open positions
            closed = await position_manager.check_exits()
            for trade in closed:
                # Apply slippage on exit
                actual_pnl = trade.pnl * (1 - settings.slippage_pct)
                daily_pnl += actual_pnl

                # Clear the trade from the appropriate timeframe state
                if state_3min.current_trade and state_3min.current_trade.get("trade") == trade:
                    state_3min.current_trade = None
                if state_5min.current_trade and state_5min.current_trade.get("trade") == trade:
                    state_5min.current_trade = None

                print(f"\n[CLOSED] {trade.option_symbol} | P&L: ${actual_pnl:+.2f} | Reason: {trade.status.value}")

            # Check for averaging down opportunities on open positions
            for trade in executor.get_open_trades():
                updated = await position_manager.check_and_average_down(trade)
                if updated:
                    print(f"\n[AVG DOWN] {updated.option_symbol} | Now {updated.total_contracts} contracts @ avg ${updated.avg_entry_price:.2f}")

            # Fast P/L monitoring loop - check every 5 seconds for PT/SL
            # This runs 12 times before we check signals again (60s / 5s = 12)
            for _ in range(12):
                if not running:
                    break

                # Quick exit check
                closed = await position_manager.check_exits()
                for trade in closed:
                    actual_pnl = trade.pnl * (1 - settings.slippage_pct)
                    daily_pnl += actual_pnl

                    if state_3min.current_trade and state_3min.current_trade.get("trade") == trade:
                        state_3min.current_trade = None
                    if state_5min.current_trade and state_5min.current_trade.get("trade") == trade:
                        state_5min.current_trade = None

                    print(f"\n[CLOSED] {trade.option_symbol} | P&L: ${actual_pnl:+.2f} | Reason: {trade.status.value}")

                # Check averaging down
                for trade in executor.get_open_trades():
                    updated = await position_manager.check_and_average_down(trade)
                    if updated:
                        print(f"\n[AVG DOWN] {updated.option_symbol} | Now {updated.total_contracts} contracts @ avg ${updated.avg_entry_price:.2f}")

                await asyncio.sleep(settings.pl_check_interval_seconds)

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(60)

    # Cleanup
    print("\n")
    print("Shutting down...")

    # Close any open positions
    open_trades = executor.get_open_trades()
    if open_trades:
        print(f"Closing {len(open_trades)} open positions...")
        closed = await position_manager.force_close_all(reason="shutdown")
        for trade in closed:
            actual_pnl = trade.pnl * (1 - settings.slippage_pct)
            print(f"  Closed {trade.option_symbol}: ${actual_pnl:.2f}")

    # Final summary
    final_pnl = executor.get_total_pnl()
    final_equity = virtual_capital + final_pnl
    print()
    print("=" * 70)
    print("Session Summary")
    print("=" * 70)
    print(f"Mode:             {'PARALLEL (3-min + 5-min)' if parallel_mode else 'SINGLE (1-min)'}")
    print(f"Starting Capital: ${virtual_capital:,.2f}")
    print(f"Final Equity:     ${final_equity:,.2f}")
    print(f"Total P&L:        ${final_pnl:+,.2f} ({final_pnl/virtual_capital*100:+.1f}%)")
    print(f"Total Trades:     {len(executor.get_closed_trades())}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
