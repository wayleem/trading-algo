#!/usr/bin/env python3
"""
Monte Carlo simulation CLI for SPXW backtest strategy.

Runs a backtest with specified parameters, then performs bootstrap resampling
of trade P&Ls to generate probability distributions of strategy outcomes.

Usage:
    python monte_carlo.py --start 2022-12-01 --end 2025-12-10 --symbol SPX \
        --profit-target-pct 0.10 --stop-loss-pct 0.45 --slippage 0.01 \
        --simulations 10000
"""

import argparse
import asyncio
import subprocess
import sys
import time
import httpx
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.backtest import BacktestService, BacktestConfig
from app.services.backtest.monte_carlo import (
    MonteCarloSimulator,
    print_monte_carlo_results,
)
from app.services.alpaca_client import AlpacaClient


def start_theta_terminal():
    """Start ThetaTerminal if not already running."""
    theta_jar = Path(__file__).parent.parent / "ThetaTerminalv3.jar"

    if not theta_jar.exists():
        print(f"Warning: ThetaTerminal not found at {theta_jar}")
        return None

    # Check if already running
    try:
        response = httpx.get(
            "http://localhost:25503/v3/option/list/expirations?symbol=SPY&format=json",
            timeout=2.0,
        )
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
    for i in range(30):
        time.sleep(1)
        try:
            response = httpx.get(
                "http://localhost:25503/v3/option/list/expirations?symbol=SPY&format=json",
                timeout=2.0,
            )
            if response.status_code == 200:
                print("ThetaTerminal started successfully")
                return process
        except:
            pass
        print(f"Waiting for ThetaTerminal... ({i+1}/30)")

    print("Warning: ThetaTerminal did not start in time")
    return process


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulation on RSI strategy backtest"
    )

    # Backtest parameters
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
        default="SPX",
        help="Symbol to backtest (default: SPX)",
    )
    parser.add_argument(
        "--strike-offset",
        type=float,
        default=2.0,
        help="Strike offset from ATM (default: 2 = $20 OTM for SPX)",
    )
    parser.add_argument(
        "--profit-target-pct",
        type=float,
        default=0.10,
        help="Profit target as decimal (default: 0.10 for 10%%)",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.45,
        help="Stop loss as decimal (default: 0.45 for 45%%)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.01,
        help="Slippage per side as decimal (default: 0.01 for 1%%)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)",
    )

    # Monte Carlo parameters
    parser.add_argument(
        "--simulations",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    return parser.parse_args()


async def run_backtest(args) -> list[float]:
    """
    Run backtest and return list of trade P&Ls.

    Returns:
        List of individual trade P&L values in dollars
    """
    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    # Handle SPX -> SPY mapping
    symbol = args.symbol.upper()
    option_symbol = ""
    underlying_multiplier = 1.0
    strike_interval = 1.0
    contract_multiplier = 1
    strike_offset = args.strike_offset

    if symbol == "SPX":
        print("SPX requested - using SPY underlying with SPXW options")
        option_symbol = "SPXW"
        underlying_multiplier = 10.0
        strike_interval = 5.0
        # strike_offset stays as-is, underlying_multiplier handles the scaling
        symbol = "SPY"

    # Create config
    config = BacktestConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        rsi_period=14,
        rsi_sma_period=14,
        profit_target_dollars=0.50,
        profit_target_pct=args.profit_target_pct,
        stop_loss_pct=args.stop_loss_pct,
        max_hold_bars=3,
        initial_capital=args.initial_capital,
        dual_timeframe_enabled=True,
        primary_timeframe="5Min",
        confirmation_timeframe="3Min",
        parallel_mode=False,
        contract_multiplier=contract_multiplier,
        strike_interval=strike_interval,
        option_symbol=option_symbol,
        underlying_multiplier=underlying_multiplier,
        strike_offset=strike_offset,
        slippage_pct=args.slippage,
    )

    # Run backtest
    print("Running backtest to collect trade data...")
    client = AlpacaClient()
    service = BacktestService(client)

    def progress_callback(current, total):
        pct = current / total * 100 if total > 0 else 0
        print(f"\rBacktest Progress: {pct:.1f}% ({current}/{total} bars)", end="", flush=True)

    result = await service.run_backtest(config, progress_callback)
    print("\n")

    # Extract trade P&Ls
    trade_pnls = [trade.pnl_dollars for trade in result.trades]

    return trade_pnls, result


async def main():
    args = parse_args()

    # Start ThetaTerminal
    theta_process = start_theta_terminal()

    print("=" * 60)
    print("Monte Carlo Simulation")
    print("=" * 60)
    print(f"Strategy Parameters:")
    print(f"  Symbol:         {args.symbol}")
    print(f"  Strike Offset:  ${args.strike_offset * 10 if args.symbol.upper() == 'SPX' else args.strike_offset} OTM")
    print(f"  Profit Target:  {args.profit_target_pct * 100:.0f}%")
    print(f"  Stop Loss:      {args.stop_loss_pct * 100:.0f}%")
    print(f"  Slippage:       {args.slippage * 100:.1f}%")
    print(f"  Period:         {args.start} to {args.end}")
    print(f"  Initial Capital: ${args.initial_capital:,.2f}")
    print()
    print(f"Monte Carlo Parameters:")
    print(f"  Simulations:    {args.simulations:,}")
    print(f"  Random Seed:    {args.seed if args.seed else 'None (random)'}")
    print("=" * 60)
    print()

    # Run backtest to get trade data
    trade_pnls, backtest_result = await run_backtest(args)

    print(f"Backtest completed: {len(trade_pnls)} trades")
    print(f"  Total P&L:      ${backtest_result.metrics.total_pnl:,.2f}")
    print(f"  Win Rate:       {backtest_result.metrics.win_rate:.1f}%")
    print(f"  Sharpe Ratio:   {backtest_result.metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:   {backtest_result.metrics.max_drawdown:.1f}%")
    print()

    if len(trade_pnls) < 10:
        print("Error: Not enough trades for meaningful Monte Carlo simulation (need at least 10)")
        return

    # Run Monte Carlo simulation
    print(f"Running {args.simulations:,} Monte Carlo simulations...")
    simulator = MonteCarloSimulator(
        pnls=trade_pnls,
        initial_capital=args.initial_capital,
    )

    result = simulator.run_simulations(
        n_simulations=args.simulations,
        seed=args.seed,
    )

    print()
    print_monte_carlo_results(result)


if __name__ == "__main__":
    asyncio.run(main())
