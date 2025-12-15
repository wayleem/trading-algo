#!/usr/bin/env python3
"""
Compare different signal modes and RSI confirmation modes via backtest.

Runs backtests across all configured modes and generates a comparison table.

Usage:
    python compare_modes.py --start 2023-01-01 --end 2025-01-01
    python compare_modes.py --start 2024-01-01 --end 2024-12-01 --modes signal
    python compare_modes.py --start 2024-01-01 --end 2024-12-01 --modes rsi_confirm
    python compare_modes.py --start 2024-01-01 --end 2024-12-01 --modes bb
    python compare_modes.py --start 2024-01-01 --end 2024-12-01 --modes all
"""

import argparse
import asyncio
import subprocess
import sys
import time
import httpx
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import json

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
    for i in range(30):
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


# Test configurations to compare
SIGNAL_MODE_CONFIGS = [
    {"name": "RSI Only (Baseline)", "signal_mode": "rsi_only"},
    {"name": "MACD Filter", "signal_mode": "macd_filter"},
    {"name": "Independent", "signal_mode": "independent"},
    {"name": "Hybrid", "signal_mode": "hybrid"},
]

RSI_CONFIRM_CONFIGS = [
    {"name": "Parallel (None)", "rsi_confirmation_mode": "none"},
    {"name": "Both Agree", "rsi_confirmation_mode": "both_agree"},
    {"name": "5min Trigger", "rsi_confirmation_mode": "5min_trigger"},
    {"name": "Either Triggers", "rsi_confirmation_mode": "either_triggers"},
]

BB_ENTRY_CONFIGS = [
    {"name": "BB Entry: None", "bb_entry_strategy": "none"},
    {"name": "BB Entry: Touch", "bb_entry_strategy": "touch"},
]

BB_EXIT_CONFIGS = [
    {"name": "BB Exit: None", "bb_exit_strategy": "none"},
    {"name": "BB Exit: Mean Reversion", "bb_exit_strategy": "mean_reversion"},
    {"name": "BB Exit: Opposite Band", "bb_exit_strategy": "opposite_band"},
]

BB_FILTER_CONFIGS = [
    {"name": "No Volatility Filter", "bb_volatility_filter": False},
    {"name": "With Volatility Filter", "bb_volatility_filter": True},
]


def get_base_config(start_date, end_date, symbol="SPY"):
    """Get base backtest config with optimized parameters."""
    option_symbol = ""
    underlying_multiplier = 1.0
    strike_interval = 1.0

    if symbol == "SPX":
        option_symbol = "SPXW"
        underlying_multiplier = 10.0
        strike_interval = 5.0
        symbol = "SPY"

    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "rsi_period": 20,
        "rsi_sma_period": 12,
        "rsi_oversold": 44.0,
        "rsi_overbought": 60.0,
        "profit_target_dollars": 0.50,
        "stop_loss_pct": 0.25,
        "initial_capital": 10000.0,
        "dual_timeframe_enabled": True,
        "primary_timeframe": "5Min",
        "confirmation_timeframe": "3Min",
        "parallel_mode": False,
        "contract_multiplier": 1,
        "strike_interval": strike_interval,
        "option_symbol": option_symbol,
        "underlying_multiplier": underlying_multiplier,
        "strike_offset": 0.5,
        "slippage_pct": 0.01,
        # Default indicator settings
        "macd_fast_period": 12,
        "macd_slow_period": 26,
        "macd_signal_period": 9,
        "bb_period": 20,
        "bb_num_std": 2.0,
        "bb_width_threshold": 2.0,
        # Defaults for modes
        "signal_mode": "rsi_only",
        "bb_entry_strategy": "none",
        "bb_exit_strategy": "none",
        "bb_volatility_filter": False,
        "rsi_confirmation_mode": "none",
        "rsi_confirm_buffer": 5.0,
    }


async def run_single_backtest(config_dict: dict, client: AlpacaClient) -> dict:
    """Run a single backtest and return key metrics."""
    config = BacktestConfig(**config_dict)
    service = BacktestService(client)

    try:
        result = await service.run_backtest(config)
        metrics = result.metrics

        return {
            "trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "total_pnl": metrics.total_pnl,
            "avg_pnl": metrics.avg_pnl,
            "max_drawdown": metrics.max_drawdown,
            "sharpe_ratio": metrics.sharpe_ratio,
            "profit_factor": metrics.profit_factor,
            "contracts": metrics.total_contracts,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {
            "trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "contracts": 0,
            "error": str(e),
        }


def print_comparison_table(results: List[Dict[str, Any]], title: str):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print(f" {title}")
    print("=" * 100)

    # Header
    print(f"{'Mode':<25} {'Trades':>7} {'Win%':>7} {'Total P&L':>12} {'Avg P&L':>10} {'Drawdown':>10} {'Sharpe':>8} {'PF':>6}")
    print("-" * 100)

    # Sort by total P&L descending
    sorted_results = sorted(results, key=lambda x: x["metrics"]["total_pnl"], reverse=True)

    for r in sorted_results:
        m = r["metrics"]
        error_mark = " *" if "error" in m else ""
        print(
            f"{r['name']:<25} "
            f"{m['trades']:>7} "
            f"{m['win_rate']*100:>6.1f}% "
            f"${m['total_pnl']:>10,.2f} "
            f"${m['avg_pnl']:>9,.2f} "
            f"{m['max_drawdown']*100:>9.1f}% "
            f"{m['sharpe_ratio']:>8.2f} "
            f"{m['profit_factor']:>6.2f}{error_mark}"
        )

    print("-" * 100)
    print("* = Error occurred during backtest\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare signal/RSI confirmation modes")

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
        help="Symbol to test (SPY or SPX, default: SPX)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="all",
        choices=["signal", "rsi_confirm", "bb_entry", "bb_exit", "bb_filter", "all"],
        help="Which modes to compare (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output file for results",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # Start ThetaTerminal
    theta_process = start_theta_terminal()

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        sys.exit(1)

    print(f"\nComparing modes for {args.symbol} from {start_date} to {end_date}")
    print(f"Mode category: {args.modes}\n")

    client = AlpacaClient()
    base_config = get_base_config(start_date, end_date, args.symbol)
    all_results = {}

    # Determine which comparisons to run
    run_signal = args.modes in ("signal", "all")
    run_rsi_confirm = args.modes in ("rsi_confirm", "all")
    run_bb_entry = args.modes in ("bb_entry", "all")
    run_bb_exit = args.modes in ("bb_exit", "all")
    run_bb_filter = args.modes in ("bb_filter", "all")

    # Run Signal Mode comparisons
    if run_signal:
        print("\n--- Running Signal Mode Comparisons ---")
        results = []
        for cfg in SIGNAL_MODE_CONFIGS:
            print(f"  Testing: {cfg['name']}...")
            test_config = base_config.copy()
            test_config["signal_mode"] = cfg["signal_mode"]
            metrics = await run_single_backtest(test_config, client)
            results.append({"name": cfg["name"], "metrics": metrics})

        print_comparison_table(results, "Signal Mode Comparison")
        all_results["signal_modes"] = results

    # Run RSI Confirmation Mode comparisons
    if run_rsi_confirm:
        print("\n--- Running RSI Confirmation Mode Comparisons ---")
        results = []
        for cfg in RSI_CONFIRM_CONFIGS:
            print(f"  Testing: {cfg['name']}...")
            test_config = base_config.copy()
            test_config["rsi_confirmation_mode"] = cfg["rsi_confirmation_mode"]
            metrics = await run_single_backtest(test_config, client)
            results.append({"name": cfg["name"], "metrics": metrics})

        print_comparison_table(results, "RSI Confirmation Mode Comparison")
        all_results["rsi_confirm_modes"] = results

    # Run BB Entry Strategy comparisons
    if run_bb_entry:
        print("\n--- Running BB Entry Strategy Comparisons ---")
        results = []
        for cfg in BB_ENTRY_CONFIGS:
            print(f"  Testing: {cfg['name']}...")
            test_config = base_config.copy()
            test_config["bb_entry_strategy"] = cfg["bb_entry_strategy"]
            metrics = await run_single_backtest(test_config, client)
            results.append({"name": cfg["name"], "metrics": metrics})

        print_comparison_table(results, "BB Entry Strategy Comparison")
        all_results["bb_entry_strategies"] = results

    # Run BB Exit Strategy comparisons
    if run_bb_exit:
        print("\n--- Running BB Exit Strategy Comparisons ---")
        results = []
        for cfg in BB_EXIT_CONFIGS:
            print(f"  Testing: {cfg['name']}...")
            test_config = base_config.copy()
            test_config["bb_exit_strategy"] = cfg["bb_exit_strategy"]
            metrics = await run_single_backtest(test_config, client)
            results.append({"name": cfg["name"], "metrics": metrics})

        print_comparison_table(results, "BB Exit Strategy Comparison")
        all_results["bb_exit_strategies"] = results

    # Run BB Volatility Filter comparisons
    if run_bb_filter:
        print("\n--- Running BB Volatility Filter Comparisons ---")
        results = []
        for cfg in BB_FILTER_CONFIGS:
            print(f"  Testing: {cfg['name']}...")
            test_config = base_config.copy()
            test_config["bb_volatility_filter"] = cfg["bb_volatility_filter"]
            # Need to enable BB entry or signal mode to see filter effect
            test_config["signal_mode"] = "independent"  # To generate BB signals
            metrics = await run_single_backtest(test_config, client)
            results.append({"name": cfg["name"], "metrics": metrics})

        print_comparison_table(results, "BB Volatility Filter Comparison")
        all_results["bb_volatility_filter"] = results

    # Save results to JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump({
                "start_date": str(start_date),
                "end_date": str(end_date),
                "symbol": args.symbol,
                "results": all_results,
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Cleanup
    if theta_process:
        print("\nTerminating ThetaTerminal...")
        theta_process.terminate()

    print("\nComparison complete!")


if __name__ == "__main__":
    asyncio.run(main())
