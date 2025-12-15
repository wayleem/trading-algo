#!/usr/bin/env python3
"""
CLI script to run backtest for modular strategies (VWAP fade, etc.).

Usage:
    python run_strategy_backtest.py --strategy vwap_fade --start 2024-01-01 --end 2024-12-01
    python run_strategy_backtest.py --strategy vwap_fade --start 2022-01-01 --end 2024-12-01
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.registry import StrategyRegistry


def parse_args():
    parser = argparse.ArgumentParser(description="Run strategy backtest")

    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy name (e.g., vwap_fade, orb)",
    )
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
        "--dte",
        type=int,
        default=0,
        help="Days to expiration (0=0DTE, 1=1DTE, etc.)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available strategies and exit",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # List strategies if requested
    if args.list:
        print("Available strategies:")
        for info in StrategyRegistry.get_strategy_info():
            print(f"  {info['name']}: {info['description']}")
        return

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        print("Use format: YYYY-MM-DD")
        sys.exit(1)

    # Get strategy
    strategy = StrategyRegistry.get_strategy(args.strategy)
    if strategy is None:
        print(f"Strategy '{args.strategy}' not found.")
        print("Available strategies:")
        for name in StrategyRegistry.list_strategies():
            print(f"  {name}")
        sys.exit(1)

    # Print header
    print("=" * 60)
    print(f"Strategy: {strategy.name}")
    print(f"Description: {strategy.description}")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 60)
    print()

    # Get default config
    config = strategy.get_default_config()

    # Override days_to_expiration if specified
    if args.dte != 0:
        config.days_to_expiration = args.dte

    # Print config highlights
    print("Configuration:")
    print(f"  Days to Expiration: {config.days_to_expiration} ({'0DTE' if config.days_to_expiration == 0 else f'{config.days_to_expiration}DTE'})")
    if hasattr(config, 'deviation_threshold_pct'):
        print(f"  VWAP Deviation: {config.deviation_threshold_pct}%")
    if hasattr(config, 'fade_below_vwap'):
        print(f"  Fade Below VWAP: {config.fade_below_vwap}")
        print(f"  Fade Above VWAP: {config.fade_above_vwap}")
    if hasattr(config, 'use_vwap_touch_exit'):
        print(f"  VWAP Touch Exit: {config.use_vwap_touch_exit}")
    if hasattr(config, 'profit_target_reversion_pct'):
        print(f"  Reversion Target: {config.profit_target_reversion_pct}%")
    if hasattr(config, 'exit_deadline_hour_utc'):
        deadline_hour = config.exit_deadline_hour_utc
        deadline_min = getattr(config, 'exit_deadline_minute_utc', 0)
        # Convert UTC to ET (rough)
        et_hour = deadline_hour - 5
        print(f"  Exit Deadline: {et_hour}:{deadline_min:02d} AM ET")
    print(f"  Stop Loss: {config.stop_loss_pct:.0%}")
    print(f"  Initial Capital: ${config.initial_capital:,.2f}")
    print()

    # Run backtest with progress
    print("Running backtest...")

    def progress_callback(current, total):
        pct = current / total * 100 if total > 0 else 0
        print(f"\rProgress: {pct:.1f}% ({current}/{total} bars)", end="", flush=True)

    result = await strategy.run_backtest(start_date, end_date, config, progress_callback)
    print("\n")

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    metrics = result.metrics
    print(f"Results for {strategy.name}:")
    print("-" * 40)
    print(f"  Total Trades:     {metrics.total_trades}")
    print(f"  Winning Trades:   {metrics.winning_trades}")
    print(f"  Losing Trades:    {metrics.losing_trades}")
    print(f"  Win Rate:         {metrics.win_rate:.1%}")
    print()
    print(f"  Total P&L:        ${metrics.total_pnl:,.2f}")
    print(f"  Average P&L:      ${metrics.avg_pnl:,.2f}")
    print(f"  Max Drawdown:     {metrics.max_drawdown:.1%}")
    print()
    print(f"  Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
    print(f"  Profit Factor:    {metrics.profit_factor:.2f}")
    print("-" * 40)

    # Final equity
    if result.equity_curve:
        final_equity = result.equity_curve[-1][1]
        total_return = (final_equity - config.initial_capital) / config.initial_capital
        print()
        print(f"  Final Equity:     ${final_equity:,.2f}")
        print(f"  Total Return:     {total_return:.1%}")

    # Trade breakdown by exit reason
    if result.trades:
        print()
        print("Trade Breakdown by Exit Reason:")
        reasons = {}
        for trade in result.trades:
            reason = trade.exit_reason
            if reason not in reasons:
                reasons[reason] = {"count": 0, "pnl": 0}
            reasons[reason]["count"] += 1
            reasons[reason]["pnl"] += trade.pnl_dollars

        for reason, data in sorted(reasons.items()):
            win_rate = sum(1 for t in result.trades if t.exit_reason == reason and t.pnl_dollars > 0) / data["count"] * 100
            print(f"  {reason}: {data['count']} trades, {win_rate:.1f}% win rate, ${data['pnl']:,.2f} P&L")

    print()
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
