#!/usr/bin/env python3
"""
Unified strategy runner CLI.

Run individual trading strategies with custom configurations.

Usage:
    # List available strategies
    python scripts/run_strategy.py --list

    # Run morning fade strategy
    python scripts/run_strategy.py --strategy morning_fade --start 2024-01-01 --end 2024-12-01

    # Run IV rank strategy with custom threshold
    python scripts/run_strategy.py --strategy iv_rank --start 2024-01-01 --end 2024-12-01 --min-iv-rank 50

    # Save results to file
    python scripts/run_strategy.py --strategy morning_fade --start 2024-01-01 --end 2024-12-01 --save
"""

import argparse
import asyncio
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.registry import StrategyRegistry
from strategies.comparison import ResultsManager, StrategyResult, print_comparison_table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run trading strategy backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all available strategies
    python scripts/run_strategy.py --list

    # Run morning fade strategy
    python scripts/run_strategy.py --strategy morning_fade --start 2024-01-01 --end 2024-12-01

    # Run IV rank strategy
    python scripts/run_strategy.py --strategy iv_rank --start 2024-01-01 --end 2024-12-01 --min-iv-rank 50

    # Save and compare results
    python scripts/run_strategy.py --strategy morning_fade --start 2024-01-01 --end 2024-12-01 --save --compare
        """,
    )

    # Strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy name to run (use --list to see available)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available strategies and exit",
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )

    # Common settings
    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Symbol to trade (default: SPY)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)",
    )

    # IV Rank strategy specific
    parser.add_argument(
        "--min-iv-rank",
        type=float,
        default=50.0,
        help="IV rank minimum threshold (default: 50)",
    )
    parser.add_argument(
        "--iv-lookback",
        type=int,
        default=45,
        help="IV rank lookback days (default: 45)",
    )

    # Morning fade strategy specific
    parser.add_argument(
        "--fade-threshold",
        type=float,
        default=0.3,
        help="Morning fade threshold percent (default: 0.3)",
    )
    parser.add_argument(
        "--spread-width",
        type=float,
        default=3.0,
        help="Credit spread width in dollars (default: 3.0)",
    )

    # Output options
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with other strategies (requires --save)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output",
    )

    return parser.parse_args()


def list_strategies():
    """List all available strategies."""
    print("\nAvailable Strategies:")
    print("-" * 60)

    strategies = StrategyRegistry.get_strategy_info()

    if not strategies:
        print("  No strategies found.")
        print("  Make sure strategy modules are in the strategies/ directory.")
        return

    for info in strategies:
        print(f"  {info['name']:<20} - {info['description']}")

    print()


async def run_strategy(args):
    """Run the selected strategy."""
    # Get strategy
    strategy = StrategyRegistry.get_strategy(args.strategy)
    if not strategy:
        print(f"Error: Unknown strategy '{args.strategy}'")
        print(f"Available strategies: {', '.join(StrategyRegistry.list_strategies())}")
        return None

    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    # Get and customize config
    config = strategy.get_default_config()
    config.symbol = args.symbol
    config.initial_capital = args.capital

    # Apply strategy-specific settings
    if args.strategy == "iv_rank":
        config.min_iv_rank = args.min_iv_rank
        config.iv_rank_lookback_days = args.iv_lookback

    elif args.strategy == "morning_fade":
        config.fade_threshold_pct = args.fade_threshold
        config.spread_width = args.spread_width

    # Print header
    if not args.quiet:
        print("\n" + "=" * 60)
        print(f" Strategy: {strategy.name}")
        print(f" Description: {strategy.description}")
        print(f" Period: {start_date} to {end_date}")
        print(f" Symbol: {config.symbol}")
        print("=" * 60)

    # Progress callback
    def progress_callback(current, total):
        if not args.quiet and total > 0:
            pct = current / total * 100
            print(f"\rProgress: {pct:.1f}% ({current}/{total})", end="", flush=True)

    # Run backtest
    try:
        result = await strategy.run_backtest(
            start_date=start_date,
            end_date=end_date,
            config=config,
            progress_callback=progress_callback if args.verbose else None,
        )
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None

    if args.verbose:
        print()  # New line after progress

    return result, config


def print_results(strategy_name: str, result, config):
    """Print backtest results."""
    metrics = result.metrics

    print(f"\nResults for {strategy_name}:")
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


async def main():
    """Main entry point."""
    args = parse_args()

    # List strategies
    if args.list:
        list_strategies()
        return

    # Validate required args
    if not args.strategy:
        print("Error: --strategy is required (use --list to see available)")
        sys.exit(1)

    if not args.start or not args.end:
        print("Error: --start and --end are required")
        sys.exit(1)

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Run strategy
    run_result = await run_strategy(args)

    if run_result is None:
        sys.exit(1)

    result, config = run_result

    # Print results
    if not args.quiet:
        print_results(args.strategy, result, config)

    # Save results
    if args.save:
        manager = ResultsManager()

        strategy_result = StrategyResult.from_backtest_result(
            strategy_name=args.strategy,
            start_date=datetime.strptime(args.start, "%Y-%m-%d").date(),
            end_date=datetime.strptime(args.end, "%Y-%m-%d").date(),
            config=config.to_dict(),
            result=result,
        )

        filepath = manager.save_result(strategy_result)
        print(f"\nResults saved to: {filepath}")

    # Compare with other strategies
    if args.compare and args.save:
        manager = ResultsManager()
        comparison = manager.compare_strategies(args.start, args.end)

        if len(comparison) > 1:
            print("\n")
            print_comparison_table(comparison)
        else:
            print("\nNo other strategies to compare. Run more strategies with --save first.")


if __name__ == "__main__":
    asyncio.run(main())
