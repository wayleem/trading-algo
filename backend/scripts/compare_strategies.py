#!/usr/bin/env python3
"""
Compare strategy results across multiple strategies.

Usage:
    # Compare all strategies for a date range
    python scripts/compare_strategies.py --start 2024-01-01 --end 2024-12-01

    # Compare specific strategies
    python scripts/compare_strategies.py --start 2024-01-01 --end 2024-12-01 --strategies morning_fade iv_rank

    # Output as markdown
    python scripts/compare_strategies.py --start 2024-01-01 --end 2024-12-01 --format markdown
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.registry import StrategyRegistry
from strategies.comparison import (
    ResultsManager,
    print_comparison_table,
    format_comparison_markdown,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare trading strategy results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--strategies",
        type=str,
        nargs="+",
        help="Specific strategies to compare (default: all)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "markdown", "json"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--list-results",
        action="store_true",
        help="List all saved results and exit",
    )

    return parser.parse_args()


def list_saved_results():
    """List all saved strategy results."""
    manager = ResultsManager()
    results = manager.list_results()

    if not results:
        print("No saved results found.")
        print("Run strategies with --save flag first:")
        print("  python scripts/run_strategy.py --strategy morning_fade --start 2024-01-01 --end 2024-12-01 --save")
        return

    print("\nSaved Strategy Results:")
    print("-" * 80)

    # Group by strategy
    by_strategy = {}
    for r in results:
        if r.strategy_name not in by_strategy:
            by_strategy[r.strategy_name] = []
        by_strategy[r.strategy_name].append(r)

    for strategy, strategy_results in sorted(by_strategy.items()):
        print(f"\n{strategy}:")
        for r in sorted(strategy_results, key=lambda x: x.start_date, reverse=True):
            print(
                f"  {r.start_date} to {r.end_date}: "
                f"{r.total_trades} trades, "
                f"${r.total_pnl:,.2f} P&L, "
                f"{r.win_rate:.1%} win rate"
            )


def main():
    """Main entry point."""
    args = parse_args()

    # List results
    if args.list_results:
        list_saved_results()
        return

    # Load results
    manager = ResultsManager()
    results = manager.compare_strategies(
        start_date=args.start,
        end_date=args.end,
        strategy_names=args.strategies,
    )

    if not results:
        print(f"No results found for period {args.start} to {args.end}")
        print("\nAvailable results:")
        list_saved_results()
        return

    # Output results
    if args.format == "table":
        print_comparison_table(results)

    elif args.format == "markdown":
        print(format_comparison_markdown(results))

    elif args.format == "json":
        import json

        output = {
            name: r.to_dict()
            for name, r in results.items()
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
