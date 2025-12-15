#!/usr/bin/env python3
"""
Grid search optimization for profit target and stop loss.

Usage:
    python scripts/grid_search_ptsl.py --start 2023-12-01 --end 2025-12-01
    python scripts/grid_search_ptsl.py --start 2023-12-01 --end 2025-12-01 --symbol SPY
"""

import argparse
import asyncio
import itertools
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Parameter grids
PROFIT_TARGETS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
STOP_LOSSES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def run_single_backtest_sync(args_tuple):
    """Run a single backtest synchronously (for multiprocessing)."""
    pt, sl, start_date, end_date, symbol, rsi_oversold, rsi_overbought = args_tuple

    # Run as subprocess to avoid async issues in multiprocessing
    cmd = [
        sys.executable, "scripts/run_backtest.py",
        "--start", start_date,
        "--end", end_date,
        "--symbol", symbol,
        "--profit-target-pct", str(pt),
        "--stop-loss-pct", str(sl),
        "--slippage", "0.01",
        "--timeframe", "parallel",
        "--use-pattern-sizing",
        "--pattern-bonus-contracts", "2",
        "--strike-offset", "0",
        "--rsi-oversold", str(rsi_oversold),
        "--rsi-overbought", str(rsi_overbought),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(__file__).parent.parent)
        )

        output = result.stdout

        # Parse metrics from output
        sharpe = 0.0
        total_return = 0.0
        win_rate = 0.0
        drawdown = 0.0
        profit_factor = 0.0

        for line in output.split('\n'):
            if 'Sharpe Ratio:' in line:
                try:
                    sharpe = float(line.split(':')[1].strip())
                except:
                    pass
            elif 'Total Return:' in line:
                try:
                    total_return = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            elif 'Win Rate:' in line and 'Entry' not in line:
                try:
                    win_rate = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            elif 'Max Drawdown:' in line:
                try:
                    drawdown = float(line.split(':')[1].strip().replace('%', ''))
                except:
                    pass
            elif 'Profit Factor:' in line:
                try:
                    profit_factor = float(line.split(':')[1].strip())
                except:
                    pass

        return {
            'pt': pt,
            'sl': sl,
            'sharpe': sharpe,
            'return': total_return,
            'win_rate': win_rate,
            'drawdown': drawdown,
            'profit_factor': profit_factor,
            'success': True
        }

    except Exception as e:
        return {
            'pt': pt,
            'sl': sl,
            'sharpe': 0.0,
            'return': 0.0,
            'win_rate': 0.0,
            'drawdown': 0.0,
            'profit_factor': 0.0,
            'success': False,
            'error': str(e)
        }


def plot_heatmap(results, metric='sharpe', title_suffix=''):
    """Generate a heatmap of results."""
    # Create matrix
    pt_vals = sorted(set(r['pt'] for r in results))
    sl_vals = sorted(set(r['sl'] for r in results))

    matrix = np.zeros((len(sl_vals), len(pt_vals)))

    for r in results:
        pt_idx = pt_vals.index(r['pt'])
        sl_idx = sl_vals.index(r['sl'])
        matrix[sl_idx, pt_idx] = r[metric]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        xticklabels=[f'{p:.0%}' for p in pt_vals],
        yticklabels=[f'{s:.0%}' for s in sl_vals],
        cmap='RdYlGn',
        ax=ax
    )

    ax.set_xlabel('Profit Target')
    ax.set_ylabel('Stop Loss')
    ax.set_title(f'{metric.title()} by PT/SL Combination {title_suffix}')

    plt.tight_layout()

    # Save
    filename = f'grid_search_{metric}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=150)
    print(f"Saved heatmap to {filename}")

    return fig


def run_grid_search(start_date, end_date, symbol='SPX', rsi_oversold=40, rsi_overbought=60, n_jobs=4):
    """Run grid search over all PT/SL combinations."""
    print("=" * 60)
    print("GRID SEARCH: Profit Target / Stop Loss Optimization")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"RSI Thresholds: {rsi_oversold}/{rsi_overbought}")
    print(f"Profit Targets: {[f'{p:.0%}' for p in PROFIT_TARGETS]}")
    print(f"Stop Losses: {[f'{s:.0%}' for s in STOP_LOSSES]}")
    print(f"Total combinations: {len(PROFIT_TARGETS) * len(STOP_LOSSES)}")
    print(f"Parallel workers: {n_jobs}")
    print("=" * 60)
    print()

    # Create all parameter combinations
    combinations = list(itertools.product(PROFIT_TARGETS, STOP_LOSSES))
    args_list = [
        (pt, sl, start_date, end_date, symbol, rsi_oversold, rsi_overbought)
        for pt, sl in combinations
    ]

    results = []
    start_time = time.time()

    # Run sequentially for now (to avoid ThetaTerminal connection issues)
    # Can enable parallel later with proper connection pooling
    for i, args in enumerate(args_list):
        pt, sl = args[0], args[1]
        print(f"[{i+1}/{len(args_list)}] Testing PT={pt:.0%}, SL={sl:.0%}...", end=' ', flush=True)

        result = run_single_backtest_sync(args)
        results.append(result)

        if result['success']:
            print(f"Sharpe={result['sharpe']:.2f}, Return={result['return']:.1f}%, WR={result['win_rate']:.1f}%")
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")

    elapsed = time.time() - start_time
    print()
    print(f"Completed {len(results)} backtests in {elapsed:.1f} seconds")
    print()

    # Find best results
    successful = [r for r in results if r['success'] and r['sharpe'] > 0]

    if successful:
        best_sharpe = max(successful, key=lambda x: x['sharpe'])
        best_return = max(successful, key=lambda x: x['return'])
        best_pf = max(successful, key=lambda x: x['profit_factor'])

        print("=" * 60)
        print("BEST RESULTS")
        print("=" * 60)
        print(f"Best Sharpe: PT={best_sharpe['pt']:.0%}, SL={best_sharpe['sl']:.0%}")
        print(f"  -> Sharpe={best_sharpe['sharpe']:.2f}, Return={best_sharpe['return']:.1f}%, PF={best_sharpe['profit_factor']:.2f}")
        print()
        print(f"Best Return: PT={best_return['pt']:.0%}, SL={best_return['sl']:.0%}")
        print(f"  -> Return={best_return['return']:.1f}%, Sharpe={best_return['sharpe']:.2f}, PF={best_return['profit_factor']:.2f}")
        print()
        print(f"Best Profit Factor: PT={best_pf['pt']:.0%}, SL={best_pf['sl']:.0%}")
        print(f"  -> PF={best_pf['profit_factor']:.2f}, Sharpe={best_pf['sharpe']:.2f}, Return={best_pf['return']:.1f}%")
        print("=" * 60)

        # Generate heatmaps
        plot_heatmap(successful, 'sharpe', f'({symbol})')
        plot_heatmap(successful, 'return', f'({symbol})')
        plot_heatmap(successful, 'profit_factor', f'({symbol})')

    else:
        print("No successful backtests!")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search optimization for PT/SL")

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
        "--rsi-oversold",
        type=float,
        default=40.0,
        help="RSI oversold threshold (default: 40)",
    )
    parser.add_argument(
        "--rsi-overbought",
        type=float,
        default=60.0,
        help="RSI overbought threshold (default: 60)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, sequential)",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    results = run_grid_search(
        start_date=args.start,
        end_date=args.end,
        symbol=args.symbol,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        n_jobs=args.jobs
    )
