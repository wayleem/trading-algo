#!/usr/bin/env python3
"""
Analyze ORB backtest results segmented by VIX regime.

Determines if VIX filtering improves ORB performance.

Decision matrix:
- Sharpe consistent across regimes → No filter needed
- High VIX Sharpe < 1.0, others > 2.0 → Add VIX < 25 filter
- Low VIX dramatically better → Add VIX < 15 filter
- Mixed/unclear → No filter (insufficient signal)
"""

import asyncio
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.alpaca_client import AlpacaClient
from strategies.registry import StrategyRegistry


async def fetch_vix_data(start_date: date, end_date: date) -> dict:
    """Fetch daily VIX index data from CBOE (free, official source)."""
    import httpx

    vix_data = {}

    # Use CBOE direct download (free, official VIX source)
    print("Fetching VIX index from CBOE...")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
            )
            response.raise_for_status()

            # Parse CSV response (DATE,OPEN,HIGH,LOW,CLOSE format: MM/DD/YYYY)
            lines = response.text.strip().split("\n")
            for line in lines[1:]:  # Skip header
                try:
                    parts = line.split(",")
                    bar_date = datetime.strptime(parts[0], "%m/%d/%Y").date()

                    # Only include dates in our range
                    if start_date <= bar_date <= end_date:
                        close = float(parts[4])  # CLOSE price
                        vix_data[bar_date] = close
                except (ValueError, IndexError):
                    continue

            print(f"Fetched {len(vix_data)} days of VIX index from CBOE")

    except Exception as e:
        print(f"CBOE VIX fetch failed: {e}")
        raise RuntimeError("Cannot run VIX analysis without real VIX data")

    return vix_data


def classify_vix_regime(vix_level: float) -> str:
    """Classify VIX into regime."""
    if vix_level < 15:
        return "low"
    elif vix_level < 25:
        return "mid"
    else:
        return "high"


def calculate_regime_metrics(trades: list, initial_capital: float = 10000) -> dict:
    """Calculate metrics for a set of trades."""
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "sharpe": 0.0,
            "profit_factor": 0.0,
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_pnl = total_pnl / len(pnls) if pnls else 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Sharpe (daily returns)
    daily_pnl = defaultdict(float)
    for t in trades:
        trade_date = t["exit_date"].date() if hasattr(t["exit_date"], "date") else t["exit_date"]
        daily_pnl[trade_date] += t["pnl"]

    if len(daily_pnl) > 1:
        daily_returns = [pnl / initial_capital for pnl in daily_pnl.values()]
        avg_ret = statistics.mean(daily_returns)
        std_ret = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0.001
        sharpe = (avg_ret / std_ret) * (252 ** 0.5) if std_ret > 0 else 0
    else:
        sharpe = 0

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
    }


async def run_orb_backtest(start_date: date, end_date: date) -> list:
    """Run ORB backtest and return trade list using strategies framework."""
    # Use the strategies framework (same as validated results)
    strategy = StrategyRegistry.get_strategy("orb")
    if strategy is None:
        raise ValueError("ORB strategy not found in registry")

    # Get default config (has validated parameters)
    config = strategy.get_default_config()

    # Run backtest
    result = await strategy.run_backtest(start_date, end_date, config)

    # Convert to simple dict format
    trades = []
    for trade in result.trades:
        trades.append({
            "entry_date": trade.entry_date,
            "exit_date": trade.exit_date,
            "pnl": trade.pnl_dollars,
            "signal_type": trade.signal_type,
            "exit_reason": trade.exit_reason,
        })

    return trades


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-01")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    print("=" * 70)
    print("ORB BACKTEST BY VIX REGIME")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 70)
    print()

    # Step 1: Fetch VIX data
    print("Fetching VIX data...")
    vix_data = await fetch_vix_data(start_date, end_date)

    if not vix_data:
        print("ERROR: Could not fetch VIX data")
        return

    # Print VIX distribution
    vix_values = list(vix_data.values())
    print(f"VIX range: {min(vix_values):.1f} - {max(vix_values):.1f}")
    print(f"VIX mean: {statistics.mean(vix_values):.1f}")
    print()

    # Step 2: Run ORB backtest
    print("Running ORB backtest...")
    trades = await run_orb_backtest(start_date, end_date)
    print(f"Total trades: {len(trades)}")
    print()

    # Step 3: Segment by VIX regime
    regime_trades = {"low": [], "mid": [], "high": []}
    unmatched = 0

    for trade in trades:
        trade_date = trade["entry_date"].date() if hasattr(trade["entry_date"], "date") else trade["entry_date"]

        # Get VIX for trade date (or nearest available)
        vix_level = vix_data.get(trade_date)

        if vix_level is None:
            # Try day before
            for offset in range(1, 5):
                prev_date = trade_date - timedelta(days=offset)
                vix_level = vix_data.get(prev_date)
                if vix_level:
                    break

        if vix_level is not None:
            regime = classify_vix_regime(vix_level)
            trade["vix"] = vix_level
            regime_trades[regime].append(trade)
        else:
            unmatched += 1

    if unmatched > 0:
        print(f"Warning: {unmatched} trades could not be matched to VIX data")
    print()

    # Step 4: Calculate metrics by regime
    print("=" * 70)
    print("RESULTS BY VIX REGIME")
    print("=" * 70)
    print()

    print(f"{'Regime':<12} {'VIX':<12} {'Trades':>8} {'Win%':>8} {'Sharpe':>8} {'PF':>8} {'Total P&L':>12}")
    print("-" * 70)

    all_metrics = {}
    for regime, vix_range in [("low", "<15"), ("mid", "15-25"), ("high", ">25")]:
        metrics = calculate_regime_metrics(regime_trades[regime])
        all_metrics[regime] = metrics

        print(f"{regime.upper():<12} {vix_range:<12} {metrics['trades']:>8} {metrics['win_rate']*100:>7.1f}% {metrics['sharpe']:>8.2f} {metrics['profit_factor']:>8.2f} ${metrics['total_pnl']:>11,.2f}")

    # Overall metrics
    all_trades = regime_trades["low"] + regime_trades["mid"] + regime_trades["high"]
    overall = calculate_regime_metrics(all_trades)
    print("-" * 70)
    print(f"{'OVERALL':<12} {'all':<12} {overall['trades']:>8} {overall['win_rate']*100:>7.1f}% {overall['sharpe']:>8.2f} {overall['profit_factor']:>8.2f} ${overall['total_pnl']:>11,.2f}")

    # Step 5: Decision
    print()
    print("=" * 70)
    print("DECISION")
    print("=" * 70)
    print()

    low_sharpe = all_metrics["low"]["sharpe"]
    mid_sharpe = all_metrics["mid"]["sharpe"]
    high_sharpe = all_metrics["high"]["sharpe"]

    # Analyze patterns
    if all_metrics["low"]["trades"] < 5 or all_metrics["mid"]["trades"] < 5 or all_metrics["high"]["trades"] < 5:
        print("INSUFFICIENT DATA: Not enough trades in all regimes for reliable analysis")
        print("Recommendation: No filter - need more data")

    elif high_sharpe < 1.0 and (low_sharpe > 2.0 or mid_sharpe > 2.0):
        print(f"HIGH VIX UNDERPERFORMS: High VIX Sharpe ({high_sharpe:.2f}) < 1.0")
        print(f"Low/Mid VIX Sharpe: {low_sharpe:.2f} / {mid_sharpe:.2f}")
        print("Recommendation: ADD VIX < 25 FILTER")

    elif low_sharpe > mid_sharpe * 1.5 and low_sharpe > high_sharpe * 1.5:
        print(f"LOW VIX DRAMATICALLY BETTER: Sharpe {low_sharpe:.2f} vs {mid_sharpe:.2f}/{high_sharpe:.2f}")
        print("Recommendation: ADD VIX < 15 FILTER or SIZE BY REGIME")

    elif abs(low_sharpe - mid_sharpe) < 0.5 and abs(mid_sharpe - high_sharpe) < 0.5:
        print(f"SHARPE CONSISTENT: {low_sharpe:.2f} / {mid_sharpe:.2f} / {high_sharpe:.2f}")
        print("Recommendation: NO FILTER NEEDED - paper trade as-is")

    else:
        print(f"MIXED PATTERN: {low_sharpe:.2f} / {mid_sharpe:.2f} / {high_sharpe:.2f}")
        print("Recommendation: NO FILTER - insufficient signal for filtering")

    print()


if __name__ == "__main__":
    asyncio.run(main())
