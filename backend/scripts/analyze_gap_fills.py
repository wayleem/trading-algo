#!/usr/bin/env python3
"""
Gap Fill Diagnostic - Does the thesis actually work?

Answers: Do overnight gaps in SPY actually fill during the trading day?

This is THESIS VALIDATION, not strategy backtesting.
We're checking the underlying price action, not option P&L.
"""

import asyncio
import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.alpaca_client import AlpacaClient


@dataclass
class GapDay:
    """Data for a single gap day."""
    date: date
    prior_close: float
    open_price: float
    gap_pct: float
    day_high: float
    day_low: float
    close_price: float

    # Fill analysis
    gap_filled: bool = False  # Did price reach prior close?
    fill_pct: float = 0.0  # How much of gap was retraced?
    fill_time: Optional[datetime] = None  # When did fill occur?
    time_to_fill_minutes: Optional[int] = None


def analyze_gap_fill(gap: GapDay) -> GapDay:
    """
    Analyze if and how much a gap filled.

    Gap UP (open > prior_close): Fill = price came back DOWN to prior_close
    Gap DOWN (open < prior_close): Fill = price came back UP to prior_close
    """
    gap_size = gap.open_price - gap.prior_close  # Positive = gap up, negative = gap down

    if gap_size > 0:  # Gap UP
        # Did price come back down to prior close?
        gap.gap_filled = gap.day_low <= gap.prior_close

        # How much did it retrace? (0% = no retrace, 100% = full fill)
        max_retrace = gap.open_price - gap.day_low
        gap.fill_pct = min(100, (max_retrace / gap_size) * 100) if gap_size > 0 else 0

    else:  # Gap DOWN (gap_size < 0)
        # Did price come back up to prior close?
        gap.gap_filled = gap.day_high >= gap.prior_close

        # How much did it retrace?
        gap_size_abs = abs(gap_size)
        max_retrace = gap.day_high - gap.open_price
        gap.fill_pct = min(100, (max_retrace / gap_size_abs) * 100) if gap_size_abs > 0 else 0

    return gap


async def find_fill_time(
    client: AlpacaClient,
    gap: GapDay,
    bars_1min: List[dict],
) -> GapDay:
    """Find when the gap filled (if it did)."""
    if not gap.gap_filled:
        return gap

    gap_size = gap.open_price - gap.prior_close

    for bar in bars_1min:
        bar_time = bar["timestamp"]

        if gap_size > 0:  # Gap UP - looking for price to come DOWN
            if bar["low"] <= gap.prior_close:
                gap.fill_time = bar_time
                # Calculate minutes from market open (9:30 AM ET = 14:30 UTC)
                market_open = bar_time.replace(hour=14, minute=30, second=0, microsecond=0)
                gap.time_to_fill_minutes = int((bar_time - market_open).total_seconds() / 60)
                break
        else:  # Gap DOWN - looking for price to come UP
            if bar["high"] >= gap.prior_close:
                gap.fill_time = bar_time
                market_open = bar_time.replace(hour=14, minute=30, second=0, microsecond=0)
                gap.time_to_fill_minutes = int((bar_time - market_open).total_seconds() / 60)
                break

    return gap


async def analyze_gaps(
    start_date: date,
    end_date: date,
    gap_threshold_pct: float = 0.3,
    max_gap_pct: float = 1.5,
) -> List[GapDay]:
    """
    Analyze all gaps in the date range.

    Returns list of GapDay objects with fill analysis.
    """
    client = AlpacaClient()

    print(f"Fetching daily bars for {start_date} to {end_date}...")

    # Fetch daily bars for gap detection
    daily_bars = await client.get_stock_bars(
        symbol="SPY",
        timeframe="1Day",
        start=datetime.combine(start_date, datetime.min.time()),
        end=datetime.combine(end_date, datetime.max.time()),
        limit=10000,
    )

    print(f"Found {len(daily_bars)} trading days")

    gap_days = []
    prior_close = None

    for bar in daily_bars:
        bar_date = bar["timestamp"].date() if hasattr(bar["timestamp"], "date") else bar["timestamp"]

        if prior_close is not None:
            gap_pct = (bar["open"] - prior_close) / prior_close * 100

            # Check if tradeable gap
            if gap_threshold_pct <= abs(gap_pct) <= max_gap_pct:
                gap = GapDay(
                    date=bar_date,
                    prior_close=prior_close,
                    open_price=bar["open"],
                    gap_pct=gap_pct,
                    day_high=bar["high"],
                    day_low=bar["low"],
                    close_price=bar["close"],
                )
                gap = analyze_gap_fill(gap)
                gap_days.append(gap)

        prior_close = bar["close"]

    print(f"Found {len(gap_days)} tradeable gaps ({gap_threshold_pct}% - {max_gap_pct}%)")

    # Fetch 1-min bars for timing analysis on filled gaps
    print("\nAnalyzing fill timing for filled gaps...")
    filled_gaps = [g for g in gap_days if g.gap_filled]

    for i, gap in enumerate(filled_gaps):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{len(filled_gaps)}...")

        try:
            bars_1min = await client.get_stock_bars(
                symbol="SPY",
                timeframe="1Min",
                start=datetime.combine(gap.date, time(14, 30)),  # Market open UTC
                end=datetime.combine(gap.date, time(21, 0)),  # Market close UTC
                limit=500,
            )
            gap = await find_fill_time(client, gap, bars_1min)
        except Exception as e:
            print(f"  Error fetching 1min data for {gap.date}: {e}")

    return gap_days


def print_analysis(gap_days: List[GapDay]):
    """Print comprehensive gap fill analysis."""
    if not gap_days:
        print("No gaps found!")
        return

    total = len(gap_days)
    filled = sum(1 for g in gap_days if g.gap_filled)
    partial_50 = sum(1 for g in gap_days if g.fill_pct >= 50)
    partial_75 = sum(1 for g in gap_days if g.fill_pct >= 75)

    # Split by direction
    gap_ups = [g for g in gap_days if g.gap_pct > 0]
    gap_downs = [g for g in gap_days if g.gap_pct < 0]

    gap_ups_filled = sum(1 for g in gap_ups if g.gap_filled)
    gap_downs_filled = sum(1 for g in gap_downs if g.gap_filled)

    print("\n" + "=" * 70)
    print("GAP FILL THESIS VALIDATION")
    print("=" * 70)

    print(f"\nTotal tradeable gaps: {total}")
    print(f"  Gap UPs:   {len(gap_ups)}")
    print(f"  Gap DOWNs: {len(gap_downs)}")

    print("\n" + "-" * 70)
    print("FILL RATES (Did price reach prior close at any point?)")
    print("-" * 70)

    print(f"\n{'Metric':<30} {'Count':>10} {'Rate':>10}")
    print("-" * 50)
    print(f"{'Full gap fill (100%)':<30} {filled:>10} {filled/total*100:>9.1f}%")
    print(f"{'Partial fill (>=75%)':<30} {partial_75:>10} {partial_75/total*100:>9.1f}%")
    print(f"{'Partial fill (>=50%)':<30} {partial_50:>10} {partial_50/total*100:>9.1f}%")

    print("\n" + "-" * 70)
    print("BY DIRECTION")
    print("-" * 70)

    print(f"\n{'Direction':<15} {'Total':>8} {'Filled':>8} {'Fill Rate':>12}")
    print("-" * 45)
    if gap_ups:
        print(f"{'Gap UP':<15} {len(gap_ups):>8} {gap_ups_filled:>8} {gap_ups_filled/len(gap_ups)*100:>11.1f}%")
    if gap_downs:
        print(f"{'Gap DOWN':<15} {len(gap_downs):>8} {gap_downs_filled:>8} {gap_downs_filled/len(gap_downs)*100:>11.1f}%")

    # Time to fill analysis
    filled_gaps = [g for g in gap_days if g.gap_filled and g.time_to_fill_minutes is not None]

    if filled_gaps:
        print("\n" + "-" * 70)
        print("TIME TO FILL (for gaps that filled)")
        print("-" * 70)

        times = [g.time_to_fill_minutes for g in filled_gaps]
        avg_time = sum(times) / len(times)

        # Bucket by time period
        first_30 = sum(1 for t in times if t <= 30)
        first_60 = sum(1 for t in times if t <= 60)
        first_120 = sum(1 for t in times if t <= 120)
        morning = sum(1 for t in times if t <= 180)  # First 3 hours

        print(f"\n{'Time Period':<25} {'Count':>8} {'Cumulative %':>15}")
        print("-" * 50)
        print(f"{'First 30 min':<25} {first_30:>8} {first_30/len(filled_gaps)*100:>14.1f}%")
        print(f"{'First 60 min':<25} {first_60:>8} {first_60/len(filled_gaps)*100:>14.1f}%")
        print(f"{'First 2 hours':<25} {first_120:>8} {first_120/len(filled_gaps)*100:>14.1f}%")
        print(f"{'First 3 hours (AM)':<25} {morning:>8} {morning/len(filled_gaps)*100:>14.1f}%")

        print(f"\nAverage time to fill: {avg_time:.0f} minutes")

    # Year-by-year breakdown
    print("\n" + "-" * 70)
    print("YEAR-BY-YEAR BREAKDOWN")
    print("-" * 70)

    by_year = defaultdict(list)
    for g in gap_days:
        by_year[g.date.year].append(g)

    print(f"\n{'Year':<8} {'Gaps':>8} {'Filled':>8} {'Fill Rate':>12} {'Avg Fill %':>12}")
    print("-" * 50)

    for year in sorted(by_year.keys()):
        year_gaps = by_year[year]
        year_filled = sum(1 for g in year_gaps if g.gap_filled)
        avg_fill_pct = sum(g.fill_pct for g in year_gaps) / len(year_gaps)
        print(f"{year:<8} {len(year_gaps):>8} {year_filled:>8} {year_filled/len(year_gaps)*100:>11.1f}% {avg_fill_pct:>11.1f}%")

    # Decision matrix
    fill_rate = filled / total * 100

    print("\n" + "=" * 70)
    print("THESIS VERDICT")
    print("=" * 70)

    print(f"\nGap fill rate: {fill_rate:.1f}%")

    if fill_rate < 45:
        verdict = "THESIS FLAWED - Gaps don't reliably fill"
        action = "Abandon gap fade strategy"
    elif fill_rate < 55:
        verdict = "WEAK EDGE - Marginal fill rate"
        action = "Not worth pursuing or needs heavy modification"
    elif fill_rate < 65:
        verdict = "THESIS VALID - Execution problem"
        action = "Fix entry/exit/instrument (spreads, timing, stops)"
    else:
        verdict = "STRONG THESIS - Definitely execution problem"
        action = "Fix urgently - edge exists but not captured"

    print(f"\nVerdict: {verdict}")
    print(f"Action:  {action}")

    # If thesis valid, diagnose execution issues
    if fill_rate >= 55:
        print("\n" + "-" * 70)
        print("EXECUTION DIAGNOSIS")
        print("-" * 70)

        if filled_gaps:
            late_fills = sum(1 for g in filled_gaps if g.time_to_fill_minutes and g.time_to_fill_minutes > 120)
            late_pct = late_fills / len(filled_gaps) * 100

            print(f"\nLate fills (>2 hours): {late_pct:.1f}%")
            if late_pct > 40:
                print("  -> Problem: Theta decay kills option before fill")
                print("  -> Fix: Use spreads or reduce profit target")

            early_fills = sum(1 for g in filled_gaps if g.time_to_fill_minutes and g.time_to_fill_minutes <= 60)
            early_pct = early_fills / len(filled_gaps) * 100

            print(f"\nEarly fills (<=1 hour): {early_pct:.1f}%")
            if early_pct > 50:
                print("  -> Good: Many gaps fill quickly")
                print("  -> Current 10% stop may be too tight for morning volatility")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze gap fill rates")
    parser.add_argument("--start", default="2022-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-01", help="End date")
    parser.add_argument("--threshold", type=float, default=0.3, help="Min gap %")
    parser.add_argument("--max-gap", type=float, default=1.5, help="Max gap %")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    gap_days = await analyze_gaps(
        start_date=start,
        end_date=end,
        gap_threshold_pct=args.threshold,
        max_gap_pct=args.max_gap,
    )

    print_analysis(gap_days)


if __name__ == "__main__":
    asyncio.run(main())
