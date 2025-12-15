#!/usr/bin/env python3
"""
VWAP Morning Window Year-by-Year Breakdown.

Does the morning edge hold stable across years, or is it declining like overall?
"""

import asyncio
import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.alpaca_client import AlpacaClient


def calculate_vwap(bars: List[dict]) -> List[float]:
    """Calculate running VWAP from bars."""
    vwap_values = []
    cumulative_pv = 0.0
    cumulative_volume = 0.0

    for bar in bars:
        typical_price = (bar["high"] + bar["low"] + bar["close"]) / 3
        volume = bar.get("volume", 0)

        if volume > 0:
            cumulative_pv += typical_price * volume
            cumulative_volume += volume
            vwap = cumulative_pv / cumulative_volume
        else:
            vwap = vwap_values[-1] if vwap_values else typical_price

        vwap_values.append(vwap)

    return vwap_values


def get_time_window_detailed(bar_time: datetime) -> str:
    """Categorize time into detailed trading windows (UTC)."""
    hour = bar_time.hour
    minute = bar_time.minute

    # Convert to minutes since midnight UTC
    mins = hour * 60 + minute

    # Market opens at 14:30 UTC (9:30 AM ET)
    # 9:30-10:00 AM ET = 14:30-15:00 UTC
    if mins < 15 * 60:  # Before 15:00 UTC
        return "early_morning"  # 9:30-10:00 AM ET
    # 10:00-10:30 AM ET = 15:00-15:30 UTC
    elif mins < 15 * 60 + 30:
        return "mid_morning"  # 10:00-10:30 AM ET
    # 10:30-11:00 AM ET = 15:30-16:00 UTC
    elif mins < 16 * 60:
        return "late_morning"  # 10:30-11:00 AM ET
    # 11:00-11:30 AM ET = 16:00-16:30 UTC
    elif mins < 16 * 60 + 30:
        return "pre_midday"  # 11:00-11:30 AM ET
    # 11:30 AM - 1:30 PM ET = 16:30-18:30 UTC
    elif mins < 18 * 60 + 30:
        return "midday"
    # 1:30-3:30 PM ET = 18:30-20:30 UTC
    elif mins < 20 * 60 + 30:
        return "afternoon"
    else:
        return "close"


def is_morning_window(bar_time: datetime) -> bool:
    """Check if bar is in morning window (9:30-11:30 AM ET = 14:30-16:30 UTC)."""
    hour = bar_time.hour
    minute = bar_time.minute
    mins = hour * 60 + minute

    # 9:30-11:30 AM ET = 14:30-16:30 UTC = 870-990 minutes
    return 14 * 60 + 30 <= mins < 16 * 60 + 30


@dataclass
class Deviation:
    date: date
    time: datetime
    deviation_pct: float
    reverted: bool
    time_window: str


async def analyze_morning_by_year(
    start_date: date,
    end_date: date,
    threshold: float = 0.3,
) -> Dict:
    """Analyze morning window reversion rates by year."""
    client = AlpacaClient()

    print(f"Analyzing VWAP morning window from {start_date} to {end_date}")
    print(f"Threshold: {threshold}%")
    print(f"Morning window: 9:30-11:30 AM ET")

    all_deviations = []
    current_date = start_date
    total_days = 0

    while current_date <= end_date:
        try:
            bars = await client.get_stock_bars(
                symbol="SPY",
                timeframe="1Min",
                start=datetime.combine(current_date, time(14, 30)),
                end=datetime.combine(current_date, time(21, 0)),
                limit=500,
            )

            if len(bars) > 30:
                total_days += 1
                vwap_values = calculate_vwap(bars)

                in_deviation = False
                current_dev = None

                for i, (bar, vwap) in enumerate(zip(bars, vwap_values)):
                    if vwap <= 0:
                        continue

                    price = bar["close"]
                    bar_time = bar["timestamp"]
                    deviation_pct = (price - vwap) / vwap * 100

                    # New deviation detected
                    if abs(deviation_pct) >= threshold and not in_deviation:
                        in_deviation = True
                        current_dev = Deviation(
                            date=current_date,
                            time=bar_time,
                            deviation_pct=deviation_pct,
                            reverted=False,
                            time_window=get_time_window_detailed(bar_time),
                        )
                        all_deviations.append(current_dev)

                    # Check for reversion
                    if current_dev and not current_dev.reverted:
                        if current_dev.deviation_pct > 0 and price <= vwap:
                            current_dev.reverted = True
                            in_deviation = False
                            current_dev = None
                        elif current_dev.deviation_pct < 0 and price >= vwap:
                            current_dev.reverted = True
                            in_deviation = False
                            current_dev = None

                if total_days % 100 == 0:
                    print(f"  Processed {total_days} days...")

        except Exception as e:
            pass

        current_date += timedelta(days=1)

    print(f"Processed {total_days} trading days")
    print(f"Total deviations: {len(all_deviations)}")

    return all_deviations


def print_morning_analysis(deviations: List[Deviation]):
    """Print morning-specific year-by-year breakdown."""

    # Filter to morning window only
    morning_devs = [d for d in deviations if is_morning_window(d.time)]

    # Also get non-morning for comparison
    other_devs = [d for d in deviations if not is_morning_window(d.time)]

    print("\n" + "=" * 70)
    print("MORNING WINDOW YEAR-BY-YEAR BREAKDOWN")
    print("Window: 9:30-11:30 AM ET")
    print("=" * 70)

    # Overall morning vs other
    morning_total = len(morning_devs)
    morning_reverted = sum(1 for d in morning_devs if d.reverted)
    other_total = len(other_devs)
    other_reverted = sum(1 for d in other_devs if d.reverted)

    print(f"\n{'Window':<20} {'Total':>10} {'Reverted':>10} {'Rate':>10}")
    print("-" * 50)
    if morning_total > 0:
        print(f"{'Morning (9:30-11:30)':<20} {morning_total:>10} {morning_reverted:>10} {morning_reverted/morning_total*100:>9.1f}%")
    if other_total > 0:
        print(f"{'Rest of Day':<20} {other_total:>10} {other_reverted:>10} {other_reverted/other_total*100:>9.1f}%")

    # Year-by-year for MORNING ONLY
    print("\n" + "-" * 70)
    print("MORNING WINDOW BY YEAR (This is the critical data)")
    print("-" * 70)

    morning_by_year = defaultdict(list)
    for d in morning_devs:
        morning_by_year[d.date.year].append(d)

    print(f"\n{'Year':<8} {'Total':>10} {'Reverted':>10} {'Rate':>10} {'Trend':>15}")
    print("-" * 55)

    prev_rate = None
    for year in sorted(morning_by_year.keys()):
        year_devs = morning_by_year[year]
        year_reverted = sum(1 for d in year_devs if d.reverted)
        rate = year_reverted / len(year_devs) * 100 if year_devs else 0

        if prev_rate is not None:
            diff = rate - prev_rate
            trend = f"{diff:+.1f}%"
        else:
            trend = "-"

        print(f"{year:<8} {len(year_devs):>10} {year_reverted:>10} {rate:>9.1f}% {trend:>15}")
        prev_rate = rate

    # Year-by-year for REST OF DAY (comparison)
    print("\n" + "-" * 70)
    print("REST OF DAY BY YEAR (for comparison)")
    print("-" * 70)

    other_by_year = defaultdict(list)
    for d in other_devs:
        other_by_year[d.date.year].append(d)

    print(f"\n{'Year':<8} {'Total':>10} {'Reverted':>10} {'Rate':>10} {'Trend':>15}")
    print("-" * 55)

    prev_rate = None
    for year in sorted(other_by_year.keys()):
        year_devs = other_by_year[year]
        year_reverted = sum(1 for d in year_devs if d.reverted)
        rate = year_reverted / len(year_devs) * 100 if year_devs else 0

        if prev_rate is not None:
            diff = rate - prev_rate
            trend = f"{diff:+.1f}%"
        else:
            trend = "-"

        print(f"{year:<8} {len(year_devs):>10} {year_reverted:>10} {rate:>9.1f}% {trend:>15}")
        prev_rate = rate

    # Detailed morning breakdown by sub-window
    print("\n" + "-" * 70)
    print("DETAILED MORNING SUB-WINDOWS BY YEAR")
    print("-" * 70)

    for window in ["early_morning", "mid_morning", "late_morning", "pre_midday"]:
        window_devs = [d for d in deviations if d.time_window == window]
        if not window_devs:
            continue

        window_by_year = defaultdict(list)
        for d in window_devs:
            window_by_year[d.date.year].append(d)

        window_name = {
            "early_morning": "9:30-10:00 AM",
            "mid_morning": "10:00-10:30 AM",
            "late_morning": "10:30-11:00 AM",
            "pre_midday": "11:00-11:30 AM",
        }.get(window, window)

        print(f"\n{window_name}:")
        print(f"{'Year':<8} {'Total':>8} {'Reverted':>10} {'Rate':>10}")
        print("-" * 40)

        for year in sorted(window_by_year.keys()):
            year_devs = window_by_year[year]
            year_reverted = sum(1 for d in year_devs if d.reverted)
            rate = year_reverted / len(year_devs) * 100 if year_devs else 0
            print(f"{year:<8} {len(year_devs):>8} {year_reverted:>10} {rate:>9.1f}%")

    # Direction breakdown in morning
    print("\n" + "-" * 70)
    print("MORNING DIRECTION BREAKDOWN BY YEAR")
    print("-" * 70)

    above_morning = [d for d in morning_devs if d.deviation_pct > 0]
    below_morning = [d for d in morning_devs if d.deviation_pct < 0]

    print("\nAbove VWAP → Fade Down:")
    above_by_year = defaultdict(list)
    for d in above_morning:
        above_by_year[d.date.year].append(d)

    print(f"{'Year':<8} {'Total':>8} {'Reverted':>10} {'Rate':>10}")
    print("-" * 40)
    for year in sorted(above_by_year.keys()):
        year_devs = above_by_year[year]
        year_reverted = sum(1 for d in year_devs if d.reverted)
        rate = year_reverted / len(year_devs) * 100 if year_devs else 0
        print(f"{year:<8} {len(year_devs):>8} {year_reverted:>10} {rate:>9.1f}%")

    print("\nBelow VWAP → Fade Up:")
    below_by_year = defaultdict(list)
    for d in below_morning:
        below_by_year[d.date.year].append(d)

    print(f"{'Year':<8} {'Total':>8} {'Reverted':>10} {'Rate':>10}")
    print("-" * 40)
    for year in sorted(below_by_year.keys()):
        year_devs = below_by_year[year]
        year_reverted = sum(1 for d in year_devs if d.reverted)
        rate = year_reverted / len(year_devs) * 100 if year_devs else 0
        print(f"{year:<8} {len(year_devs):>8} {year_reverted:>10} {rate:>9.1f}%")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if morning_by_year:
        latest_year = max(morning_by_year.keys())
        latest_devs = morning_by_year[latest_year]
        latest_rate = sum(1 for d in latest_devs if d.reverted) / len(latest_devs) * 100 if latest_devs else 0

        print(f"\n2024 Morning Reversion Rate: {latest_rate:.1f}%")

        if latest_rate > 80:
            print("Verdict: BUILD IT - Morning edge is stable")
        elif latest_rate > 65:
            print("Verdict: BUILD WITH CAUTION - Edge weakening but viable")
        else:
            print("Verdict: ABANDON - Same pattern as gap fade")


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-01")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    deviations = await analyze_morning_by_year(start, end, args.threshold)
    print_morning_analysis(deviations)


if __name__ == "__main__":
    asyncio.run(main())
