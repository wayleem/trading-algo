#!/usr/bin/env python3
"""
VWAP Reversion Thesis Validation.

Does price actually revert to VWAP after deviating?

This is THESIS VALIDATION before building any strategy.
"""

import asyncio
import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.alpaca_client import AlpacaClient


@dataclass
class VWAPDeviation:
    """A single VWAP deviation event."""
    date: date
    deviation_time: datetime
    deviation_pct: float  # Positive = above VWAP, negative = below
    price_at_deviation: float
    vwap_at_deviation: float

    # Reversion analysis
    reverted_to_vwap: bool = False
    reverted_50pct: bool = False
    reversion_time: Optional[datetime] = None
    time_to_reversion_minutes: Optional[int] = None
    max_adverse_excursion_pct: float = 0.0  # How much further before reverting

    # Context
    time_window: str = ""  # morning, midday, afternoon


@dataclass
class DayVWAPData:
    """VWAP data for a single day."""
    date: date
    bars: List[dict] = field(default_factory=list)
    vwap_values: List[float] = field(default_factory=list)
    deviations: List[VWAPDeviation] = field(default_factory=list)


def calculate_vwap(bars: List[dict]) -> List[float]:
    """
    Calculate running VWAP from bars.

    VWAP = cumulative(price * volume) / cumulative(volume)
    Typical price = (high + low + close) / 3
    """
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
            # No volume, use previous VWAP or typical price
            vwap = vwap_values[-1] if vwap_values else typical_price

        vwap_values.append(vwap)

    return vwap_values


def get_time_window(bar_time: datetime) -> str:
    """Categorize time into trading windows (UTC times for ET market hours)."""
    hour = bar_time.hour
    minute = bar_time.minute

    # Market hours in UTC: 14:30 - 21:00 (9:30 AM - 4:00 PM ET)
    if hour < 15 or (hour == 14 and minute >= 30):
        return "morning"  # 9:30-10:30 AM ET
    elif hour < 16:
        return "late_morning"  # 10:30-11:30 AM ET
    elif hour < 18:
        return "midday"  # 11:30 AM - 1:30 PM ET
    elif hour < 20:
        return "afternoon"  # 1:30-3:30 PM ET
    else:
        return "close"  # 3:30-4:00 PM ET


def find_deviations(
    bars: List[dict],
    vwap_values: List[float],
    threshold_pct: float,
    day_date: date,
) -> List[VWAPDeviation]:
    """
    Find all deviations from VWAP exceeding threshold.

    Only triggers once per direction until price returns to VWAP.
    """
    deviations = []
    in_deviation_above = False
    in_deviation_below = False
    current_deviation: Optional[VWAPDeviation] = None

    for i, (bar, vwap) in enumerate(zip(bars, vwap_values)):
        price = bar["close"]
        bar_time = bar["timestamp"]

        if vwap <= 0:
            continue

        deviation_pct = (price - vwap) / vwap * 100

        # Check for new deviation above VWAP
        if deviation_pct >= threshold_pct and not in_deviation_above:
            in_deviation_above = True
            in_deviation_below = False
            current_deviation = VWAPDeviation(
                date=day_date,
                deviation_time=bar_time,
                deviation_pct=deviation_pct,
                price_at_deviation=price,
                vwap_at_deviation=vwap,
                time_window=get_time_window(bar_time),
            )
            deviations.append(current_deviation)

        # Check for new deviation below VWAP
        elif deviation_pct <= -threshold_pct and not in_deviation_below:
            in_deviation_below = True
            in_deviation_above = False
            current_deviation = VWAPDeviation(
                date=day_date,
                deviation_time=bar_time,
                deviation_pct=deviation_pct,
                price_at_deviation=price,
                vwap_at_deviation=vwap,
                time_window=get_time_window(bar_time),
            )
            deviations.append(current_deviation)

        # Track max adverse excursion for current deviation
        if current_deviation is not None and not current_deviation.reverted_to_vwap:
            if current_deviation.deviation_pct > 0:  # Above VWAP
                # Adverse = going further above
                current_adverse = max(0, deviation_pct - current_deviation.deviation_pct)
                current_deviation.max_adverse_excursion_pct = max(
                    current_deviation.max_adverse_excursion_pct, current_adverse
                )
            else:  # Below VWAP
                # Adverse = going further below
                current_adverse = max(0, abs(deviation_pct) - abs(current_deviation.deviation_pct))
                current_deviation.max_adverse_excursion_pct = max(
                    current_deviation.max_adverse_excursion_pct, current_adverse
                )

        # Check for reversion to VWAP
        if current_deviation is not None and not current_deviation.reverted_to_vwap:
            initial_dev = current_deviation.deviation_pct

            if initial_dev > 0:  # Was above VWAP
                # Check 50% reversion
                if not current_deviation.reverted_50pct:
                    halfway = current_deviation.vwap_at_deviation + (
                        current_deviation.price_at_deviation - current_deviation.vwap_at_deviation
                    ) / 2
                    if price <= halfway:
                        current_deviation.reverted_50pct = True

                # Check full reversion
                if price <= vwap:
                    current_deviation.reverted_to_vwap = True
                    current_deviation.reversion_time = bar_time
                    current_deviation.time_to_reversion_minutes = int(
                        (bar_time - current_deviation.deviation_time).total_seconds() / 60
                    )
                    in_deviation_above = False
                    current_deviation = None

            else:  # Was below VWAP
                # Check 50% reversion
                if not current_deviation.reverted_50pct:
                    halfway = current_deviation.vwap_at_deviation - (
                        current_deviation.vwap_at_deviation - current_deviation.price_at_deviation
                    ) / 2
                    if price >= halfway:
                        current_deviation.reverted_50pct = True

                # Check full reversion
                if price >= vwap:
                    current_deviation.reverted_to_vwap = True
                    current_deviation.reversion_time = bar_time
                    current_deviation.time_to_reversion_minutes = int(
                        (bar_time - current_deviation.deviation_time).total_seconds() / 60
                    )
                    in_deviation_below = False
                    current_deviation = None

    return deviations


async def analyze_vwap_reversion(
    start_date: date,
    end_date: date,
    thresholds: List[float] = [0.3, 0.5, 0.7, 1.0],
) -> Dict[float, List[VWAPDeviation]]:
    """
    Analyze VWAP reversion at multiple thresholds.

    Returns dict mapping threshold -> list of deviations
    """
    client = AlpacaClient()

    print(f"Analyzing VWAP reversion from {start_date} to {end_date}")
    print(f"Thresholds: {thresholds}")

    results = {t: [] for t in thresholds}

    # Process day by day
    current_date = start_date
    total_days = 0

    while current_date <= end_date:
        # Fetch 1-min bars for the day
        try:
            bars = await client.get_stock_bars(
                symbol="SPY",
                timeframe="1Min",
                start=datetime.combine(current_date, time(14, 30)),  # 9:30 AM ET
                end=datetime.combine(current_date, time(21, 0)),  # 4:00 PM ET
                limit=500,
            )

            if len(bars) > 30:  # Need enough bars for meaningful VWAP
                total_days += 1
                vwap_values = calculate_vwap(bars)

                for threshold in thresholds:
                    deviations = find_deviations(bars, vwap_values, threshold, current_date)
                    results[threshold].extend(deviations)

                if total_days % 50 == 0:
                    print(f"  Processed {total_days} days...")

        except Exception as e:
            pass  # Skip days with no data

        current_date += timedelta(days=1)

    print(f"Processed {total_days} trading days")

    return results


def print_analysis(results: Dict[float, List[VWAPDeviation]]):
    """Print comprehensive VWAP reversion analysis."""
    print("\n" + "=" * 80)
    print("VWAP REVERSION THESIS VALIDATION")
    print("=" * 80)

    for threshold in sorted(results.keys()):
        deviations = results[threshold]

        if not deviations:
            print(f"\n{threshold}% threshold: No deviations found")
            continue

        total = len(deviations)
        reverted = sum(1 for d in deviations if d.reverted_to_vwap)
        reverted_50 = sum(1 for d in deviations if d.reverted_50pct)

        # By direction
        above = [d for d in deviations if d.deviation_pct > 0]
        below = [d for d in deviations if d.deviation_pct < 0]
        above_reverted = sum(1 for d in above if d.reverted_to_vwap)
        below_reverted = sum(1 for d in below if d.reverted_to_vwap)

        print(f"\n{'='*80}")
        print(f"THRESHOLD: {threshold}% deviation from VWAP")
        print(f"{'='*80}")

        print(f"\nTotal deviations: {total}")
        print(f"  Above VWAP: {len(above)}")
        print(f"  Below VWAP: {len(below)}")

        print(f"\n{'-'*60}")
        print("REVERSION RATES")
        print(f"{'-'*60}")

        print(f"\n{'Metric':<35} {'Count':>10} {'Rate':>10}")
        print("-" * 55)
        print(f"{'Full reversion to VWAP':<35} {reverted:>10} {reverted/total*100:>9.1f}%")
        print(f"{'Partial reversion (>=50%)':<35} {reverted_50:>10} {reverted_50/total*100:>9.1f}%")

        print(f"\n{'Direction':<20} {'Total':>8} {'Reverted':>10} {'Rate':>10}")
        print("-" * 50)
        if above:
            print(f"{'Above VWAP (fade down)':<20} {len(above):>8} {above_reverted:>10} {above_reverted/len(above)*100:>9.1f}%")
        if below:
            print(f"{'Below VWAP (fade up)':<20} {len(below):>8} {below_reverted:>10} {below_reverted/len(below)*100:>9.1f}%")

        # Time to reversion
        reverted_devs = [d for d in deviations if d.reverted_to_vwap and d.time_to_reversion_minutes]
        if reverted_devs:
            times = [d.time_to_reversion_minutes for d in reverted_devs]
            avg_time = statistics.mean(times)
            median_time = statistics.median(times)

            print(f"\n{'-'*60}")
            print("TIME TO REVERSION (for deviations that reverted)")
            print(f"{'-'*60}")

            within_15 = sum(1 for t in times if t <= 15)
            within_30 = sum(1 for t in times if t <= 30)
            within_60 = sum(1 for t in times if t <= 60)
            within_120 = sum(1 for t in times if t <= 120)

            print(f"\n{'Time Window':<25} {'Count':>8} {'Cumulative %':>15}")
            print("-" * 50)
            print(f"{'Within 15 min':<25} {within_15:>8} {within_15/len(reverted_devs)*100:>14.1f}%")
            print(f"{'Within 30 min':<25} {within_30:>8} {within_30/len(reverted_devs)*100:>14.1f}%")
            print(f"{'Within 60 min':<25} {within_60:>8} {within_60/len(reverted_devs)*100:>14.1f}%")
            print(f"{'Within 2 hours':<25} {within_120:>8} {within_120/len(reverted_devs)*100:>14.1f}%")

            print(f"\nAverage time to reversion: {avg_time:.0f} minutes")
            print(f"Median time to reversion: {median_time:.0f} minutes")

        # Max adverse excursion
        adverse = [d.max_adverse_excursion_pct for d in deviations if d.max_adverse_excursion_pct > 0]
        if adverse:
            print(f"\n{'-'*60}")
            print("MAX ADVERSE EXCURSION (how much further before reverting)")
            print(f"{'-'*60}")

            avg_adverse = statistics.mean(adverse)
            p75_adverse = sorted(adverse)[int(len(adverse) * 0.75)]
            p90_adverse = sorted(adverse)[int(len(adverse) * 0.90)]

            print(f"\nAverage: {avg_adverse:.2f}%")
            print(f"75th percentile: {p75_adverse:.2f}%")
            print(f"90th percentile: {p90_adverse:.2f}%")
            print(f"\n→ Stop loss should be at least {p75_adverse:.1f}% beyond entry")

        # By time window
        print(f"\n{'-'*60}")
        print("BY TIME WINDOW")
        print(f"{'-'*60}")

        by_window = defaultdict(list)
        for d in deviations:
            by_window[d.time_window].append(d)

        print(f"\n{'Window':<15} {'Total':>8} {'Reverted':>10} {'Rate':>10}")
        print("-" * 45)
        for window in ["morning", "late_morning", "midday", "afternoon", "close"]:
            if window in by_window:
                w_devs = by_window[window]
                w_reverted = sum(1 for d in w_devs if d.reverted_to_vwap)
                print(f"{window:<15} {len(w_devs):>8} {w_reverted:>10} {w_reverted/len(w_devs)*100:>9.1f}%")

        # Year-by-year
        print(f"\n{'-'*60}")
        print("YEAR-BY-YEAR TREND")
        print(f"{'-'*60}")

        by_year = defaultdict(list)
        for d in deviations:
            by_year[d.date.year].append(d)

        print(f"\n{'Year':<8} {'Total':>8} {'Reverted':>10} {'Rate':>10}")
        print("-" * 40)
        for year in sorted(by_year.keys()):
            y_devs = by_year[year]
            y_reverted = sum(1 for d in y_devs if d.reverted_to_vwap)
            print(f"{year:<8} {len(y_devs):>8} {y_reverted:>10} {y_reverted/len(y_devs)*100:>9.1f}%")

    # Final verdict for best threshold
    print("\n" + "=" * 80)
    print("THESIS VERDICT BY THRESHOLD")
    print("=" * 80)

    for threshold in sorted(results.keys()):
        deviations = results[threshold]
        if not deviations:
            continue

        total = len(deviations)
        reverted = sum(1 for d in deviations if d.reverted_to_vwap)
        rate = reverted / total * 100

        # Check year-over-year trend
        by_year = defaultdict(list)
        for d in deviations:
            by_year[d.date.year].append(d)

        years = sorted(by_year.keys())
        if len(years) >= 2:
            first_year = years[0]
            last_year = years[-1]
            first_rate = sum(1 for d in by_year[first_year] if d.reverted_to_vwap) / len(by_year[first_year]) * 100
            last_rate = sum(1 for d in by_year[last_year] if d.reverted_to_vwap) / len(by_year[last_year]) * 100
            trend = "DECLINING" if last_rate < first_rate - 5 else "STABLE" if abs(last_rate - first_rate) <= 5 else "IMPROVING"
        else:
            trend = "UNKNOWN"

        if rate < 50:
            verdict = "ABANDON"
        elif rate < 55:
            verdict = "WEAK"
        elif rate < 65:
            verdict = "VALID"
        else:
            verdict = "STRONG"

        print(f"\n{threshold}% threshold:")
        print(f"  Reversion rate: {rate:.1f}%")
        print(f"  Year trend: {trend}")
        print(f"  Verdict: {verdict}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze VWAP reversion rates")
    parser.add_argument("--start", default="2022-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-01", help="End date")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    results = await analyze_vwap_reversion(
        start_date=start,
        end_date=end,
        thresholds=[0.3, 0.5, 0.7, 1.0],
    )

    print_analysis(results)


if __name__ == "__main__":
    asyncio.run(main())
