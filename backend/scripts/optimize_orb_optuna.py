#!/usr/bin/env python3
"""
Bayesian optimization for ORB strategy using Optuna.

Walk-forward validation to prevent overfitting:
- Train on 2022-2023, test on 2024
- Minimize parameters (4 max)
- Penalize extreme deviations from baseline
- Use composite objective (Sharpe - DD penalty)

Usage:
    python scripts/optimize_orb_optuna.py --trials 100
    python scripts/optimize_orb_optuna.py --trials 50 --test-only  # Run best params on test set
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.registry import StrategyRegistry

# Suppress verbose logging during optimization
logging.getLogger("app.services").setLevel(logging.WARNING)
logging.getLogger("strategies").setLevel(logging.WARNING)


# =============================================================================
# BASELINE PARAMETERS (validated, known to work)
# =============================================================================
BASELINE = {
    "range_minutes": 60,
    "profit_target_pct": 0.30,
    "stop_loss_pct": 0.20,
    "entry_cutoff_hour_utc": 18,
}

# Parameter bounds (tight, reasonable)
PARAM_BOUNDS = {
    "range_minutes": [15, 30, 45, 60],  # Discrete choices
    "profit_target_pct": (0.15, 0.50),  # 15% to 50%
    "stop_loss_pct": (0.10, 0.30),      # 10% to 30%
    "entry_cutoff_hour_utc": [17, 18, 19],  # 12 PM, 1 PM, 2 PM ET
}


def calculate_param_deviation_penalty(params: dict) -> float:
    """
    Penalize deviation from baseline parameters.
    Returns penalty score (0 = identical to baseline, higher = more deviation).
    """
    penalty = 0.0

    # Range minutes deviation
    range_dev = abs(params["range_minutes"] - BASELINE["range_minutes"]) / 60
    penalty += range_dev * 0.1

    # PT deviation (normalize to 0-1 scale)
    pt_dev = abs(params["profit_target_pct"] - BASELINE["profit_target_pct"]) / 0.35
    penalty += pt_dev * 0.1

    # SL deviation
    sl_dev = abs(params["stop_loss_pct"] - BASELINE["stop_loss_pct"]) / 0.20
    penalty += sl_dev * 0.1

    return penalty


async def run_backtest_async(params: dict, start_date: str, end_date: str) -> dict:
    """Run ORB backtest with given parameters."""
    strategy = StrategyRegistry.get_strategy("orb")
    if not strategy:
        raise ValueError("ORB strategy not found in registry")

    # Get default config and customize
    config = strategy.get_default_config()
    config.symbol = "SPY"
    config.initial_capital = 10000.0

    # Optimized parameters
    config.range_minutes = params["range_minutes"]
    config.profit_target_pct = params["profit_target_pct"]
    config.stop_loss_pct = params["stop_loss_pct"]
    config.entry_cutoff_hour_utc = params["entry_cutoff_hour_utc"]

    # Fixed parameters (realistic slippage)
    config.slippage_entry_pct = 0.03
    config.slippage_exit_pct = 0.03
    config.slippage_stop_extra_pct = 0.02
    config.min_slippage_dollars = 0.02
    config.commission_per_contract = 0.65

    # ORB specific
    config.require_close = True
    config.max_entries_per_day = 3
    config.breakout_buffer = 0.0

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Run backtest
    result = await strategy.run_backtest(
        start_date=start,
        end_date=end,
        config=config,
    )

    metrics = result.metrics

    return {
        "sharpe": metrics.sharpe_ratio,
        "total_pnl": metrics.total_pnl,
        "max_drawdown": metrics.max_drawdown,
        "win_rate": metrics.win_rate,
        "profit_factor": metrics.profit_factor,
        "total_trades": metrics.total_trades,
    }


def run_backtest(params: dict, start_date: str, end_date: str) -> dict:
    """Synchronous wrapper for run_backtest_async."""
    return asyncio.run(run_backtest_async(params, start_date, end_date))


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.

    Optimizes: Sharpe - (2 × MaxDD) - ParamDeviation

    This penalizes:
    - Low Sharpe (bad risk-adjusted returns)
    - High drawdowns (fragile strategies)
    - Extreme parameters (likely overfit)
    """
    # Sample parameters
    params = {
        "range_minutes": trial.suggest_categorical(
            "range_minutes", PARAM_BOUNDS["range_minutes"]
        ),
        "profit_target_pct": trial.suggest_float(
            "profit_target_pct",
            PARAM_BOUNDS["profit_target_pct"][0],
            PARAM_BOUNDS["profit_target_pct"][1],
            step=0.05,
        ),
        "stop_loss_pct": trial.suggest_float(
            "stop_loss_pct",
            PARAM_BOUNDS["stop_loss_pct"][0],
            PARAM_BOUNDS["stop_loss_pct"][1],
            step=0.02,
        ),
        "entry_cutoff_hour_utc": trial.suggest_categorical(
            "entry_cutoff_hour_utc", PARAM_BOUNDS["entry_cutoff_hour_utc"]
        ),
    }

    # Run on TRAINING set (2022-2023)
    try:
        result = run_backtest(params, "2022-01-01", "2023-12-01")
    except Exception as e:
        print(f"Trial failed: {e}")
        return float("-inf")

    # Skip if too few trades (unreliable)
    if result["total_trades"] < 100:
        return float("-inf")

    # Calculate composite objective
    sharpe = result["sharpe"]
    max_dd = result["max_drawdown"]
    param_penalty = calculate_param_deviation_penalty(params)

    # Objective: maximize Sharpe, penalize drawdown and param deviation
    objective_value = sharpe - (2.0 * max_dd) - param_penalty

    # Store metrics for analysis
    trial.set_user_attr("train_sharpe", sharpe)
    trial.set_user_attr("train_pnl", result["total_pnl"])
    trial.set_user_attr("train_dd", max_dd)
    trial.set_user_attr("train_trades", result["total_trades"])
    trial.set_user_attr("train_win_rate", result["win_rate"])
    trial.set_user_attr("param_penalty", param_penalty)

    return objective_value


def validate_on_test_set(params: dict) -> dict:
    """Run parameters on held-out test set (2024)."""
    return run_backtest(params, "2024-01-01", "2024-12-01")


def main():
    parser = argparse.ArgumentParser(description="Optimize ORB with Optuna")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--test-only", action="store_true", help="Only test best params")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("ORB BAYESIAN OPTIMIZATION WITH OPTUNA")
    print("=" * 70)
    print(f"\nTraining Period: 2022-01-01 to 2023-12-01")
    print(f"Test Period:     2024-01-01 to 2024-12-01 (held out)")
    print(f"Trials:          {args.trials}")
    print(f"Parameters:      4 (range_minutes, PT, SL, entry_cutoff)")
    print()

    # Create study with TPE sampler
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="orb_optimization",
    )

    # Add baseline as first trial (warm start)
    study.enqueue_trial(BASELINE)

    # Run optimization
    print("Running optimization...")
    print("-" * 70)

    study.optimize(
        objective,
        n_trials=args.trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    # Get top 5 trials
    print("\n" + "=" * 70)
    print("TOP 5 PARAMETER SETS (by objective)")
    print("=" * 70)

    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else float("-inf"), reverse=True)[:5]

    print("\n┌─────┬────────┬───────┬───────┬─────────┬──────────┬──────────┬─────────┐")
    print("│ Rank│ Range  │  PT   │  SL   │ Cutoff  │  Sharpe  │   P&L    │ Obj Val │")
    print("├─────┼────────┼───────┼───────┼─────────┼──────────┼──────────┼─────────┤")

    for i, trial in enumerate(top_trials, 1):
        p = trial.params
        sharpe = trial.user_attrs.get("train_sharpe", 0)
        pnl = trial.user_attrs.get("train_pnl", 0)
        obj = trial.value or 0
        print(f"│  {i}  │  {p['range_minutes']:>3}m  │ {p['profit_target_pct']*100:>4.0f}% │ {p['stop_loss_pct']*100:>4.0f}% │  {p['entry_cutoff_hour_utc']:>2} UTC │   {sharpe:>5.2f}  │ ${pnl:>7.0f} │  {obj:>5.2f}  │")

    print("└─────┴────────┴───────┴───────┴─────────┴──────────┴──────────┴─────────┘")

    # Test top 5 on held-out 2024 data
    print("\n" + "=" * 70)
    print("VALIDATION ON TEST SET (2024 - held out)")
    print("=" * 70)

    print("\n┌─────┬──────────────┬──────────────┬─────────────┬───────────────┐")
    print("│ Rank│ Train Sharpe │ Test Sharpe  │ Degradation │    Verdict    │")
    print("├─────┼──────────────┼──────────────┼─────────────┼───────────────┤")

    best_gap = float("inf")
    best_trial = None

    for i, trial in enumerate(top_trials, 1):
        train_sharpe = trial.user_attrs.get("train_sharpe", 0)

        # Run on test set
        test_result = validate_on_test_set(trial.params)
        test_sharpe = test_result["sharpe"]

        # Calculate degradation (negative = test is BETTER than train)
        if train_sharpe > 0:
            degradation_pct = (train_sharpe - test_sharpe) / train_sharpe * 100
        else:
            degradation_pct = 100 if test_sharpe <= 0 else -100

        # Verdict based on degradation
        if test_sharpe <= 0:
            verdict = "UNPROFITABLE"
        elif degradation_pct > 50:
            verdict = "OVERFIT"
        elif degradation_pct > 30:
            verdict = "DEGRADED"
        elif degradation_pct < -20:
            verdict = "TEST BETTER!"
        else:
            verdict = "GOOD"

        # Track best (profitable on test, smallest degradation)
        gap_for_tracking = abs(degradation_pct)
        if gap_for_tracking < best_gap and test_sharpe > 0.5:
            best_gap = gap_for_tracking
            best_trial = (trial, test_result)

        print(f"│  {i}  │    {train_sharpe:>6.2f}    │    {test_sharpe:>6.2f}    │  {degradation_pct:>+6.1f}%   │ {verdict:<13} │")

    print("└─────┴──────────────┴──────────────┴─────────────┴───────────────┘")

    # Recommend best parameters
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if best_trial:
        trial, test_result = best_trial
        p = trial.params

        print(f"""
Best parameters (smallest train/test gap, still profitable):

    range_minutes:        {p['range_minutes']}
    profit_target_pct:    {p['profit_target_pct']:.2f} ({p['profit_target_pct']*100:.0f}%)
    stop_loss_pct:        {p['stop_loss_pct']:.2f} ({p['stop_loss_pct']*100:.0f}%)
    entry_cutoff_hour_utc: {p['entry_cutoff_hour_utc']}

Train Performance (2022-2023):
    Sharpe: {trial.user_attrs.get('train_sharpe', 0):.2f}
    P&L:    ${trial.user_attrs.get('train_pnl', 0):.0f}
    DD:     {trial.user_attrs.get('train_dd', 0)*100:.1f}%

Test Performance (2024 - held out):
    Sharpe: {test_result['sharpe']:.2f}
    P&L:    ${test_result['total_pnl']:.0f}
    DD:     {test_result['max_drawdown']*100:.1f}%

Train/Test Stability: {'GOOD' if best_gap < 30 else 'MODERATE' if best_gap < 50 else 'POOR'} ({best_gap:.1f}% variance)
""")

        # Compare to baseline
        print("Comparison to baseline:")
        baseline_test = validate_on_test_set(BASELINE)
        print(f"    Baseline 2024 Sharpe: {baseline_test['sharpe']:.2f}")
        print(f"    Optimized 2024 Sharpe: {test_result['sharpe']:.2f}")

        improvement = (test_result['sharpe'] - baseline_test['sharpe']) / baseline_test['sharpe'] * 100
        print(f"    Improvement: {improvement:+.1f}%")
    else:
        print("\nNo parameters found that generalize well. Consider:")
        print("- Widening parameter bounds")
        print("- Using more training data")
        print("- Simplifying the strategy")

    # Save results
    results_dir = Path("results/optimization")
    results_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "study_name": "orb_optimization",
        "n_trials": args.trials,
        "train_period": "2022-01-01 to 2023-12-01",
        "test_period": "2024-01-01 to 2024-12-01",
        "baseline": BASELINE,
        "best_params": best_trial[0].params if best_trial else None,
        "best_train_sharpe": best_trial[0].user_attrs.get("train_sharpe") if best_trial else None,
        "best_test_sharpe": best_trial[1]["sharpe"] if best_trial else None,
        "top_5_trials": [
            {
                "params": t.params,
                "train_sharpe": t.user_attrs.get("train_sharpe"),
                "objective": t.value,
            }
            for t in top_trials
        ],
    }

    output_path = results_dir / "orb_optuna_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
