#!/usr/bin/env python3
"""
Bayesian optimization using Optuna for full parameter tuning.

Usage:
    python scripts/optimize_bayesian.py --start 2023-12-01 --end 2025-12-01 --trials 100
    python scripts/optimize_bayesian.py --start 2023-12-01 --end 2025-12-01 --trials 200 --symbol SPY
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_backtest_sync(params, start_date, end_date, symbol):
    """Run a single backtest and return metrics."""
    cmd = [
        sys.executable, "scripts/run_backtest.py",
        "--start", start_date,
        "--end", end_date,
        "--symbol", symbol,
        "--profit-target-pct", str(params['profit_target']),
        "--stop-loss-pct", str(params['stop_loss']),
        "--slippage", "0.01",
        "--timeframe", "parallel",
        "--use-pattern-sizing",
        "--pattern-bonus-contracts", str(params['pattern_bonus']),
        "--strike-offset", str(params['strike_offset']),
        "--rsi-oversold", str(params['rsi_oversold']),
        "--rsi-overbought", str(params['rsi_overbought']),
        "--rsi-period", str(params['rsi_period']),
        "--sma-period", str(params['sma_period']),
        "--entry-cutoff-hour", str(params['entry_cutoff_hour']),
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
        total_trades = 0

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
            elif 'Total Trades:' in line:
                try:
                    total_trades = int(line.split(':')[1].strip())
                except:
                    pass

        return {
            'sharpe': sharpe,
            'return': total_return,
            'win_rate': win_rate,
            'drawdown': drawdown,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'success': True
        }

    except Exception as e:
        return {
            'sharpe': 0.0,
            'return': 0.0,
            'win_rate': 0.0,
            'drawdown': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'success': False,
            'error': str(e)
        }


def create_objective(start_date, end_date, symbol, objective_type='sharpe'):
    """Create an Optuna objective function."""

    def objective(trial):
        # Suggest parameters
        params = {
            'rsi_oversold': trial.suggest_int('rsi_oversold', 30, 45),
            'rsi_overbought': trial.suggest_int('rsi_overbought', 55, 70),
            'profit_target': trial.suggest_float('profit_target', 0.05, 0.25, step=0.01),
            'stop_loss': trial.suggest_float('stop_loss', 0.25, 0.60, step=0.05),
            'rsi_period': trial.suggest_int('rsi_period', 10, 20),
            'sma_period': trial.suggest_int('sma_period', 10, 20),
            'pattern_bonus': trial.suggest_int('pattern_bonus', 0, 4),
            'strike_offset': trial.suggest_float('strike_offset', -1.0, 2.0, step=0.5),
            # Entry cutoff: 17=12PM ET, 18=1PM ET, 19=2PM ET, 20=3PM ET
            'entry_cutoff_hour': trial.suggest_int('entry_cutoff_hour', 17, 20),
        }

        # Constraint: RSI overbought > RSI oversold + 20
        if params['rsi_overbought'] <= params['rsi_oversold'] + 10:
            return float('-inf')

        # Run backtest
        result = run_backtest_sync(params, start_date, end_date, symbol)

        if not result['success']:
            return float('-inf')

        # Calculate score based on objective type
        if objective_type == 'sharpe':
            score = result['sharpe']
        elif objective_type == 'return':
            score = result['return']
        elif objective_type == 'combined':
            # Multi-objective: Sharpe + scaled Return - Drawdown penalty
            sharpe_component = result['sharpe'] * 1.0
            return_component = result['return'] / 500  # Scale down return
            drawdown_penalty = result['drawdown'] / 100  # Penalty for drawdown
            min_trades_penalty = 0 if result['total_trades'] >= 100 else -1  # Require min trades

            score = sharpe_component + return_component - drawdown_penalty + min_trades_penalty
        else:
            score = result['sharpe']

        # Log trial
        trial.set_user_attr('return', result['return'])
        trial.set_user_attr('win_rate', result['win_rate'])
        trial.set_user_attr('drawdown', result['drawdown'])
        trial.set_user_attr('profit_factor', result['profit_factor'])
        trial.set_user_attr('total_trades', result['total_trades'])

        return score

    return objective


def run_optimization(start_date, end_date, symbol='SPX', n_trials=100, objective_type='combined'):
    """Run Bayesian optimization."""
    print("=" * 60)
    print("BAYESIAN OPTIMIZATION: Full Parameter Tuning")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Trials: {n_trials}")
    print(f"Objective: {objective_type}")
    print("=" * 60)
    print()

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=10),
        study_name=f'rsi_optimization_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )

    # Create objective
    objective = create_objective(start_date, end_date, symbol, objective_type)

    # Optimize
    start_time = time.time()

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,  # Sequential due to ThetaTerminal connection
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Trials completed: {len(study.trials)}")
    print()

    # Best trial
    best = study.best_trial
    print("BEST PARAMETERS:")
    print("-" * 40)
    for key, value in best.params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print()
    print("BEST METRICS:")
    print("-" * 40)
    print(f"  Score: {best.value:.4f}")
    print(f"  Return: {best.user_attrs.get('return', 'N/A')}%")
    print(f"  Win Rate: {best.user_attrs.get('win_rate', 'N/A')}%")
    print(f"  Drawdown: {best.user_attrs.get('drawdown', 'N/A')}%")
    print(f"  Profit Factor: {best.user_attrs.get('profit_factor', 'N/A')}")
    print(f"  Total Trades: {best.user_attrs.get('total_trades', 'N/A')}")
    print("=" * 60)

    # Generate CLI command with best params
    print()
    print("RECOMMENDED CLI COMMAND:")
    print("-" * 40)
    # Map entry cutoff hour to ET time
    cutoff_et_map = {17: "12PM", 18: "1PM", 19: "2PM", 20: "3PM"}
    cutoff_hour = best.params.get('entry_cutoff_hour', 19)
    cutoff_et = cutoff_et_map.get(cutoff_hour, f"{cutoff_hour}:00 UTC")
    cmd = f"""python scripts/run_backtest.py \\
    --start {start_date} --end {end_date} \\
    --symbol {symbol} \\
    --profit-target-pct {best.params['profit_target']:.2f} \\
    --stop-loss-pct {best.params['stop_loss']:.2f} \\
    --rsi-oversold {best.params['rsi_oversold']} \\
    --rsi-overbought {best.params['rsi_overbought']} \\
    --rsi-period {best.params['rsi_period']} \\
    --sma-period {best.params['sma_period']} \\
    --strike-offset {best.params['strike_offset']:.1f} \\
    --entry-cutoff-hour {cutoff_hour} \\
    --slippage 0.01 --timeframe parallel \\
    --use-pattern-sizing --pattern-bonus-contracts {best.params['pattern_bonus']}"""
    print(cmd)
    print(f"(Entry cutoff: {cutoff_et} ET)")
    print()

    # Save visualizations if possible
    try:
        import matplotlib.pyplot as plt

        # Optimization history
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.savefig(f'optuna_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=150)
        print("Saved optimization history plot")

        # Parameter importances
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.savefig(f'optuna_importance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=150)
        print("Saved parameter importance plot")

        plt.close('all')
    except Exception as e:
        print(f"Could not save visualizations: {e}")

    # Print top 5 trials
    print()
    print("TOP 5 TRIALS:")
    print("-" * 60)
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else float('-inf'), reverse=True)
    for i, trial in enumerate(sorted_trials[:5]):
        if trial.value:
            print(f"{i+1}. Score={trial.value:.3f}, Return={trial.user_attrs.get('return', 'N/A')}%, "
                  f"Sharpe implied, PT={trial.params['profit_target']:.0%}, SL={trial.params['stop_loss']:.0%}")

    return study


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian optimization for strategy parameters")

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
        "--trials",
        type=int,
        default=100,
        help="Number of optimization trials (default: 100)",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="combined",
        choices=["sharpe", "return", "combined"],
        help="Optimization objective (default: combined)",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    study = run_optimization(
        start_date=args.start,
        end_date=args.end,
        symbol=args.symbol,
        n_trials=args.trials,
        objective_type=args.objective
    )
