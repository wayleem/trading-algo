#!/usr/bin/env python3
"""
Bayesian Strategy Selector using Thompson Sampling.

Implements a multi-armed bandit approach to strategy selection, balancing
exploration (gathering data on uncertain strategies) with exploitation
(using the strategy with highest expected utility).

Usage:
    # Initialize with backtest data
    python bayesian_strategy_selector.py init

    # Get recommendation for today
    python bayesian_strategy_selector.py recommend

    # Record a trade result
    python bayesian_strategy_selector.py record --strategy rsi_only --outcome win --pnl 55.00

    # Show current posteriors and statistics
    python bayesian_strategy_selector.py status

    # Run simulation to test adaptation
    python bayesian_strategy_selector.py simulate --days 30
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats


# Default state file location
DEFAULT_STATE_FILE = Path(__file__).parent.parent / "data" / "bayesian_state.json"


@dataclass
class StrategyPosterior:
    """
    Beta posterior distribution for a trading strategy.

    The Beta distribution is the conjugate prior for binomial outcomes (win/loss),
    making it ideal for modeling win rate uncertainty.

    Attributes:
        name: Strategy identifier
        alpha: Pseudo-count of wins (prior + observed)
        beta: Pseudo-count of losses (prior + observed)
        trades_per_day: Expected number of trades per trading day
        avg_win: Average profit on winning trades
        avg_loss: Average loss on losing trades (positive number)
        commission_per_trade: Round-trip commission cost
    """

    name: str
    alpha: float
    beta: float
    trades_per_day: float
    avg_win: float
    avg_loss: float
    commission_per_trade: float = 1.30  # $0.65 each way

    @property
    def mean(self) -> float:
        """Expected win rate (posterior mean)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Posterior variance of win rate."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total**2 * (total + 1))

    @property
    def std(self) -> float:
        """Posterior standard deviation of win rate."""
        return np.sqrt(self.variance)

    @property
    def total_trades(self) -> int:
        """Total observed trades (alpha + beta - 2 for uniform prior)."""
        return int(self.alpha + self.beta - 2)

    def ci_95(self) -> tuple[float, float]:
        """95% credible interval for win rate."""
        return (
            stats.beta.ppf(0.025, self.alpha, self.beta),
            stats.beta.ppf(0.975, self.alpha, self.beta),
        )

    def sample(self) -> float:
        """Draw a sample from the posterior (for Thompson Sampling)."""
        return stats.beta.rvs(self.alpha, self.beta)

    def kelly_fraction(self, full_kelly: bool = False) -> float:
        """
        Calculate Kelly Criterion fraction for position sizing.

        Kelly formula: f* = (p × b - q) / b
        where p = win rate, q = loss rate, b = win/loss ratio

        Args:
            full_kelly: If True, return full Kelly. Otherwise, fractional (25%).

        Returns:
            Fraction of capital to allocate per trade.
        """
        p = self.mean  # win rate
        q = 1 - p
        b = self.avg_win / self.avg_loss if self.avg_loss > 0 else 100.0  # win/loss ratio

        full = (p * b - q) / b
        full = max(0, min(1, full))  # Clamp to [0, 1]

        return full if full_kelly else full * 0.25

    def kelly_contracts(self, base_contracts: float = 1.0, baseline_kelly: float = 0.684) -> float:
        """
        Calculate Kelly-adjusted contract count.

        Scales contract size relative to a baseline strategy (default: RSI Only).
        Higher win rate / better edge = more contracts.

        Args:
            base_contracts: Baseline number of contracts
            baseline_kelly: Kelly fraction of the baseline strategy (RSI Only = 0.684)

        Returns:
            Kelly-adjusted number of contracts
        """
        my_kelly = self.kelly_fraction(full_kelly=True)
        return base_contracts * (my_kelly / baseline_kelly) if baseline_kelly > 0 else base_contracts

    def expected_daily_profit(
        self, win_rate: Optional[float] = None, use_kelly: bool = False
    ) -> float:
        """
        Calculate expected daily profit given a win rate.

        Args:
            win_rate: Win rate to use (defaults to posterior mean)
            use_kelly: If True, scale by Kelly-adjusted contracts

        Returns:
            Expected profit per trading day
        """
        if win_rate is None:
            win_rate = self.mean

        contracts = self.kelly_contracts() if use_kelly else 1.0

        expected_wins = win_rate * self.trades_per_day * contracts
        expected_losses = (1 - win_rate) * self.trades_per_day * contracts

        gross_profit = expected_wins * self.avg_win - expected_losses * self.avg_loss
        commissions = self.trades_per_day * contracts * self.commission_per_trade

        return gross_profit - commissions

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyPosterior":
        """Create from dictionary."""
        return cls(**data)


class BayesianStrategySelector:
    """
    Thompson Sampling-based strategy selector.

    Uses Bayesian inference to maintain posterior distributions over each
    strategy's win rate, and Thompson Sampling to balance exploration and
    exploitation when selecting strategies.
    """

    def __init__(self, state_file: Path = DEFAULT_STATE_FILE):
        self.state_file = state_file
        self.strategies: dict[str, StrategyPosterior] = {}
        self.trade_history: list[dict] = []
        self._load_state()

    def _load_state(self) -> None:
        """Load state from JSON file if it exists."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)

            self.strategies = {
                name: StrategyPosterior.from_dict(strat_data)
                for name, strat_data in data.get("strategies", {}).items()
            }
            self.trade_history = data.get("trade_history", [])

    def _save_state(self) -> None:
        """Save state to JSON file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "strategies": {
                name: strat.to_dict() for name, strat in self.strategies.items()
            },
            "trade_history": self.trade_history,
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def initialize_from_backtest(
        self,
        rsi_wins: int = 421,
        rsi_losses: int = 193,
        both_wins: int = 96,
        both_losses: int = 32,
        macd_wins: int = 118,
        macd_losses: int = 67,
    ) -> None:
        """
        Initialize posteriors from backtest results.

        Default values are from Jan-Jun 2024 backtest:
        - RSI Only: 614 trades, 68.6% win rate
        - Both Agree: 128 trades, 75.0% win rate
        - MACD Filter: 185 trades, 63.8% win rate
        """
        # Use uniform prior Beta(1,1), so alpha = wins + 1, beta = losses + 1
        self.strategies = {
            "rsi_only": StrategyPosterior(
                name="rsi_only",
                alpha=rsi_wins + 1,
                beta=rsi_losses + 1,
                trades_per_day=4.9,  # 614 trades / 126 trading days
                avg_win=71.80,  # From backtest data
                avg_loss=0.50,  # Minimal loss due to stop loss
            ),
            "both_agree": StrategyPosterior(
                name="both_agree",
                alpha=both_wins + 1,
                beta=both_losses + 1,
                trades_per_day=1.0,  # 128 trades / 126 trading days
                avg_win=130.64,  # Higher avg win due to quality signals
                avg_loss=0.50,
            ),
            "macd_filter": StrategyPosterior(
                name="macd_filter",
                alpha=macd_wins + 1,
                beta=macd_losses + 1,
                trades_per_day=1.5,  # 185 trades / 126 trading days
                avg_win=86.23,
                avg_loss=0.50,
            ),
        }
        self.trade_history = []
        self._save_state()
        print("Initialized posteriors from backtest data:")
        self.print_status()

    def thompson_sample(self, n_samples: int = 1, use_kelly: bool = False) -> str:
        """
        Select strategy using Thompson Sampling.

        For each strategy, samples from its posterior and calculates expected
        daily profit. Returns the strategy with highest sampled expected profit.

        Args:
            n_samples: Number of Thompson samples to average (1 = pure Thompson)
            use_kelly: If True, use Kelly-adjusted position sizing

        Returns:
            Name of recommended strategy
        """
        if not self.strategies:
            raise ValueError("No strategies initialized. Run 'init' first.")

        best_strategy = None
        best_expected = float("-inf")
        sample_results = {}

        for name, strat in self.strategies.items():
            # Sample win rate from posterior
            if n_samples == 1:
                sampled_win_rate = strat.sample()
            else:
                # Average multiple samples for more stable recommendation
                sampled_win_rate = np.mean([strat.sample() for _ in range(n_samples)])

            expected = strat.expected_daily_profit(sampled_win_rate, use_kelly=use_kelly)
            contracts = strat.kelly_contracts() if use_kelly else 1.0

            sample_results[name] = {
                "sampled_win_rate": sampled_win_rate,
                "expected_profit": expected,
                "contracts": contracts,
            }

            if expected > best_expected:
                best_expected = expected
                best_strategy = name

        return best_strategy, sample_results

    def update_posterior(self, strategy: str, won: bool, pnl: float) -> None:
        """
        Update posterior with new trade observation.

        Args:
            strategy: Strategy name
            won: Whether the trade was profitable
            pnl: Actual P&L of the trade
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        strat = self.strategies[strategy]
        if won:
            strat.alpha += 1
        else:
            strat.beta += 1

        # Record in history
        self.trade_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "strategy": strategy,
                "won": won,
                "pnl": pnl,
            }
        )

        self._save_state()

    def probability_of_superiority(
        self, strat_a: str, strat_b: str, n_samples: int = 10000
    ) -> float:
        """
        Monte Carlo estimate of P(A's win rate > B's win rate).

        Args:
            strat_a: First strategy name
            strat_b: Second strategy name
            n_samples: Number of Monte Carlo samples

        Returns:
            Probability that strat_a has higher true win rate than strat_b
        """
        a = self.strategies[strat_a]
        b = self.strategies[strat_b]

        a_samples = stats.beta.rvs(a.alpha, a.beta, size=n_samples)
        b_samples = stats.beta.rvs(b.alpha, b.beta, size=n_samples)

        return np.mean(a_samples > b_samples)

    def expected_utility_comparison(self, n_samples: int = 10000) -> dict:
        """
        Compare expected daily profit across strategies using Monte Carlo.

        Returns:
            Dictionary with comparison statistics
        """
        results = {}
        for name, strat in self.strategies.items():
            # Sample many win rates
            win_rates = stats.beta.rvs(strat.alpha, strat.beta, size=n_samples)
            profits = [strat.expected_daily_profit(wr) for wr in win_rates]

            results[name] = {
                "mean": np.mean(profits),
                "std": np.std(profits),
                "ci_5": np.percentile(profits, 5),
                "ci_95": np.percentile(profits, 95),
            }

        return results

    def print_status(self, use_kelly: bool = False) -> None:
        """Print current posterior status in a formatted table."""
        print()
        print("=" * 80)
        title = "Bayesian Strategy Selector Status"
        if use_kelly:
            title += " (Kelly Sizing)"
        print(f"              {title}")
        print("=" * 80)

        if not self.strategies:
            print("No strategies initialized. Run 'init' first.")
            return

        # Header
        if use_kelly:
            print(
                f"{'Strategy':<15} | {'Win Rate (95% CI)':<22} | {'Kelly':<6} | {'Contracts':<9} | {'E[Daily $]':<10}"
            )
        else:
            print(
                f"{'Strategy':<15} | {'Win Rate (95% CI)':<25} | {'Trades':<8} | {'E[Daily $]':<12}"
            )
        print("-" * 80)

        # Strategy rows
        for name, strat in self.strategies.items():
            ci = strat.ci_95()
            win_rate_str = f"{strat.mean*100:.1f}% ({ci[0]*100:.1f}%-{ci[1]*100:.1f}%)"
            expected = strat.expected_daily_profit(use_kelly=use_kelly)

            if use_kelly:
                kelly = strat.kelly_fraction(full_kelly=True)
                contracts = strat.kelly_contracts()
                print(
                    f"{name:<15} | {win_rate_str:<22} | {kelly*100:>5.1f}% | {contracts:>8.2f}x | ${expected:>9.2f}"
                )
            else:
                print(
                    f"{name:<15} | {win_rate_str:<25} | {strat.total_trades:<8} | ${expected:>10.2f}"
                )

        print("-" * 80)

        # Probability comparisons
        if len(self.strategies) >= 2:
            strat_names = list(self.strategies.keys())
            print("\nProbability of Superiority (Win Rate):")
            for i, a in enumerate(strat_names):
                for b in strat_names[i + 1 :]:
                    p = self.probability_of_superiority(a, b)
                    print(f"  P({a} > {b}) = {p*100:.1f}%")

        # Kelly comparison (only show if not already in Kelly mode)
        if not use_kelly:
            print("\nKelly Criterion Analysis:")
            for name, strat in self.strategies.items():
                kelly = strat.kelly_fraction(full_kelly=True)
                contracts = strat.kelly_contracts()
                kelly_profit = strat.expected_daily_profit(use_kelly=True)
                print(
                    f"  {name}: Full Kelly={kelly*100:.1f}%, "
                    f"Contracts={contracts:.2f}x, E[Daily]=${kelly_profit:.2f}"
                )

        # Expected utility comparison
        print("\nExpected Daily Profit Distribution:")
        util_results = self.expected_utility_comparison()
        for name, result in util_results.items():
            print(
                f"  {name}: ${result['mean']:.2f} "
                f"(90% CI: ${result['ci_5']:.2f} to ${result['ci_95']:.2f})"
            )

        # Thompson sample recommendation
        print("\n" + "=" * 80)
        best, samples = self.thompson_sample(use_kelly=use_kelly)
        print(f"TODAY'S RECOMMENDATION: {best}")
        print("Thompson samples" + (" (Kelly-adjusted):" if use_kelly else ":"))
        for name, result in samples.items():
            contracts_str = f" ({result['contracts']:.2f}x)" if use_kelly else ""
            print(
                f"  {name}: {result['sampled_win_rate']*100:.1f}%{contracts_str} -> ${result['expected_profit']:.2f}/day"
            )
        print("=" * 80)

        # Recent trades
        if self.trade_history:
            print(f"\nRecent trades: {len(self.trade_history)} total")
            for trade in self.trade_history[-5:]:
                outcome = "WIN" if trade["won"] else "LOSS"
                print(
                    f"  {trade['timestamp'][:10]} {trade['strategy']}: {outcome} ${trade['pnl']:+.2f}"
                )

    def simulate_adaptation(self, days: int = 30, seed: int = 42) -> None:
        """
        Simulate adaptive strategy selection over multiple days.

        Uses Thompson Sampling to select strategies and simulates trade outcomes
        based on current posteriors. Shows how posteriors evolve over time.

        Args:
            days: Number of trading days to simulate
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        print(f"\nSimulating {days} days of adaptive trading...")
        print("-" * 60)

        # Assume true win rates (for simulation only - unknown in practice)
        true_rates = {
            "rsi_only": 0.686,
            "both_agree": 0.750,
            "macd_filter": 0.638,
        }

        strategy_counts = {name: 0 for name in self.strategies}
        strategy_profits = {name: 0.0 for name in self.strategies}

        for day in range(1, days + 1):
            # Select strategy using Thompson Sampling
            selected, _ = self.thompson_sample()
            strategy_counts[selected] += 1

            strat = self.strategies[selected]
            true_rate = true_rates[selected]

            # Simulate trades for the day
            n_trades = int(np.random.poisson(strat.trades_per_day))
            day_pnl = 0.0

            for _ in range(n_trades):
                won = np.random.random() < true_rate
                if won:
                    pnl = strat.avg_win
                else:
                    pnl = -strat.avg_loss
                pnl -= strat.commission_per_trade

                day_pnl += pnl

                # Update posterior (but don't save to file in simulation)
                if won:
                    self.strategies[selected].alpha += 1
                else:
                    self.strategies[selected].beta += 1

            strategy_profits[selected] += day_pnl

            if day % 5 == 0:
                print(f"Day {day:3d}: Selected {selected}, {n_trades} trades, ${day_pnl:+.2f}")

        print("-" * 60)
        print("\nSimulation Results:")
        print(f"{'Strategy':<15} | {'Days Selected':<15} | {'Total Profit':<15} | {'Final Win Rate':<15}")
        print("-" * 70)
        for name in self.strategies:
            final_rate = self.strategies[name].mean
            print(
                f"{name:<15} | {strategy_counts[name]:<15} | "
                f"${strategy_profits[name]:>12.2f} | {final_rate*100:.1f}%"
            )

        total_profit = sum(strategy_profits.values())
        print("-" * 70)
        print(f"{'TOTAL':<15} | {days:<15} | ${total_profit:>12.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian Strategy Selector using Thompson Sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python bayesian_strategy_selector.py init
    python bayesian_strategy_selector.py recommend
    python bayesian_strategy_selector.py record --strategy rsi_only --outcome win --pnl 55.00
    python bayesian_strategy_selector.py status
    python bayesian_strategy_selector.py simulate --days 30
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize from backtest data")
    init_parser.add_argument("--rsi-wins", type=int, default=421)
    init_parser.add_argument("--rsi-losses", type=int, default=193)
    init_parser.add_argument("--both-wins", type=int, default=96)
    init_parser.add_argument("--both-losses", type=int, default=32)
    init_parser.add_argument("--macd-wins", type=int, default=118)
    init_parser.add_argument("--macd-losses", type=int, default=67)

    # Recommend command
    recommend_parser = subparsers.add_parser("recommend", help="Get strategy recommendation for today")
    recommend_parser.add_argument(
        "--kelly", action="store_true", help="Use Kelly-adjusted position sizing"
    )

    # Record command
    record_parser = subparsers.add_parser("record", help="Record a trade result")
    record_parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["rsi_only", "both_agree", "macd_filter"],
    )
    record_parser.add_argument(
        "--outcome", type=str, required=True, choices=["win", "loss"]
    )
    record_parser.add_argument("--pnl", type=float, required=True)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show current posteriors and statistics")
    status_parser.add_argument(
        "--kelly", action="store_true", help="Show Kelly-adjusted position sizing"
    )

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Simulate adaptive trading")
    sim_parser.add_argument("--days", type=int, default=30)
    sim_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    selector = BayesianStrategySelector()

    if args.command == "init":
        selector.initialize_from_backtest(
            rsi_wins=args.rsi_wins,
            rsi_losses=args.rsi_losses,
            both_wins=args.both_wins,
            both_losses=args.both_losses,
            macd_wins=args.macd_wins,
            macd_losses=args.macd_losses,
        )

    elif args.command == "recommend":
        if not selector.strategies:
            print("No strategies initialized. Run 'init' first.")
            sys.exit(1)

        use_kelly = args.kelly
        best, samples = selector.thompson_sample(use_kelly=use_kelly)

        kelly_note = " (Kelly-adjusted)" if use_kelly else ""
        print(f"\nRecommendation{kelly_note}: {best}")
        print(f"\nThompson samples{kelly_note}:")
        for name, result in samples.items():
            marker = " <-- SELECTED" if name == best else ""
            contracts_str = f" ({result['contracts']:.2f}x)" if use_kelly else ""
            print(
                f"  {name}: {result['sampled_win_rate']*100:.1f}% win rate{contracts_str} "
                f"-> ${result['expected_profit']:.2f}/day{marker}"
            )

    elif args.command == "record":
        if not selector.strategies:
            print("No strategies initialized. Run 'init' first.")
            sys.exit(1)

        won = args.outcome == "win"
        selector.update_posterior(args.strategy, won, args.pnl)

        strat = selector.strategies[args.strategy]
        outcome_str = "WIN" if won else "LOSS"
        print(f"\nRecorded: {args.strategy} {outcome_str} ${args.pnl:+.2f}")
        print(
            f"Updated posterior: Beta({strat.alpha:.0f}, {strat.beta:.0f}) "
            f"-> {strat.mean*100:.1f}% win rate"
        )

    elif args.command == "status":
        use_kelly = args.kelly
        selector.print_status(use_kelly=use_kelly)

    elif args.command == "simulate":
        if not selector.strategies:
            print("No strategies initialized. Run 'init' first.")
            sys.exit(1)
        selector.simulate_adaptation(days=args.days, seed=args.seed)


if __name__ == "__main__":
    main()
