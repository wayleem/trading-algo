"""
Monte Carlo simulation for backtesting analysis.

Implements bootstrap resampling of trade P&Ls to generate probability
distributions of strategy outcomes.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    n_simulations: int
    n_trades: int
    initial_capital: float

    # Distributions (arrays of length n_simulations)
    final_pnl_distribution: np.ndarray
    max_drawdown_distribution: np.ndarray
    sharpe_distribution: np.ndarray
    return_distribution: np.ndarray

    # Percentiles dict: {metric: {percentile: value}}
    percentiles: dict

    # Probabilities
    probability_of_profit: float
    probability_of_50pct_return: float
    probability_of_ruin: float  # < -50% drawdown at any point

    # Risk metrics
    var_5pct: float  # Value at Risk (5th percentile P&L)
    expected_shortfall_5pct: float  # Mean of worst 5% outcomes


class MonteCarloSimulator:
    """Bootstrap Monte Carlo simulator for trade P&L resampling."""

    def __init__(
        self,
        pnls: list[float],
        initial_capital: float = 10000.0,
    ):
        """
        Initialize simulator with trade P&Ls.

        Args:
            pnls: List of individual trade P&Ls (in dollars)
            initial_capital: Starting capital for equity curve calculations
        """
        self.pnls = np.array(pnls)
        self.initial_capital = initial_capital
        self.n_trades = len(pnls)

    def run_simulations(
        self,
        n_simulations: int = 10000,
        seed: Optional[int] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulations by bootstrap resampling trades.

        Args:
            n_simulations: Number of Monte Carlo iterations
            seed: Random seed for reproducibility

        Returns:
            MonteCarloResult with distributions and statistics
        """
        if seed is not None:
            np.random.seed(seed)

        # Pre-allocate result arrays
        final_pnls = np.zeros(n_simulations)
        max_drawdowns = np.zeros(n_simulations)
        sharpes = np.zeros(n_simulations)
        returns = np.zeros(n_simulations)
        ruin_count = 0

        for i in range(n_simulations):
            # Bootstrap resample with replacement
            resampled_pnls = np.random.choice(
                self.pnls, size=self.n_trades, replace=True
            )

            # Calculate metrics for this simulation
            metrics = self._calculate_simulation_metrics(resampled_pnls)

            final_pnls[i] = metrics["final_pnl"]
            max_drawdowns[i] = metrics["max_drawdown"]
            sharpes[i] = metrics["sharpe"]
            returns[i] = metrics["return_pct"]

            if metrics["max_drawdown"] > 50.0:  # > 50% drawdown
                ruin_count += 1

        # Calculate percentiles
        percentile_levels = [5, 25, 50, 75, 95]
        percentiles = {
            "pnl": {p: np.percentile(final_pnls, p) for p in percentile_levels},
            "max_drawdown": {
                p: np.percentile(max_drawdowns, p) for p in percentile_levels
            },
            "sharpe": {p: np.percentile(sharpes, p) for p in percentile_levels},
            "return": {p: np.percentile(returns, p) for p in percentile_levels},
        }

        # Calculate probabilities
        prob_profit = np.mean(final_pnls > 0) * 100
        prob_50pct_return = np.mean(returns > 50) * 100
        prob_ruin = (ruin_count / n_simulations) * 100

        # Risk metrics
        var_5pct = np.percentile(final_pnls, 5)
        worst_5pct_mask = final_pnls <= var_5pct
        expected_shortfall = np.mean(final_pnls[worst_5pct_mask])

        return MonteCarloResult(
            n_simulations=n_simulations,
            n_trades=self.n_trades,
            initial_capital=self.initial_capital,
            final_pnl_distribution=final_pnls,
            max_drawdown_distribution=max_drawdowns,
            sharpe_distribution=sharpes,
            return_distribution=returns,
            percentiles=percentiles,
            probability_of_profit=prob_profit,
            probability_of_50pct_return=prob_50pct_return,
            probability_of_ruin=prob_ruin,
            var_5pct=var_5pct,
            expected_shortfall_5pct=expected_shortfall,
        )

    def _calculate_simulation_metrics(self, pnls: np.ndarray) -> dict:
        """
        Calculate metrics for a single simulated P&L sequence.

        Args:
            pnls: Array of resampled P&Ls

        Returns:
            Dict with final_pnl, max_drawdown, sharpe, return_pct
        """
        # Build equity curve
        equity_curve = np.zeros(len(pnls) + 1)
        equity_curve[0] = self.initial_capital
        equity_curve[1:] = self.initial_capital + np.cumsum(pnls)

        # Final P&L
        final_pnl = equity_curve[-1] - self.initial_capital

        # Return percentage
        return_pct = (final_pnl / self.initial_capital) * 100

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        # Sharpe ratio (annualized assuming ~45 trades/year based on ~134 trades over 3 years)
        sharpe = self._calculate_sharpe(pnls)

        return {
            "final_pnl": final_pnl,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "return_pct": return_pct,
        }

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown percentage from equity curve.

        Args:
            equity_curve: Array of equity values over time

        Returns:
            Max drawdown as percentage (0-100)
        """
        # Running maximum
        running_max = np.maximum.accumulate(equity_curve)

        # Drawdown at each point
        drawdowns = (running_max - equity_curve) / running_max * 100

        return np.max(drawdowns)

    def _calculate_sharpe(self, pnls: np.ndarray, trades_per_year: float = 252) -> float:
        """
        Calculate annualized Sharpe ratio from trade P&Ls.

        Normalizes P&Ls by initial capital to get returns, then annualizes.

        Note: For Monte Carlo simulations where we're bootstrapping trades
        (not calendar days), we use trades_per_year estimate for annualization.
        This differs from the main backtest metrics which aggregate by calendar day.

        Args:
            pnls: Array of trade P&Ls
            trades_per_year: Estimated trading days per year (default 252)

        Returns:
            Annualized Sharpe ratio
        """
        if len(pnls) < 2:
            return 0.0

        # Normalize by capital to get returns
        returns = pnls / self.initial_capital

        if np.std(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        # Sharpe = (mean / std) * sqrt(trades_per_year)
        sharpe = (mean_return / std_return) * np.sqrt(trades_per_year)

        return sharpe


def print_monte_carlo_results(result: MonteCarloResult) -> None:
    """Print formatted Monte Carlo results."""
    print("=" * 60)
    print(f"MONTE CARLO SIMULATION RESULTS ({result.n_simulations:,} iterations)")
    print("=" * 60)
    print(f"Based on {result.n_trades} historical trades, resampled with replacement")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print()

    print("P&L Distribution:")
    for p in [5, 25, 50, 75, 95]:
        label = "(worst case)" if p == 5 else "(best case)" if p == 95 else ""
        print(f"  {p}th percentile:   ${result.percentiles['pnl'][p]:>10,.2f}  {label}")
    print()

    print("Return Distribution:")
    for p in [5, 25, 50, 75, 95]:
        print(f"  {p}th percentile:   {result.percentiles['return'][p]:>10.1f}%")
    print()

    print("Probabilities:")
    print(f"  Probability of Profit:      {result.probability_of_profit:>6.1f}%")
    print(f"  Probability of >50% Return: {result.probability_of_50pct_return:>6.1f}%")
    print(f"  Probability of Ruin (>50% DD): {result.probability_of_ruin:>6.1f}%")
    print()

    print("Max Drawdown Distribution:")
    for p in [5, 25, 50, 75, 95]:
        label = "(best case)" if p == 5 else "(worst case)" if p == 95 else ""
        print(
            f"  {p}th percentile:   {result.percentiles['max_drawdown'][p]:>10.1f}%  {label}"
        )
    print()

    print("Sharpe Ratio Distribution:")
    for p in [5, 25, 50, 75, 95]:
        print(f"  {p}th percentile:   {result.percentiles['sharpe'][p]:>10.2f}")
    print()

    print("Risk Metrics:")
    print(f"  Value at Risk (5%):        ${result.var_5pct:>10,.2f}")
    print(f"  Expected Shortfall (5%):   ${result.expected_shortfall_5pct:>10,.2f}")
    print("=" * 60)
