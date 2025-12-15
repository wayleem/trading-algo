"""
Results storage and comparison utilities.

Provides:
- StrategyResult: Serializable result from a strategy backtest
- ResultsManager: Save/load results to JSON files
- print_comparison_table: Format comparison as ASCII table
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """
    Serializable result from a strategy backtest.

    Contains key metrics for comparison across strategies.
    """

    strategy_name: str
    start_date: str
    end_date: str
    config: Dict[str, Any]

    # Core metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float

    # Additional metrics
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Metadata
    run_timestamp: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyResult":
        """Deserialize from dictionary."""
        return cls(**data)

    @classmethod
    def from_backtest_result(
        cls,
        strategy_name: str,
        start_date: date,
        end_date: date,
        config: Dict[str, Any],
        result,
    ) -> "StrategyResult":
        """
        Create StrategyResult from BacktestResult.

        Args:
            strategy_name: Name of the strategy
            start_date: Backtest start date
            end_date: Backtest end date
            config: Strategy configuration dict
            result: BacktestResult from backtest

        Returns:
            StrategyResult instance
        """
        metrics = result.metrics

        # Calculate additional metrics from trades
        wins = [t.pnl_dollars for t in result.trades if t.pnl_dollars and t.pnl_dollars > 0]
        losses = [t.pnl_dollars for t in result.trades if t.pnl_dollars and t.pnl_dollars < 0]

        return cls(
            strategy_name=strategy_name,
            start_date=str(start_date),
            end_date=str(end_date),
            config=config,
            total_trades=metrics.total_trades,
            winning_trades=metrics.winning_trades,
            losing_trades=metrics.losing_trades,
            win_rate=metrics.win_rate,
            total_pnl=metrics.total_pnl,
            avg_pnl=metrics.avg_pnl,
            max_drawdown=metrics.max_drawdown,
            sharpe_ratio=metrics.sharpe_ratio,
            profit_factor=metrics.profit_factor,
            largest_win=max(wins) if wins else 0.0,
            largest_loss=min(losses) if losses else 0.0,
            avg_win=sum(wins) / len(wins) if wins else 0.0,
            avg_loss=sum(losses) / len(losses) if losses else 0.0,
            run_timestamp=datetime.now().isoformat(),
        )


class ResultsManager:
    """
    Manages storage and retrieval of strategy results.

    Results are stored as JSON files in:
    results/{strategy_name}/{start_date}_{end_date}.json
    """

    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize results manager.

        Args:
            results_dir: Directory for results (default: backend/results)
        """
        if results_dir is None:
            results_dir = Path(__file__).parent.parent / "results"
        self._results_dir = results_dir
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, result: StrategyResult) -> Path:
        """
        Save strategy result to JSON file.

        Args:
            result: StrategyResult to save

        Returns:
            Path to saved file
        """
        strategy_dir = self._results_dir / result.strategy_name
        strategy_dir.mkdir(exist_ok=True)

        filename = f"{result.start_date}_{result.end_date}.json"
        filepath = strategy_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved result to {filepath}")
        return filepath

    def load_result(
        self,
        strategy_name: str,
        start_date: str,
        end_date: str,
    ) -> Optional[StrategyResult]:
        """
        Load a specific result.

        Args:
            strategy_name: Strategy name
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)

        Returns:
            StrategyResult or None if not found
        """
        filepath = self._results_dir / strategy_name / f"{start_date}_{end_date}.json"

        if not filepath.exists():
            return None

        with open(filepath) as f:
            data = json.load(f)

        return StrategyResult.from_dict(data)

    def list_results(
        self,
        strategy_name: Optional[str] = None,
    ) -> List[StrategyResult]:
        """
        List all results, optionally filtered by strategy.

        Args:
            strategy_name: Optional strategy name filter

        Returns:
            List of StrategyResult instances
        """
        results = []

        if strategy_name:
            dirs = [self._results_dir / strategy_name]
        else:
            dirs = [d for d in self._results_dir.iterdir() if d.is_dir()]

        for strategy_dir in dirs:
            if not strategy_dir.exists():
                continue

            for result_file in strategy_dir.glob("*.json"):
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    results.append(StrategyResult.from_dict(data))
                except Exception as e:
                    logger.warning(f"Error loading {result_file}: {e}")

        return results

    def compare_strategies(
        self,
        start_date: str,
        end_date: str,
        strategy_names: Optional[List[str]] = None,
    ) -> Dict[str, StrategyResult]:
        """
        Load and compare results for multiple strategies.

        Args:
            start_date: Start date string
            end_date: End date string
            strategy_names: Optional list of strategy names (all if None)

        Returns:
            Dict mapping strategy name to StrategyResult
        """
        from strategies.registry import StrategyRegistry

        names = strategy_names or StrategyRegistry.list_strategies()
        results = {}

        for name in names:
            result = self.load_result(name, start_date, end_date)
            if result:
                results[name] = result

        return results

    def delete_result(
        self,
        strategy_name: str,
        start_date: str,
        end_date: str,
    ) -> bool:
        """
        Delete a specific result file.

        Returns:
            True if deleted, False if not found
        """
        filepath = self._results_dir / strategy_name / f"{start_date}_{end_date}.json"

        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted {filepath}")
            return True
        return False


def print_comparison_table(results: Dict[str, StrategyResult]) -> None:
    """
    Print formatted comparison table to stdout.

    Args:
        results: Dict mapping strategy name to StrategyResult
    """
    if not results:
        print("No results to compare.")
        return

    print("\n" + "=" * 110)
    print(" Strategy Comparison Results")
    print("=" * 110)

    # Header
    print(
        f"{'Strategy':<20} "
        f"{'Trades':>7} "
        f"{'Win%':>7} "
        f"{'Total P&L':>12} "
        f"{'Avg P&L':>10} "
        f"{'Drawdown':>10} "
        f"{'Sharpe':>8} "
        f"{'PF':>6}"
    )
    print("-" * 110)

    # Sort by total P&L descending
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].total_pnl,
        reverse=True,
    )

    for name, r in sorted_results:
        pnl_color = "" if r.total_pnl >= 0 else ""
        print(
            f"{name:<20} "
            f"{r.total_trades:>7} "
            f"{r.win_rate * 100:>6.1f}% "
            f"${r.total_pnl:>10,.2f} "
            f"${r.avg_pnl:>9,.2f} "
            f"{r.max_drawdown * 100:>9.1f}% "
            f"{r.sharpe_ratio:>8.2f} "
            f"{r.profit_factor:>6.2f}"
        )

    print("-" * 110)

    # Summary
    if len(results) > 1:
        total_pnls = [r.total_pnl for r in results.values()]
        avg_sharpe = sum(r.sharpe_ratio for r in results.values()) / len(results)
        print(f"\nBest P&L: {max(total_pnls):,.2f} | Worst P&L: {min(total_pnls):,.2f} | Avg Sharpe: {avg_sharpe:.2f}")


def format_comparison_markdown(results: Dict[str, StrategyResult]) -> str:
    """
    Format comparison as markdown table.

    Args:
        results: Dict mapping strategy name to StrategyResult

    Returns:
        Markdown formatted string
    """
    if not results:
        return "No results to compare."

    lines = [
        "## Strategy Comparison Results",
        "",
        "| Strategy | Trades | Win% | Total P&L | Avg P&L | Drawdown | Sharpe | PF |",
        "|----------|--------|------|-----------|---------|----------|--------|-----|",
    ]

    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].total_pnl,
        reverse=True,
    )

    for name, r in sorted_results:
        lines.append(
            f"| {name} | {r.total_trades} | {r.win_rate * 100:.1f}% | "
            f"${r.total_pnl:,.2f} | ${r.avg_pnl:,.2f} | "
            f"{r.max_drawdown * 100:.1f}% | {r.sharpe_ratio:.2f} | {r.profit_factor:.2f} |"
        )

    return "\n".join(lines)
