"""Performance metrics calculation module for backtest.

Calculates win rate, Sharpe ratio, max drawdown, profit factor, etc.
"""

import math
from collections import defaultdict
from datetime import date, datetime
from typing import Dict, List, Tuple

from app.models.schemas import BacktestMetrics, BacktestResult, BacktestTrade

from .config import BacktestConfig
from .models import SimulatedTrade


class MetricsCalculator:
    """
    Calculates performance metrics from backtest trades.

    Metrics include:
    - Trade-level: win rate, total P&L, avg P&L
    - Contract-level: win rate per entry (including add-ons)
    - Risk-adjusted: Sharpe ratio, max drawdown, profit factor
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize metrics calculator.

        Args:
            config: Backtest configuration with initial capital
        """
        self._config = config

    def calculate_metrics(self, trades: List[SimulatedTrade]) -> BacktestMetrics:
        """
        Calculate all performance metrics from trades.

        Args:
            trades: List of completed trades

        Returns:
            BacktestMetrics with all calculated values
        """
        if not trades:
            return self._empty_metrics()

        pnls = [t.pnl for t in trades if t.pnl is not None]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls) if pnls else 0

        # Win rate (trade-level)
        win_rate = len(winning) / len(pnls) if pnls else 0

        # Contract-level metrics
        total_contracts, winning_contracts, losing_contracts = self._count_contract_wins(trades)
        contract_win_rate = winning_contracts / total_contracts if total_contracts > 0 else 0.0

        # Max drawdown
        max_dd = self._calculate_max_drawdown(pnls)

        # Profit factor
        profit_factor = self._calculate_profit_factor(winning, losing)

        # Sharpe ratio (using daily returns for proper annualization)
        sharpe = self._calculate_sharpe_ratio(trades)

        return BacktestMetrics(
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            total_contracts=total_contracts,
            winning_contracts=winning_contracts,
            losing_contracts=losing_contracts,
            contract_win_rate=contract_win_rate,
        )

    def _count_contract_wins(
        self, trades: List[SimulatedTrade]
    ) -> Tuple[int, int, int]:
        """
        Count wins/losses at contract level (each entry counted independently).

        Args:
            trades: List of completed trades

        Returns:
            (total_contracts, winning_contracts, losing_contracts)
        """
        total_contracts = 0
        winning_contracts = 0
        losing_contracts = 0

        for trade in trades:
            if trade.pnl is None or trade.exit_price is None:
                continue

            # Initial entry
            total_contracts += 1
            if trade.exit_price >= trade.entry_price:
                winning_contracts += 1
            else:
                losing_contracts += 1

            # Each add-on entry
            for addon_price, addon_contracts, addon_time in trade.add_on_entries:
                total_contracts += 1
                if trade.exit_price >= addon_price:
                    winning_contracts += 1
                else:
                    losing_contracts += 1

        return total_contracts, winning_contracts, losing_contracts

    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """
        Calculate maximum drawdown from P&L series.

        Args:
            pnls: List of trade P&Ls

        Returns:
            Maximum drawdown as percentage (0.0 to 1.0)
        """
        equity = self._config.initial_capital
        peak = equity
        max_dd = 0

        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_profit_factor(
        self, winning: List[float], losing: List[float]
    ) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            winning: List of winning P&Ls
            losing: List of losing P&Ls

        Returns:
            Profit factor (higher is better, 1.0 is breakeven)
        """
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 1
        return gross_profit / gross_loss if gross_loss > 0 else 0

    def _calculate_sharpe_ratio(self, trades: List[SimulatedTrade]) -> float:
        """
        Calculate annualized Sharpe ratio using daily returns.

        Aggregates trade P&Ls by day to avoid frequency inflation.
        Uses sample standard deviation (n-1 denominator) for statistical correctness.

        Args:
            trades: List of completed trades

        Returns:
            Annualized Sharpe ratio
        """
        if not trades:
            return 0.0

        # Aggregate P&Ls by day
        daily_pnls: Dict[date, float] = defaultdict(float)
        for trade in trades:
            if trade.exit_time and trade.pnl is not None:
                day = trade.exit_time.date()
                daily_pnls[day] += trade.pnl

        if len(daily_pnls) <= 1:
            return 0.0

        # Convert to daily returns (relative to capital)
        daily_returns = [pnl / self._config.initial_capital for pnl in daily_pnls.values()]

        n = len(daily_returns)
        avg_return = sum(daily_returns) / n

        # Sample standard deviation (n-1 denominator, Bessel's correction)
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / (n - 1)
        std_return = math.sqrt(variance)

        if std_return <= 0:
            return 0.0

        # Annualize with sqrt(252) - correct for daily returns
        return (avg_return / std_return) * math.sqrt(252)

    def build_equity_curve(
        self, trades: List[SimulatedTrade]
    ) -> List[Tuple[datetime, float]]:
        """
        Build equity curve from trades.

        Args:
            trades: List of completed trades

        Returns:
            List of (timestamp, equity) tuples
        """
        curve = []
        equity = self._config.initial_capital

        for trade in trades:
            if trade.exit_time and trade.pnl is not None:
                equity += trade.pnl
                curve.append((trade.exit_time, equity))

        return curve

    def to_backtest_trade(self, trade: SimulatedTrade) -> BacktestTrade:
        """
        Convert SimulatedTrade to BacktestTrade schema.

        Args:
            trade: Internal trade representation

        Returns:
            API-compatible trade schema
        """
        pnl_pct = (
            (trade.exit_price - trade.entry_price) / trade.entry_price
            if trade.entry_price > 0 and trade.exit_price
            else 0
        )

        return BacktestTrade(
            entry_date=trade.entry_time,
            exit_date=trade.exit_time or trade.entry_time,
            signal_type=trade.signal_type,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price if trade.exit_price is not None else trade.entry_price,
            pnl_dollars=trade.pnl or 0,
            pnl_percent=pnl_pct,
            exit_reason=trade.exit_reason or "unknown",
            entry_rsi=trade.entry_rsi,
            rsi_history=trade.rsi_history,
            timeframe=trade.timeframe,
            strike=trade.strike,
            settlement_underlying=trade.settlement_underlying,
        )

    def build_result(
        self,
        trades: List[SimulatedTrade],
    ) -> BacktestResult:
        """
        Build complete backtest result with trades, metrics, and equity curve.

        Args:
            trades: List of completed trades

        Returns:
            Complete BacktestResult
        """
        metrics = self.calculate_metrics(trades)
        equity_curve = self.build_equity_curve(trades)
        backtest_trades = [self.to_backtest_trade(t) for t in trades]

        return BacktestResult(
            trades=backtest_trades,
            metrics=metrics,
            equity_curve=equity_curve,
        )

    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics when no trades."""
        return BacktestMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_pnl=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            total_contracts=0,
            winning_contracts=0,
            losing_contracts=0,
            contract_win_rate=0.0,
        )

    def empty_result(self) -> BacktestResult:
        """Return empty result when no data available."""
        return BacktestResult(
            trades=[],
            metrics=self._empty_metrics(),
            equity_curve=[],
        )


def calculate_holding_periods(trades: List[SimulatedTrade]) -> dict:
    """
    Calculate average holding periods for winners and losers.

    Args:
        trades: List of completed trades

    Returns:
        Dict with 'winners_avg_minutes' and 'losers_avg_minutes'
    """
    winner_minutes = []
    loser_minutes = []

    for trade in trades:
        if trade.exit_time and trade.pnl is not None:
            duration = (trade.exit_time - trade.entry_time).total_seconds() / 60

            if trade.pnl > 0:
                winner_minutes.append(duration)
            else:
                loser_minutes.append(duration)

    return {
        "winners_avg_minutes": sum(winner_minutes) / len(winner_minutes) if winner_minutes else 0,
        "losers_avg_minutes": sum(loser_minutes) / len(loser_minutes) if loser_minutes else 0,
        "winners_count": len(winner_minutes),
        "losers_count": len(loser_minutes),
    }


def calculate_exit_breakdown(trades: List[SimulatedTrade]) -> dict:
    """
    Calculate P&L breakdown by exit reason.

    Args:
        trades: List of completed trades

    Returns:
        Dict mapping exit_reason to {'count': int, 'pnl': float}
    """
    breakdown = {}

    for trade in trades:
        if trade.exit_reason and trade.pnl is not None:
            reason = trade.exit_reason
            if reason not in breakdown:
                breakdown[reason] = {"count": 0, "pnl": 0.0}
            breakdown[reason]["count"] += 1
            breakdown[reason]["pnl"] += trade.pnl

    return breakdown
