"""Position management module for backtest simulation.

Handles trade closing, P&L calculation, and settlement.
"""

from datetime import datetime
from typing import Optional

from app.models.schemas import SignalType

from .config import BacktestConfig
from .models import SimulatedTrade


class PositionManager:
    """
    Manages position lifecycle including closing trades and calculating P&L.

    Responsibilities:
    - Close trades (normal exit, stop loss, profit target, EOD)
    - Calculate settlement value at expiration
    - Apply slippage and commissions
    - Track P&L accurately with averaging down
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize position manager.

        Args:
            config: Backtest configuration with slippage/commission settings
        """
        self._config = config

    def calculate_settlement_value(self, trade: SimulatedTrade) -> float:
        """
        Calculate option settlement value at expiration.

        For calls: max(0, settlement_price - strike)
        For puts: max(0, strike - settlement_price)

        Uses trade.settlement_underlying (final underlying price at EOD).
        Works for both physical (SPY) and cash (SPXW) settlement.

        Args:
            trade: Trade with settlement_underlying set

        Returns:
            Intrinsic value per share (multiply by 100 for contract value)
        """
        settlement_price = trade.settlement_underlying
        if settlement_price <= 0:
            # Fallback to entry underlying if settlement not set
            settlement_price = trade.underlying_price

        if trade.signal_type == SignalType.BUY_CALL:
            intrinsic = settlement_price - trade.strike
        else:  # PUT
            intrinsic = trade.strike - settlement_price

        return max(0, intrinsic)

    def close_trade(
        self,
        trade: SimulatedTrade,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        contracts_to_close: Optional[int] = None,
    ) -> SimulatedTrade:
        """
        Close a trade and calculate P&L.

        Args:
            trade: Trade to close
            exit_time: Time of exit
            exit_price: Price at exit (before slippage)
            exit_reason: Reason for exit (profit_target, stop_loss, end_of_day, etc.)
            contracts_to_close: Number of contracts to close (default: all)

        Returns:
            Updated trade with exit details and P&L
        """
        # For EOD exits, use settlement value (intrinsic) instead of last traded price
        if exit_reason == "end_of_day":
            settlement_value = self.calculate_settlement_value(trade)
            exit_price = settlement_value
            # No slippage on settlement - it's automatic
        else:
            # Apply slippage on normal exits
            exit_price *= 1 - self._config.slippage_exit_pct

        if contracts_to_close is None:
            contracts_to_close = trade.total_contracts

        # Use avg_entry_price for P&L calc (handles averaging down)
        avg_price = trade.get_effective_entry_price()

        # Calculate P&L for contracts being closed (options are 100 shares per contract)
        gross_pnl = (exit_price - avg_price) * contracts_to_close * 100

        # Commission: entry for all entries (initial + add-ons), exit for all contracts
        total_entries = 1 + len(trade.add_on_entries)  # Initial entry + all add-on entries
        entry_commission = self._config.commission_per_contract * total_entries
        exit_commission = self._config.commission_per_contract * contracts_to_close
        net_pnl = gross_pnl - entry_commission - exit_commission

        # Update trade
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.pnl = net_pnl
        trade.remaining_contracts = 0

        return trade

    def add_to_position(
        self,
        trade: SimulatedTrade,
        add_price: float,
        add_contracts: int,
        add_time: datetime,
    ) -> SimulatedTrade:
        """
        Add contracts to an existing position (averaging down).

        Args:
            trade: Existing trade to add to
            add_price: Price of new entry
            add_contracts: Number of contracts to add
            add_time: Time of add-on entry

        Returns:
            Updated trade with new avg_entry_price and total_contracts
        """
        # Apply slippage to add-on entry
        add_price *= 1 + self._config.slippage_entry_pct

        # Record the add-on
        trade.add_on_entries.append((add_price, add_contracts, add_time))

        # Update totals
        old_total = trade.total_contracts
        new_total = old_total + add_contracts
        trade.total_contracts = new_total
        trade.remaining_contracts = new_total

        # Calculate new average entry price
        old_cost = trade.avg_entry_price * old_total if trade.avg_entry_price > 0 else trade.entry_price * old_total
        new_cost = add_price * add_contracts
        trade.avg_entry_price = (old_cost + new_cost) / new_total

        return trade

    def should_average_down(
        self,
        trade: SimulatedTrade,
        current_price: float,
    ) -> bool:
        """
        Check if position should be averaged down.

        Triggers when price drops by avg_down_trigger_pct from original entry.

        Args:
            trade: Current trade
            current_price: Current option price

        Returns:
            True if should add more contracts
        """
        # Only average down up to max_add_ons
        if len(trade.add_on_entries) >= self._config.max_add_ons:
            return False

        # Calculate drop threshold based on number of add-ons already made
        num_add_ons = len(trade.add_on_entries)
        trigger_drop_pct = self._config.avg_down_trigger_pct * (num_add_ons + 1)

        # Check if current price is below threshold from original entry
        threshold_price = trade.entry_price * (1 - trigger_drop_pct)

        return current_price <= threshold_price

    def calculate_unrealized_pnl(
        self,
        trade: SimulatedTrade,
        current_price: float,
    ) -> float:
        """
        Calculate unrealized P&L for an open position.

        Args:
            trade: Open trade
            current_price: Current option price

        Returns:
            Unrealized P&L in dollars
        """
        avg_price = trade.get_effective_entry_price()
        return (current_price - avg_price) * trade.total_contracts * 100

    def get_position_value(
        self,
        trade: SimulatedTrade,
        current_price: float,
    ) -> float:
        """
        Get current market value of position.

        Args:
            trade: Open trade
            current_price: Current option price

        Returns:
            Market value in dollars
        """
        return current_price * trade.total_contracts * 100
