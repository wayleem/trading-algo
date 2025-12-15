"""
Position Management Service.

Monitors open positions and handles exit conditions including profit targets,
stop losses, and end-of-day 0DTE option expiration. Supports averaging down.
"""

from datetime import datetime
from typing import Optional
import logging
import pytz

from app.services.alpaca_client import AlpacaClient
from app.services.order_executor import OrderExecutor
from app.services.indicators import calculate_rsi_series, calculate_sma_series
from app.models.schemas import TradeRecord, SignalType
from app.core.config import settings

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


class PositionManager:
    """
    Monitor and manage open positions.

    Exit Logic (checked in order):
    1. Profit target hit (18% gain) - take profits first
    2. Stop loss hit (25% loss from avg entry) - cut losses
    3. End of day (forced close for 0DTE at 3:55 PM ET)

    Averaging Down:
    - Add 1 contract every -10% from original entry
    - Max 3 add-ons (4 contracts total)
    - Stop loss based on AVERAGE entry price
    """

    def __init__(
        self,
        alpaca_client: Optional[AlpacaClient] = None,
        order_executor: Optional[OrderExecutor] = None,
        eod_close_minutes: int = 5,
    ):
        self.client = alpaca_client or AlpacaClient()
        self.executor = order_executor or OrderExecutor(self.client)
        self.eod_close_minutes = eod_close_minutes  # Minutes before close to exit

    async def check_exits(self) -> list[TradeRecord]:
        """
        Check all open positions for exit conditions.

        Returns:
            List of closed trades
        """
        closed_trades = []

        for trade_id, trade in list(self.executor.open_trades.items()):
            exit_reason = await self._check_exit_conditions(trade)

            if exit_reason:
                current_price = await self._get_option_price(trade.option_symbol)
                if current_price is not None:
                    closed_trade = self.executor.close_trade(
                        trade_id=trade_id,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                    )
                    if closed_trade:
                        closed_trades.append(closed_trade)

                        # Actually close the position via Alpaca
                        try:
                            self.client.close_position(trade.option_symbol)
                        except Exception as e:
                            logger.error(f"Failed to close position via API: {e}")

        return closed_trades

    async def _check_exit_conditions(self, trade: TradeRecord) -> Optional[str]:
        """
        Check if any exit condition is met.

        Returns:
            Exit reason string or None
        """
        current_price = await self._get_option_price(trade.option_symbol)

        if current_price is None:
            return None

        # Use average entry price for stop loss calculation (handles averaging down)
        avg_entry = trade.get_effective_entry_price()
        dynamic_stop_loss = avg_entry * (1 - settings.stop_loss_pct)

        # 1. Check profit target (18% gain from avg entry)
        if current_price >= trade.profit_target:
            logger.info(
                f"Profit target hit for {trade.option_symbol}: "
                f"${current_price:.2f} >= ${trade.profit_target:.2f}"
            )
            return "profit_target"

        # 2. Check stop loss (25% loss from avg entry)
        if current_price <= dynamic_stop_loss:
            logger.info(
                f"Stop loss hit for {trade.option_symbol}: "
                f"${current_price:.2f} <= ${dynamic_stop_loss:.2f} (avg entry: ${avg_entry:.2f})"
            )
            return "stop_loss"

        # 3. Check end of day (5 minutes before close)
        if self._is_near_market_close():
            logger.info(f"End of day exit for {trade.option_symbol}")
            return "end_of_day"

        return None

    async def _get_option_price(self, option_symbol: str) -> Optional[float]:
        """
        Get current price for an option.

        In production, this would query the options quote.
        For paper trading, we can use the position's market value.
        """
        try:
            position = self.client.get_position(option_symbol)
            if position:
                return float(position.current_price)
        except Exception as e:
            logger.warning(f"Could not get price for {option_symbol}: {e}")

        return None

    async def _get_current_rsi_and_sma(self, symbol: str) -> Optional[tuple[float, float]]:
        """
        Get current RSI and RSI SMA values for the underlying symbol.

        Fetches recent bars and calculates RSI and its SMA.
        Returns tuple of (rsi, sma) or None if calculation fails.
        """
        try:
            # Need enough bars for RSI + SMA calculation
            bars = await self.client.get_stock_bars(
                symbol=symbol,
                timeframe="1Min",
                limit=settings.rsi_period + settings.rsi_sma_period + 10,
            )

            if len(bars) < settings.rsi_period + settings.rsi_sma_period:
                logger.warning(f"Not enough bars for RSI/SMA calculation: {len(bars)}")
                return None

            closes = [bar["close"] for bar in bars]
            rsi_values = calculate_rsi_series(closes, settings.rsi_period)
            sma_values = calculate_sma_series(rsi_values, settings.rsi_sma_period)

            # Return the most recent RSI and SMA values
            if rsi_values and sma_values and rsi_values[-1] is not None and sma_values[-1] is not None:
                return (rsi_values[-1], sma_values[-1])

        except Exception as e:
            logger.warning(f"Could not calculate RSI/SMA for {symbol}: {e}")

        return None

    def _is_near_market_close(self) -> bool:
        """Check if we're within buffer minutes of market close (4:00 PM ET)."""
        now = datetime.now(ET)

        # Market close is 4:00 PM ET
        close_hour = 16
        close_minute = 0

        close_time = now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
        time_to_close = (close_time - now).total_seconds() / 60

        # Exit if within buffer minutes of close
        return 0 < time_to_close <= self.eod_close_minutes

    def is_market_hours(self) -> bool:
        """Check if currently within market hours (9:30 AM - 4:00 PM ET)."""
        now = datetime.now(ET)

        # Skip weekends
        if now.weekday() >= 5:
            return False

        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close

    async def force_close_all(self, reason: str = "manual") -> list[TradeRecord]:
        """
        Force close all open positions.

        Args:
            reason: Reason for closing

        Returns:
            List of closed trades
        """
        closed_trades = []

        for trade_id, trade in list(self.executor.open_trades.items()):
            current_price = await self._get_option_price(trade.option_symbol)

            if current_price is None:
                current_price = trade.entry_price  # Use entry if can't get current

            closed_trade = self.executor.close_trade(
                trade_id=trade_id,
                exit_price=current_price,
                exit_reason=reason,
            )

            if closed_trade:
                closed_trades.append(closed_trade)

                try:
                    self.client.close_position(trade.option_symbol)
                except Exception as e:
                    logger.error(f"Failed to close position: {e}")

        return closed_trades

    def should_average_down(self, trade: TradeRecord, current_price: float) -> bool:
        """
        Check if position should be averaged down.

        Triggers when price drops by avg_down_trigger_pct from original entry.
        Each add-on requires an additional drop (e.g., -10%, -20%, -30%).

        Args:
            trade: Current trade
            current_price: Current option price

        Returns:
            True if should add more contracts
        """
        # Only average down up to max_add_ons
        num_add_ons = len(trade.add_on_entries)
        if num_add_ons >= settings.max_add_ons:
            return False

        # Calculate drop threshold based on number of add-ons already made
        # First add-on at -10%, second at -20%, third at -30%
        trigger_drop_pct = settings.avg_down_trigger_pct * (num_add_ons + 1)

        # Check if current price is below threshold from ORIGINAL entry
        threshold_price = trade.entry_price * (1 - trigger_drop_pct)

        return current_price <= threshold_price

    async def check_and_average_down(self, trade: TradeRecord) -> Optional[TradeRecord]:
        """
        Check if we should average down and execute if conditions are met.

        Args:
            trade: Trade to check for averaging down

        Returns:
            Updated trade if averaged down, None otherwise
        """
        current_price = await self._get_option_price(trade.option_symbol)

        if current_price is None:
            return None

        if not self.should_average_down(trade, current_price):
            return None

        # Execute averaging down - buy 1 more contract
        logger.info(
            f"Averaging down {trade.option_symbol}: price ${current_price:.2f} "
            f"(entry ${trade.entry_price:.2f}, add-on #{len(trade.add_on_entries) + 1})"
        )

        try:
            # Use limit order strategy for add-on
            # Add same number of contracts as initial position (matches backtest)
            add_contracts = settings.contracts_per_trade
            order, fill_price = self.executor.execute_with_limit_walkup(
                contract_symbol=trade.option_symbol,
                side="buy",
                qty=add_contracts,
                max_attempts=3,
                wait_seconds=1.0,
                fallback_to_market=True,
            )

            if order is None or fill_price == 0:
                logger.warning(f"Failed to average down {trade.option_symbol}")
                return None

            # Apply slippage
            fill_price *= 1 + settings.slippage_pct

            # Update trade with new average
            add_time = datetime.now()
            trade.add_on_entries.append((fill_price, add_contracts, add_time))

            # Calculate new average entry price
            old_total = trade.total_contracts
            new_total = old_total + add_contracts
            old_cost = trade.get_effective_entry_price() * old_total
            new_cost = fill_price * add_contracts
            trade.avg_entry_price = (old_cost + new_cost) / new_total
            trade.total_contracts = new_total
            trade.quantity = new_total

            # Update profit target based on new average
            trade.profit_target = trade.avg_entry_price * (1 + settings.profit_target_pct)
            trade.stop_loss = trade.avg_entry_price * (1 - settings.stop_loss_pct)

            logger.info(
                f"Averaged down {trade.option_symbol}: "
                f"now {new_total} contracts @ avg ${trade.avg_entry_price:.2f}, "
                f"new PT=${trade.profit_target:.2f}, SL=${trade.stop_loss:.2f}"
            )

            return trade

        except Exception as e:
            logger.error(f"Error averaging down {trade.option_symbol}: {e}")
            return None
