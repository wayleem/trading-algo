"""
Order Execution Service.

Handles trade execution via Alpaca API with smart limit order strategy.
Implements mid-price walkup for better fills and manages trade records.
"""

from datetime import date, datetime
from typing import Optional
import uuid
import logging
import time

from app.models.schemas import SignalType, TradingSignal, TradeRecord, TradeStatus
from app.services.alpaca_client import AlpacaClient
from app.core.config import settings

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Execute trades via Alpaca API.

    Handles:
    - ATM option selection for 0DTE
    - Smart limit order execution with mid-price walkup
    - Trade record management
    """

    def __init__(self, alpaca_client: Optional[AlpacaClient] = None):
        self.client = alpaca_client or AlpacaClient()
        self.open_trades: dict[str, TradeRecord] = {}
        self.closed_trades: list[TradeRecord] = []

    def execute_with_limit_walkup(
        self,
        contract_symbol: str,
        side: str,
        qty: int,
        max_attempts: int = 5,
        wait_seconds: float = 2.0,
        tick_size: float = 0.01,
        fallback_to_market: bool = True,
    ) -> tuple[Optional[object], float]:
        """
        Execute order using mid-price limit with walk-up strategy.

        Strategy:
        1. Get current bid/ask quote
        2. Place limit order at midpoint
        3. Wait for fill
        4. If not filled, cancel and re-submit 1 tick closer to market
        5. Repeat until filled or max attempts reached
        6. Optionally fall back to market order

        Args:
            contract_symbol: Option contract symbol
            side: "buy" or "sell"
            qty: Number of contracts
            max_attempts: Maximum limit order attempts before fallback
            wait_seconds: Seconds to wait for each limit order
            tick_size: Price increment for walk-up
            fallback_to_market: If True, use market order as final fallback

        Returns:
            Tuple of (order, fill_price) or (None, 0) if failed
        """
        # Get current quote
        quote = self.client.get_option_quote(contract_symbol)

        if not quote:
            logger.warning(f"Could not get quote for {contract_symbol}")
            if fallback_to_market:
                logger.info("Falling back to market order")
                return self._execute_market_order(contract_symbol, side, qty)
            return None, 0.0

        bid, ask, mid = quote["bid"], quote["ask"], quote["mid"]
        spread = quote["spread"]

        logger.info(
            f"Quote for {contract_symbol}: bid=${bid:.2f}, ask=${ask:.2f}, "
            f"mid=${mid:.2f}, spread=${spread:.2f} ({quote['spread_pct']:.1f}%)"
        )

        # Start at midpoint
        limit_price = mid

        for attempt in range(max_attempts):
            # Round to tick size (options typically trade in $0.01 increments)
            limit_price = round(limit_price / tick_size) * tick_size

            logger.info(
                f"Attempt {attempt + 1}/{max_attempts}: {side} limit @ ${limit_price:.2f}"
            )

            try:
                order = self.client.submit_option_order(
                    contract_symbol=contract_symbol,
                    side=side,
                    qty=qty,
                    order_type="limit",
                    limit_price=limit_price,
                )

                # Wait for fill
                time.sleep(wait_seconds)

                # Check if filled
                order = self.client.trading_client.get_order_by_id(order.id)

                if order.filled_avg_price and float(order.filled_avg_price) > 0:
                    fill_price = float(order.filled_avg_price)
                    savings = ask - fill_price if side == "buy" else fill_price - bid
                    logger.info(
                        f"Filled at ${fill_price:.2f} "
                        f"(saved ${savings:.2f} vs {'ask' if side == 'buy' else 'bid'})"
                    )
                    return order, fill_price

                # Not filled - cancel and try again
                logger.info(f"Not filled, cancelling order {order.id}")
                try:
                    self.client.cancel_order(str(order.id))
                except Exception as e:
                    logger.warning(f"Cancel failed: {e}")

            except Exception as e:
                logger.error(f"Limit order attempt failed: {e}")

            # Walk price toward market
            if side == "buy":
                limit_price += tick_size  # Move up toward ask
            else:
                limit_price -= tick_size  # Move down toward bid

            # Don't exceed market price
            if side == "buy" and limit_price >= ask:
                logger.info("Limit price reached ask, stopping walkup")
                break
            elif side == "sell" and limit_price <= bid:
                logger.info("Limit price reached bid, stopping walkup")
                break

        # Fallback to market order
        if fallback_to_market:
            logger.warning(
                f"Limit orders not filled after {max_attempts} attempts, using market order"
            )
            return self._execute_market_order(contract_symbol, side, qty)

        logger.warning(f"Limit orders not filled, aborting")
        return None, 0.0

    def _execute_market_order(
        self, contract_symbol: str, side: str, qty: int
    ) -> tuple[Optional[object], float]:
        """Execute a market order and poll for fill price."""
        try:
            order = self.client.submit_option_order(
                contract_symbol=contract_symbol,
                side=side,
                qty=qty,
                order_type="market",
            )

            # Poll for fill
            for attempt in range(10):
                try:
                    order = self.client.trading_client.get_order_by_id(order.id)
                    if order.filled_avg_price and float(order.filled_avg_price) > 0:
                        fill_price = float(order.filled_avg_price)
                        logger.info(f"Market order filled at ${fill_price:.2f}")
                        return order, fill_price
                except Exception as e:
                    logger.warning(f"Error polling: {e}")
                time.sleep(1)

            logger.warning("Market order not filled after 10s")
            try:
                self.client.cancel_order(str(order.id))
            except:
                pass
            return None, 0.0

        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return None, 0.0

    def execute_signal(self, signal: TradingSignal) -> Optional[TradeRecord]:
        """
        Execute a trading signal using smart limit order strategy.

        Steps:
        1. Get today's expiration (0DTE)
        2. Find ATM strike with offset
        3. Execute with mid-price limit walkup strategy
        4. Record trade with profit target and stop loss

        Args:
            signal: TradingSignal to execute

        Returns:
            TradeRecord if trade was placed, None otherwise
        """
        if signal.signal_type == SignalType.NO_SIGNAL:
            return None

        # Determine option type
        option_type = "call" if signal.signal_type == SignalType.BUY_CALL else "put"

        # Get 0DTE expiration (today)
        today = date.today()

        try:
            # Find ATM option
            contract = self.client.get_atm_option(
                underlying_symbol=settings.symbol,
                option_type=option_type,
                expiration_date=today,
                current_price=signal.close_price,
                strike_offset=settings.strike_offset,
            )

            if not contract:
                logger.warning(f"No 0DTE {option_type} contract found for {settings.symbol}")
                return None

            logger.info(
                f"Found ${settings.strike_offset} OTM {option_type}: {contract['symbol']} "
                f"strike={contract['strike']} for price={signal.close_price}"
            )

            # Execute with smart limit order strategy
            order, entry_price = self.execute_with_limit_walkup(
                contract_symbol=contract["symbol"],
                side="buy",
                qty=settings.contracts_per_trade,
                max_attempts=5,
                wait_seconds=2.0,
                tick_size=0.01,
                fallback_to_market=True,
            )

            if entry_price == 0 or order is None:
                logger.warning("Failed to execute order")
                return None

            # Calculate exit levels
            profit_target = entry_price * (1 + settings.profit_target_pct)
            stop_loss = entry_price * (1 - settings.stop_loss_pct)

            # Record trade
            trade = TradeRecord(
                id=str(uuid.uuid4()),
                timestamp=signal.timestamp,
                signal_type=signal.signal_type,
                option_symbol=contract["symbol"],
                underlying_price=signal.close_price,
                strike=contract["strike"],
                option_type=option_type,
                quantity=settings.contracts_per_trade,
                entry_price=entry_price,
                profit_target=profit_target,
                stop_loss=stop_loss,
                status=TradeStatus.OPEN,
            )

            self.open_trades[trade.id] = trade

            logger.info(
                f"Trade opened: {trade.option_symbol} @ ${entry_price:.2f}, "
                f"target=${profit_target:.2f}, stop=${stop_loss:.2f}"
            )

            return trade

        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            return None

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
    ) -> Optional[TradeRecord]:
        """
        Close an open trade.

        Args:
            trade_id: ID of the trade to close
            exit_price: Exit price
            exit_reason: Reason for closing ("profit_target", "stop_loss", "end_of_day")

        Returns:
            Updated TradeRecord
        """
        if trade_id not in self.open_trades:
            logger.warning(f"Trade {trade_id} not found in open trades")
            return None

        trade = self.open_trades[trade_id]

        # Calculate P&L using average entry price (handles averaging down)
        avg_entry = trade.get_effective_entry_price()
        pnl = (exit_price - avg_entry) * trade.total_contracts * 100  # Options are 100 shares

        # Determine status based on exit reason
        if exit_reason == "profit_target":
            status = TradeStatus.CLOSED_PROFIT
        elif exit_reason == "stop_loss":
            status = TradeStatus.CLOSED_LOSS
        elif exit_reason == "timeout":
            status = TradeStatus.CLOSED_TIMEOUT
        elif exit_reason == "rsi_convergence":
            status = TradeStatus.CLOSED_RSI_CONVERGENCE
        else:
            status = TradeStatus.CLOSED_EOD

        # Update trade record
        trade.exit_price = exit_price
        trade.exit_timestamp = datetime.now()
        trade.pnl = pnl
        trade.status = status

        # Move to closed trades
        del self.open_trades[trade_id]
        self.closed_trades.append(trade)

        logger.info(
            f"Trade closed: {trade.option_symbol} @ ${exit_price:.2f}, "
            f"P&L=${pnl:.2f}, reason={exit_reason}, "
            f"contracts={trade.total_contracts}, avg_entry=${avg_entry:.2f}"
        )

        return trade

    def get_open_trades(self) -> list[TradeRecord]:
        """Get all open trades."""
        return list(self.open_trades.values())

    def get_closed_trades(self) -> list[TradeRecord]:
        """Get all closed trades."""
        return self.closed_trades

    def get_total_pnl(self) -> float:
        """Get total P&L from closed trades."""
        return sum(t.pnl or 0 for t in self.closed_trades)

    def reset_daily(self):
        """Reset for new trading day."""
        self.open_trades.clear()
        self.closed_trades.clear()
