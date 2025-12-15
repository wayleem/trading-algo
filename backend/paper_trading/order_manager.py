"""
Order Manager for Paper Trading.

Handles order submission, bracket orders (PT/SL), and order state tracking.
Alpaca doesn't support true OCO for options, so we manage manually.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from enum import Enum

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderPurpose(Enum):
    """Purpose of the order in our strategy."""
    ENTRY = "entry"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    FORCE_EXIT = "force_exit"


@dataclass
class ManagedOrder:
    """
    An order we're tracking.
    """
    order_id: str
    position_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    order_type: OrderType
    purpose: OrderPurpose
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    status: str = "pending"
    alpaca_order: Optional[object] = None


class OrderManager:
    """
    Manages order submission and tracking for paper trading.

    Since Alpaca doesn't support OCO (one-cancels-other) for options,
    we implement manual bracket order management:
    1. Submit market order for entry
    2. After fill, submit limit order for profit target
    3. Monitor price and submit market order for stop loss if triggered
    """

    def __init__(
        self,
        trading_client: TradingClient,
        profit_target_pct: float = 0.30,
        stop_loss_pct: float = 0.20,  # 20% validated (NOT 6%)
    ):
        """
        Initialize order manager.

        Args:
            trading_client: Alpaca TradingClient instance
            profit_target_pct: Profit target percentage (0.30 = 30%)
            stop_loss_pct: Stop loss percentage (0.06 = 6%)
        """
        self.trading_client = trading_client
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct

        # Order tracking
        self.pending_orders: List[ManagedOrder] = []
        self.filled_orders: List[ManagedOrder] = []
        self.cancelled_orders: List[ManagedOrder] = []

    def submit_entry_order(
        self,
        position_id: str,
        symbol: str,
        quantity: int,
        use_market_order: bool = True,
        limit_price: Optional[float] = None,
    ) -> ManagedOrder:
        """
        Submit entry order (buy to open).

        Args:
            position_id: Associated position ID
            symbol: Option symbol
            quantity: Number of contracts
            use_market_order: Use market order (True) or limit (False)
            limit_price: Limit price if using limit order

        Returns:
            ManagedOrder object
        """
        if use_market_order:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order_type = OrderType.MARKET
        else:
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            )
            order_type = OrderType.LIMIT

        try:
            alpaca_order = self.trading_client.submit_order(order_request)

            managed_order = ManagedOrder(
                order_id=str(alpaca_order.id),
                position_id=position_id,
                symbol=symbol,
                side="buy",
                quantity=quantity,
                order_type=order_type,
                purpose=OrderPurpose.ENTRY,
                limit_price=limit_price,
                submitted_at=datetime.now(),
                status=str(alpaca_order.status.value),
                alpaca_order=alpaca_order,
            )

            self.pending_orders.append(managed_order)

            logger.info(
                f"Entry order submitted: {symbol} x{quantity} "
                f"({order_type.value}) | Order ID: {managed_order.order_id}"
            )

            return managed_order

        except Exception as e:
            logger.error(f"Failed to submit entry order: {e}")
            raise

    def submit_profit_target_order(
        self,
        position_id: str,
        symbol: str,
        quantity: int,
        entry_price: float,
    ) -> ManagedOrder:
        """
        Submit profit target order (sell to close at limit).

        Args:
            position_id: Associated position ID
            symbol: Option symbol
            quantity: Number of contracts
            entry_price: Entry price for PT calculation

        Returns:
            ManagedOrder object
        """
        target_price = round(entry_price * (1 + self.profit_target_pct), 2)

        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=target_price,
        )

        try:
            alpaca_order = self.trading_client.submit_order(order_request)

            managed_order = ManagedOrder(
                order_id=str(alpaca_order.id),
                position_id=position_id,
                symbol=symbol,
                side="sell",
                quantity=quantity,
                order_type=OrderType.LIMIT,
                purpose=OrderPurpose.PROFIT_TARGET,
                limit_price=target_price,
                submitted_at=datetime.now(),
                status=str(alpaca_order.status.value),
                alpaca_order=alpaca_order,
            )

            self.pending_orders.append(managed_order)

            logger.info(
                f"Profit target order submitted: SELL {symbol} x{quantity} "
                f"@ ${target_price:.2f} (+{self.profit_target_pct*100:.0f}%) | "
                f"Order ID: {managed_order.order_id}"
            )

            return managed_order

        except Exception as e:
            logger.error(f"Failed to submit profit target order: {e}")
            raise

    def submit_stop_loss_order(
        self,
        position_id: str,
        symbol: str,
        quantity: int,
    ) -> ManagedOrder:
        """
        Submit stop loss order (market sell to close).

        Since stop orders for options can be tricky, we use market order
        when stop is triggered.

        Args:
            position_id: Associated position ID
            symbol: Option symbol
            quantity: Number of contracts

        Returns:
            ManagedOrder object
        """
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        try:
            alpaca_order = self.trading_client.submit_order(order_request)

            managed_order = ManagedOrder(
                order_id=str(alpaca_order.id),
                position_id=position_id,
                symbol=symbol,
                side="sell",
                quantity=quantity,
                order_type=OrderType.MARKET,
                purpose=OrderPurpose.STOP_LOSS,
                submitted_at=datetime.now(),
                status=str(alpaca_order.status.value),
                alpaca_order=alpaca_order,
            )

            self.pending_orders.append(managed_order)

            logger.info(
                f"Stop loss order submitted: SELL {symbol} x{quantity} (MARKET) | "
                f"Order ID: {managed_order.order_id}"
            )

            return managed_order

        except Exception as e:
            logger.error(f"Failed to submit stop loss order: {e}")
            raise

    def submit_force_exit_order(
        self,
        position_id: str,
        symbol: str,
        quantity: int,
    ) -> ManagedOrder:
        """
        Submit force exit order (market sell for 3:50 PM deadline).

        Args:
            position_id: Associated position ID
            symbol: Option symbol
            quantity: Number of contracts

        Returns:
            ManagedOrder object
        """
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        try:
            alpaca_order = self.trading_client.submit_order(order_request)

            managed_order = ManagedOrder(
                order_id=str(alpaca_order.id),
                position_id=position_id,
                symbol=symbol,
                side="sell",
                quantity=quantity,
                order_type=OrderType.MARKET,
                purpose=OrderPurpose.FORCE_EXIT,
                submitted_at=datetime.now(),
                status=str(alpaca_order.status.value),
                alpaca_order=alpaca_order,
            )

            self.pending_orders.append(managed_order)

            logger.warning(
                f"Force exit order submitted: SELL {symbol} x{quantity} (MARKET) | "
                f"Order ID: {managed_order.order_id}"
            )

            return managed_order

        except Exception as e:
            logger.error(f"Failed to submit force exit order: {e}")
            raise

    def cancel_order(self, order: ManagedOrder) -> bool:
        """
        Cancel a pending order.

        Args:
            order: Order to cancel

        Returns:
            True if cancelled successfully
        """
        try:
            self.trading_client.cancel_order_by_id(order.order_id)
            order.status = "cancelled"

            if order in self.pending_orders:
                self.pending_orders.remove(order)
            self.cancelled_orders.append(order)

            logger.info(f"Order cancelled: {order.order_id} ({order.purpose.value})")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order.order_id}: {e}")
            return False

    def cancel_position_orders(self, position_id: str) -> int:
        """
        Cancel all pending orders for a position.

        Args:
            position_id: Position ID

        Returns:
            Number of orders cancelled
        """
        orders_to_cancel = [
            o for o in self.pending_orders
            if o.position_id == position_id
        ]

        cancelled_count = 0
        for order in orders_to_cancel:
            if self.cancel_order(order):
                cancelled_count += 1

        return cancelled_count

    def get_order_status(self, order_id: str) -> Optional[dict]:
        """
        Get current status of an order from Alpaca.

        Args:
            order_id: Order ID

        Returns:
            Order status dict or None
        """
        try:
            alpaca_order = self.trading_client.get_order_by_id(order_id)
            return {
                "id": str(alpaca_order.id),
                "status": str(alpaca_order.status.value),
                "filled_qty": int(alpaca_order.filled_qty or 0),
                "filled_avg_price": float(alpaca_order.filled_avg_price or 0),
                "created_at": alpaca_order.created_at,
                "filled_at": alpaca_order.filled_at,
            }
        except Exception as e:
            logger.error(f"Failed to get order status {order_id}: {e}")
            return None

    def check_order_fills(self) -> List[ManagedOrder]:
        """
        Check pending orders for fills.

        Returns:
            List of orders that were filled
        """
        filled = []

        for order in list(self.pending_orders):
            status = self.get_order_status(order.order_id)
            if status is None:
                continue

            order.status = status["status"]

            if status["status"] == "filled":
                order.filled_at = datetime.now()
                order.filled_price = status["filled_avg_price"]

                self.pending_orders.remove(order)
                self.filled_orders.append(order)
                filled.append(order)

                logger.info(
                    f"Order FILLED: {order.purpose.value} {order.symbol} "
                    f"x{order.quantity} @ ${order.filled_price:.2f}"
                )

            elif status["status"] in ["cancelled", "expired", "rejected"]:
                self.pending_orders.remove(order)
                self.cancelled_orders.append(order)

                logger.warning(
                    f"Order {status['status'].upper()}: {order.order_id} "
                    f"({order.purpose.value})"
                )

        return filled

    def get_pending_orders_for_position(self, position_id: str) -> List[ManagedOrder]:
        """Get all pending orders for a position."""
        return [o for o in self.pending_orders if o.position_id == position_id]

    def calculate_stop_price(self, entry_price: float) -> float:
        """Calculate stop loss trigger price."""
        return round(entry_price * (1 - self.stop_loss_pct), 2)

    def calculate_target_price(self, entry_price: float) -> float:
        """Calculate profit target price."""
        return round(entry_price * (1 + self.profit_target_pct), 2)

    def should_trigger_stop(self, entry_price: float, current_price: float) -> bool:
        """
        Check if stop loss should be triggered.

        Args:
            entry_price: Position entry price
            current_price: Current option price

        Returns:
            True if stop should be triggered
        """
        stop_price = self.calculate_stop_price(entry_price)
        return current_price <= stop_price

    def get_summary(self) -> dict:
        """Get order manager summary."""
        return {
            "pending_orders": len(self.pending_orders),
            "filled_orders": len(self.filled_orders),
            "cancelled_orders": len(self.cancelled_orders),
            "pending_by_purpose": {
                purpose.value: len([o for o in self.pending_orders if o.purpose == purpose])
                for purpose in OrderPurpose
            },
        }
