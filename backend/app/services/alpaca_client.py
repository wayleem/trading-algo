"""
Alpaca API Client.

Unified client for Alpaca trading and market data APIs. Handles stock
bars, option contracts, quotes, and order execution for paper/live trading.
"""

from datetime import date, datetime, timedelta
from typing import Optional
import pytz

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOptionContractsRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, ContractType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from app.core.config import settings


class AlpacaClient:
    """Unified Alpaca client for trading and market data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        self.api_key = api_key or settings.alpaca_api_key
        self.secret_key = secret_key or settings.alpaca_secret_key
        self.paper = paper if paper is not None else settings.alpaca_paper

        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )

        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )

        self.option_data_client = OptionHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )

    def get_account(self):
        """Get account information."""
        return self.trading_client.get_account()

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self.trading_client.get_clock()
        return clock.is_open

    def get_next_market_open(self) -> datetime:
        """Get the next market open time."""
        clock = self.trading_client.get_clock()
        return clock.next_open

    def get_next_market_close(self) -> datetime:
        """Get the next market close time."""
        clock = self.trading_client.get_clock()
        return clock.next_close

    async def get_stock_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        limit: int = 100,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Fetch historical bars for a symbol.

        Args:
            symbol: Stock symbol (e.g., "SPY")
            timeframe: Bar timeframe ("1Min", "5Min", "15Min", "1Hour", "1Day")
            limit: Maximum number of bars to return
            start: Start datetime
            end: End datetime

        Returns:
            List of bar dictionaries with open, high, low, close, volume, timestamp
        """
        tf_map = {
            "1Min": TimeFrame.Minute,
            "3Min": TimeFrame(3, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }

        tf = tf_map.get(timeframe, TimeFrame.Minute)

        if start is None:
            # Use shorter lookback for intraday timeframes to get recent data
            if timeframe in ["1Min", "3Min", "5Min", "15Min"]:
                start = datetime.now(pytz.UTC) - timedelta(hours=4)
            else:
                start = datetime.now(pytz.UTC) - timedelta(days=5)
        if end is None:
            end = datetime.now(pytz.UTC)

        # Don't pass limit to API - it returns bars from START, not END
        # We'll fetch all bars in the range then take the last N
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            feed=DataFeed.SIP,  # Use SIP feed (real-time, Algo Trader Plus)
        )

        bars = self.data_client.get_stock_bars(request)

        result = []
        bar_data = bars.data if hasattr(bars, 'data') else bars
        if symbol in bar_data:
            for bar in bar_data[symbol]:
                result.append(
                    {
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                    }
                )

        # Return only the most recent N bars
        return result[-limit:] if len(result) > limit else result

    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        bars = self.data_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                limit=1,
                feed=DataFeed.IEX,
            )
        )
        bar_data = bars.data if hasattr(bars, 'data') else bars
        if symbol in bar_data and len(bar_data[symbol]) > 0:
            return float(bar_data[symbol][-1].close)
        return 0.0

    def get_option_contracts(
        self,
        underlying_symbol: str,
        expiration_date: date,
        option_type: Optional[str] = None,
    ) -> list:
        """
        Get available options contracts.

        Args:
            underlying_symbol: Underlying stock symbol
            expiration_date: Expiration date
            option_type: "call" or "put" (None for both)

        Returns:
            List of option contracts
        """
        contract_type = None
        if option_type:
            contract_type = (
                ContractType.CALL if option_type.lower() == "call" else ContractType.PUT
            )

        request = GetOptionContractsRequest(
            underlying_symbols=[underlying_symbol],
            expiration_date=expiration_date,
            type=contract_type,
        )

        contracts = self.trading_client.get_option_contracts(request)
        return contracts.option_contracts if contracts else []

    def get_atm_option(
        self,
        underlying_symbol: str,
        option_type: str,
        expiration_date: date,
        current_price: Optional[float] = None,
        strike_offset: float = 0.0,
    ) -> Optional[dict]:
        """
        Find an option contract at specified offset from current price.

        Args:
            underlying_symbol: Underlying stock symbol
            option_type: "call" or "put"
            expiration_date: Expiration date
            current_price: Current stock price (fetched if not provided)
            strike_offset: Offset from current price ($20 = OTM by $20)

        Returns:
            Option contract info or None
        """
        if current_price is None:
            current_price = self.get_latest_price(underlying_symbol)

        contracts = self.get_option_contracts(
            underlying_symbol=underlying_symbol,
            expiration_date=expiration_date,
            option_type=option_type,
        )

        if not contracts:
            return None

        # Calculate target strike based on option type and offset
        # Calls: strike ABOVE current price (OTM)
        # Puts: strike BELOW current price (OTM)
        if option_type == "call":
            target_strike = current_price + strike_offset
        else:
            target_strike = current_price - strike_offset

        # Find strike closest to target
        contract = min(
            contracts, key=lambda c: abs(float(c.strike_price) - target_strike)
        )

        return {
            "symbol": contract.symbol,
            "strike": float(contract.strike_price),
            "expiration": contract.expiration_date,
            "type": option_type,
            "underlying": underlying_symbol,
        }

    def get_option_quote(self, option_symbol: str) -> Optional[dict]:
        """
        Get current bid/ask quote for an option.

        Args:
            option_symbol: Option contract symbol (e.g., "SPY251212C00681000")

        Returns:
            Dictionary with bid, ask, mid, bid_size, ask_size or None if unavailable
        """
        try:
            request = OptionLatestQuoteRequest(symbol_or_symbols=option_symbol)
            quotes = self.option_data_client.get_option_latest_quote(request)

            if option_symbol in quotes:
                quote = quotes[option_symbol]
                bid = float(quote.bid_price) if quote.bid_price else 0.0
                ask = float(quote.ask_price) if quote.ask_price else 0.0

                # Validate we have valid bid/ask
                if bid <= 0 or ask <= 0:
                    return None

                return {
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2,
                    "bid_size": quote.bid_size if quote.bid_size else 0,
                    "ask_size": quote.ask_size if quote.ask_size else 0,
                    "spread": ask - bid,
                    "spread_pct": (ask - bid) / ask * 100 if ask > 0 else 0,
                }
        except Exception as e:
            # Log but don't crash - caller will handle None
            pass

        return None

    def submit_option_order(
        self,
        contract_symbol: str,
        side: str,
        qty: int,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ):
        """
        Submit an option order.

        Args:
            contract_symbol: Option contract symbol
            side: "buy" or "sell"
            qty: Number of contracts
            order_type: "market" or "limit"
            limit_price: Limit price (required for limit orders)

        Returns:
            Order object
        """
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        if order_type == "limit" and limit_price is not None:
            order_request = LimitOrderRequest(
                symbol=contract_symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
            )
        else:
            order_request = MarketOrderRequest(
                symbol=contract_symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

        return self.trading_client.submit_order(order_request)

    def get_positions(self) -> list:
        """Get all open positions."""
        return self.trading_client.get_all_positions()

    def get_position(self, symbol: str):
        """Get position for a specific symbol."""
        try:
            return self.trading_client.get_open_position(symbol)
        except Exception:
            return None

    def close_position(self, symbol: str):
        """Close a position by symbol."""
        return self.trading_client.close_position(symbol)

    def get_orders(self, status: str = "open") -> list:
        """Get orders by status."""
        return self.trading_client.get_orders(
            filter={"status": status},
        )

    def cancel_order(self, order_id: str):
        """Cancel an order by ID."""
        return self.trading_client.cancel_order_by_id(order_id)


# Singleton instance
alpaca_client = AlpacaClient()
