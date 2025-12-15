#!/usr/bin/env python3
"""
Paper Trader for ORB (Opening Range Breakout) Strategy.

Main runner script that orchestrates:
1. Market data from ThetaData/Alpaca
2. ORB signal generation
3. Order execution on Alpaca paper trading
4. Position monitoring and risk management
5. Comprehensive logging

Usage:
    python paper_trading/paper_trader.py [--config CONFIG_FILE]

Environment Variables (loaded from .env):
    ALPACA_API_KEY: Alpaca API key
    ALPACA_SECRET_KEY: Alpaca secret key
"""

import asyncio
import signal
import sys
import os
from datetime import datetime, date, time, timedelta
from typing import Optional
import pytz

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file
from dotenv import load_dotenv
load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from paper_trading.config import PaperTradingConfig, TradingSchedule, DEFAULT_CONFIG, DEFAULT_SCHEDULE
from paper_trading.position_manager import PositionManager, Position, ExitReason
from paper_trading.order_manager import OrderManager, OrderPurpose
from paper_trading.trade_logger import TradeLogger, setup_logging

from app.services.orb_signal_generator import ORBSignalGenerator, ORBSignal
from app.models.schemas import SignalType


# Timezone
ET = pytz.timezone("America/New_York")
UTC = pytz.UTC


class PaperTrader:
    """
    Main paper trading orchestrator for ORB strategy.

    Lifecycle:
    1. Initialize connections and state
    2. Wait for market open
    3. Collect opening range (9:30-10:30 AM ET)
    4. Monitor for breakouts (10:30 AM - 2:00 PM ET)
    5. Manage positions (PT/SL monitoring)
    6. Force exit by 3:50 PM ET
    7. Generate daily summary
    """

    def __init__(
        self,
        config: PaperTradingConfig = DEFAULT_CONFIG,
        schedule: TradingSchedule = DEFAULT_SCHEDULE,
    ):
        """
        Initialize paper trader.

        Args:
            config: Trading configuration
            schedule: Trading schedule configuration
        """
        self.config = config
        self.schedule = schedule

        # Initialize logger
        self.logger = setup_logging(
            log_dir="logs/paper_trading",
            log_level=config.log_level,
        )

        # Alpaca clients
        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables required"
            )

        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,  # ALWAYS paper trading
        )

        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

        self.option_data_client = OptionHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

        # Position and order managers
        self.position_manager = PositionManager(
            max_positions=config.max_positions,
            daily_loss_limit=config.daily_loss_limit,
        )

        self.order_manager = OrderManager(
            trading_client=self.trading_client,
            profit_target_pct=config.profit_target_pct,
            stop_loss_pct=config.stop_loss_pct,
        )

        # ORB signal generator
        self.signal_generator: Optional[ORBSignalGenerator] = None

        # State
        self.running = False
        self.current_date: Optional[date] = None
        self.entry_cutoff_reached = False
        self.range_defined = False

        # Position counter for IDs
        self.position_counter = 0

    async def run(self) -> None:
        """Main run loop."""
        self.running = True
        self.logger.log_info("Paper Trader starting...")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        try:
            while self.running:
                await self._run_trading_day()

                # Check if we should continue
                if not self.running:
                    break

                # Wait for next trading day
                await self._wait_for_next_trading_day()

        except Exception as e:
            self.logger.log_error("Fatal error in paper trader", e)
            raise
        finally:
            await self._shutdown()

    async def _run_trading_day(self) -> None:
        """Run a single trading day."""
        now = datetime.now(ET)
        today = now.date()

        # Check if today is a trading day
        if not self.schedule.is_trading_day(today):
            self.logger.log_info(f"{today} is not a trading day (weekend/holiday)")
            return

        self.current_date = today
        self.logger.start_session(today)

        # Reset state for new day
        self._reset_for_new_day()

        try:
            # Phase 1: Wait for market open
            await self._wait_for_market_open()

            if not self.running:
                return

            # Phase 2: Build opening range (9:30-10:30 AM ET)
            await self._build_opening_range()

            if not self.running or not self.range_defined:
                return

            # Phase 3: Monitor for breakouts and manage positions
            await self._trading_loop()

        except Exception as e:
            self.logger.log_error(f"Error during trading day: {e}", e)

        finally:
            # End of day cleanup
            await self._end_of_day_cleanup()
            self.logger.end_session(self.position_manager.daily_stats)

    def _reset_for_new_day(self) -> None:
        """Reset all state for a new trading day."""
        self.signal_generator = ORBSignalGenerator(
            range_minutes=self.config.range_minutes,
            entry_cutoff_hour_utc=self.config.entry_cutoff_hour_utc,
            breakout_buffer=self.config.breakout_buffer,
            require_close=self.config.require_close,
        )
        self.signal_generator.max_entries_per_day = 1  # Only 1 position at a time

        self.position_manager.reset_for_new_day(self.current_date)
        self.entry_cutoff_reached = False
        self.range_defined = False

    async def _wait_for_market_open(self) -> None:
        """Wait until market opens."""
        clock = self.trading_client.get_clock()

        if clock.is_open:
            self.logger.log_market_status(True, "close", clock.next_close)
            return

        self.logger.log_market_status(False, "open", clock.next_open)

        # Wait until 9:25 AM ET (5 min before open)
        target_time = datetime.combine(
            self.current_date,
            time(9, 25),
        ).replace(tzinfo=ET)

        now = datetime.now(ET)
        if now < target_time:
            wait_seconds = (target_time - now).total_seconds()
            self.logger.log_waiting("Market not open yet", target_time)
            await asyncio.sleep(wait_seconds)

        # Now poll until market is open
        while not clock.is_open and self.running:
            await asyncio.sleep(5)
            clock = self.trading_client.get_clock()

        if clock.is_open:
            self.logger.log_info("Market is now OPEN")

    async def _build_opening_range(self) -> None:
        """
        Build the opening range from 9:30-10:30 AM ET.

        Collects 1-minute bars and feeds them to the ORB signal generator.
        """
        self.logger.log_info("Building opening range (9:30-10:30 AM ET)...")

        range_end_time = datetime.combine(
            self.current_date,
            self.config.range_end_time,
        ).replace(tzinfo=ET)

        bars_collected = 0

        while datetime.now(ET) < range_end_time and self.running:
            # Fetch latest 1-min bar
            bar = await self._get_latest_bar()

            if bar:
                signal = self.signal_generator.process_bar(bar)
                bars_collected += 1

                # Log progress every 10 bars
                if bars_collected % 10 == 0:
                    self.logger.log_debug(f"Range building: {bars_collected} bars collected")

            # Wait for next bar
            await asyncio.sleep(60)  # 1-minute bars

        # Check if range was defined
        if self.signal_generator.is_range_defined():
            range_info = self.signal_generator.get_range_info()
            self.range_defined = True
            self.logger.log_range_formed(
                range_high=range_info["high"],
                range_low=range_info["low"],
                range_size=range_info["range_size"],
            )

            # Check if range is tradeable
            range_size = range_info["range_size"]
            if not (self.config.min_range_size <= range_size <= self.config.max_range_size):
                self.logger.log_warning(
                    f"Range size ${range_size:.2f} outside bounds "
                    f"[${self.config.min_range_size:.2f}-${self.config.max_range_size:.2f}]. "
                    "No trading today."
                )
                self.range_defined = False
        else:
            self.logger.log_warning("Failed to define opening range")

    async def _trading_loop(self) -> None:
        """
        Main trading loop: monitor for breakouts and manage positions.

        Runs from 10:30 AM until 3:50 PM ET (force exit deadline).
        """
        self.logger.log_info("Starting trading loop (monitoring for breakouts)...")

        force_exit_time = datetime.combine(
            self.current_date,
            self.config.force_exit_time,
        ).replace(tzinfo=ET)

        entry_cutoff_time = datetime.combine(
            self.current_date,
            self.config.entry_cutoff_time,
        ).replace(tzinfo=ET)

        while datetime.now(ET) < force_exit_time and self.running:
            now = datetime.now(ET)

            # Check entry cutoff
            if now >= entry_cutoff_time and not self.entry_cutoff_reached:
                self.entry_cutoff_reached = True
                self.logger.log_entry_cutoff()

            # Check force exit deadline (3:50 PM)
            if now >= force_exit_time:
                break

            # Check for filled orders
            filled_orders = self.order_manager.check_order_fills()
            await self._handle_filled_orders(filled_orders)

            # Get current position
            position = self.position_manager.get_open_position()

            if position:
                # Monitor existing position
                await self._monitor_position(position)
            elif not self.entry_cutoff_reached and not self.position_manager.circuit_breaker_active:
                # Look for new entry signals
                await self._check_for_entry_signal()

            # Poll interval
            await asyncio.sleep(self.config.price_poll_interval)

        # Force exit any remaining positions
        await self._force_exit_all_positions()

    async def _check_for_entry_signal(self) -> None:
        """Check for new entry signal."""
        can_open, reason = self.position_manager.can_open_position()
        if not can_open:
            return

        # Get latest bar
        bar = await self._get_latest_bar()
        if not bar:
            return

        # Process bar through signal generator
        signal = self.signal_generator.process_bar(bar)

        if signal and signal.signal_type in (SignalType.BUY_CALL, SignalType.BUY_PUT):
            await self._execute_entry(signal, bar)

    async def _execute_entry(self, signal: ORBSignal, bar: dict) -> None:
        """
        Execute entry trade based on signal.

        Args:
            signal: ORB signal (BUY_CALL or BUY_PUT)
            bar: Current price bar
        """
        option_type = "call" if signal.signal_type == SignalType.BUY_CALL else "put"
        underlying_price = bar["close"]

        self.logger.log_signal(
            signal_type=signal.signal_type.name,
            underlying_price=underlying_price,
            range_high=signal.range_high,
            range_low=signal.range_low,
            reason=signal.reason,
            action_taken=f"Attempting to buy {option_type.upper()}",
        )

        # Find 0DTE ATM option
        option_contract = await self._find_option_contract(
            underlying_price=underlying_price,
            option_type=option_type,
        )

        if not option_contract:
            self.logger.log_warning("No suitable option contract found")
            return

        # Get option quote
        quote = await self._get_option_quote(option_contract["symbol"])
        if not quote:
            self.logger.log_warning(f"No quote available for {option_contract['symbol']}")
            return

        # Check spread
        if quote["spread_pct"] > self.config.max_spread_pct * 100:
            self.logger.log_warning(
                f"Bid-ask spread too wide: {quote['spread_pct']:.1f}% > "
                f"{self.config.max_spread_pct * 100:.1f}%"
            )
            return

        # Create position
        self.position_counter += 1
        position_id = f"ORB_{self.current_date}_{self.position_counter}"

        position = self.position_manager.create_position(
            position_id=position_id,
            option_symbol=option_contract["symbol"],
            underlying_symbol=self.config.underlying_symbol,
            option_type=option_type,
            strike=option_contract["strike"],
            expiration=option_contract["expiration"],
            entry_price=quote["ask"],  # Expect to pay the ask
            quantity=self.config.contracts_per_trade,
            signal_reason=signal.reason,
            range_high=signal.range_high,
            range_low=signal.range_low,
        )

        # Submit entry order
        try:
            entry_order = self.order_manager.submit_entry_order(
                position_id=position_id,
                symbol=option_contract["symbol"],
                quantity=self.config.contracts_per_trade,
                use_market_order=self.config.use_market_orders_for_entry,
            )

            position.entry_order_id = entry_order.order_id

            self.logger.log_order_submitted(
                order_type="MARKET" if self.config.use_market_orders_for_entry else "LIMIT",
                symbol=option_contract["symbol"],
                side="BUY",
                quantity=self.config.contracts_per_trade,
                price=None if self.config.use_market_orders_for_entry else quote["ask"],
                order_id=entry_order.order_id,
            )

        except Exception as e:
            self.logger.log_error(f"Failed to submit entry order: {e}", e)

    async def _handle_filled_orders(self, filled_orders: list) -> None:
        """Handle filled orders."""
        for order in filled_orders:
            self.logger.log_order_filled(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=order.filled_price,
                order_id=order.order_id,
            )

            if order.purpose == OrderPurpose.ENTRY:
                await self._handle_entry_fill(order)
            elif order.purpose == OrderPurpose.PROFIT_TARGET:
                await self._handle_profit_target_fill(order)
            elif order.purpose in (OrderPurpose.STOP_LOSS, OrderPurpose.FORCE_EXIT):
                await self._handle_exit_fill(order)

    async def _handle_entry_fill(self, order) -> None:
        """Handle entry order fill."""
        # Find the position
        for position in self.position_manager.all_positions:
            if position.position_id == order.position_id:
                # Confirm entry
                self.position_manager.confirm_entry(
                    position=position,
                    fill_price=order.filled_price,
                    order_id=order.order_id,
                )

                # Calculate exit prices
                profit_target = self.order_manager.calculate_target_price(order.filled_price)
                stop_loss = self.order_manager.calculate_stop_price(order.filled_price)

                self.logger.log_position_opened(
                    symbol=position.option_symbol,
                    option_type=position.option_type,
                    strike=position.strike,
                    quantity=position.quantity,
                    entry_price=order.filled_price,
                    profit_target=profit_target,
                    stop_loss=stop_loss,
                )

                # Submit profit target order
                try:
                    pt_order = self.order_manager.submit_profit_target_order(
                        position_id=position.position_id,
                        symbol=position.option_symbol,
                        quantity=position.quantity,
                        entry_price=order.filled_price,
                    )
                    position.profit_target_order_id = pt_order.order_id

                except Exception as e:
                    self.logger.log_error(f"Failed to submit PT order: {e}", e)

                break

    async def _handle_profit_target_fill(self, order) -> None:
        """Handle profit target fill."""
        position = self.position_manager.get_open_position()
        if position and position.position_id == order.position_id:
            self.position_manager.close_position(
                position=position,
                exit_price=order.filled_price,
                exit_reason=ExitReason.PROFIT_TARGET,
            )
            self.logger.log_position_closed(position)

    async def _handle_exit_fill(self, order) -> None:
        """Handle stop loss or force exit fill."""
        position = self.position_manager.get_open_position()
        if position and position.position_id == order.position_id:
            exit_reason = (
                ExitReason.STOP_LOSS
                if order.purpose == OrderPurpose.STOP_LOSS
                else ExitReason.FORCE_EXIT
            )
            self.position_manager.close_position(
                position=position,
                exit_price=order.filled_price,
                exit_reason=exit_reason,
            )
            self.logger.log_position_closed(position)

    async def _monitor_position(self, position: Position) -> None:
        """
        Monitor open position for stop loss trigger.

        Since Alpaca doesn't support stop orders for options reliably,
        we monitor price and trigger market sell when stop is hit.
        """
        # Get current option price
        quote = await self._get_option_quote(position.option_symbol)
        if not quote:
            return

        current_price = quote["bid"]  # Use bid for exit estimation
        self.position_manager.update_position_price(position, current_price)

        # Check stop loss
        if self.order_manager.should_trigger_stop(position.entry_price, current_price):
            self.logger.log_info(
                f"STOP LOSS TRIGGERED: {position.option_symbol} "
                f"${current_price:.2f} <= ${self.order_manager.calculate_stop_price(position.entry_price):.2f}"
            )

            # Cancel profit target order
            self.order_manager.cancel_position_orders(position.position_id)

            # Submit market sell
            try:
                stop_order = self.order_manager.submit_stop_loss_order(
                    position_id=position.position_id,
                    symbol=position.option_symbol,
                    quantity=position.quantity,
                )
                position.stop_loss_order_id = stop_order.order_id

            except Exception as e:
                self.logger.log_error(f"Failed to submit stop loss order: {e}", e)

    async def _force_exit_all_positions(self) -> None:
        """
        Force exit all remaining positions (3:50 PM deadline).

        For 0DTE options, we check intrinsic value:
        - If OTM (intrinsic = 0), option expires worthless - record $0 exit
        - If ITM (intrinsic > 0), submit market sell to capture remaining value
        """
        position = self.position_manager.get_open_position()
        if not position:
            return

        self.logger.log_force_exit_deadline()

        # Cancel all pending orders first
        self.order_manager.cancel_position_orders(position.position_id)

        # Get current underlying price to calculate intrinsic value
        bar = await self._get_latest_bar()
        underlying_price = bar["close"] if bar else 0.0

        # Calculate intrinsic (settlement) value
        intrinsic = self.position_manager.calculate_settlement_value(
            position, underlying_price
        )

        # Store settlement price for logging
        position.settlement_underlying = underlying_price

        if intrinsic <= 0:
            # Option is OTM - will expire worthless
            # No point submitting market order, just record $0 exit
            self.logger.log_option_expired_worthless(
                symbol=position.option_symbol,
                strike=position.strike,
                underlying_price=underlying_price,
                option_type=position.option_type,
                entry_price=position.entry_price,
                quantity=position.quantity,
            )

            self.position_manager.close_position(
                position=position,
                exit_price=0.0,  # Worthless
                exit_reason=ExitReason.FORCE_EXIT,
            )
            self.logger.log_position_closed(position)
            return

        # Option has intrinsic value - submit market sell
        self.logger.log_info(
            f"Option ITM at exit: {position.option_symbol} | "
            f"Strike=${position.strike:.2f} | SPY=${underlying_price:.2f} | "
            f"Intrinsic=${intrinsic:.2f}"
        )

        try:
            exit_order = self.order_manager.submit_force_exit_order(
                position_id=position.position_id,
                symbol=position.option_symbol,
                quantity=position.quantity,
            )

            # Wait for fill
            for _ in range(30):  # 30 second timeout
                filled = self.order_manager.check_order_fills()
                if filled:
                    await self._handle_filled_orders(filled)
                    break
                await asyncio.sleep(1)

        except Exception as e:
            self.logger.log_error(f"Failed to force exit position: {e}", e)
            # If order fails, close at intrinsic value as fallback
            self.position_manager.close_position(
                position=position,
                exit_price=intrinsic,
                exit_reason=ExitReason.FORCE_EXIT,
            )
            self.logger.log_position_closed(position)

    async def _end_of_day_cleanup(self) -> None:
        """End of day cleanup."""
        # Cancel any remaining orders
        for order in list(self.order_manager.pending_orders):
            self.order_manager.cancel_order(order)

        # Close any remaining positions (shouldn't happen if force exit worked)
        position = self.position_manager.get_open_position()
        if position:
            self.logger.log_warning("Position still open after force exit!")
            # Try one more time
            try:
                self.trading_client.close_position(position.option_symbol)
            except Exception as e:
                self.logger.log_error(f"Failed to close position: {e}", e)

    async def _wait_for_next_trading_day(self) -> None:
        """Wait until the next trading day."""
        now = datetime.now(ET)

        # Find next trading day
        next_day = now.date() + timedelta(days=1)
        while not self.schedule.is_trading_day(next_day):
            next_day += timedelta(days=1)

        # Calculate wait time (until 9:25 AM on next trading day)
        next_session = datetime.combine(next_day, time(9, 25)).replace(tzinfo=ET)
        wait_seconds = (next_session - now).total_seconds()

        if wait_seconds > 0:
            self.logger.log_info(
                f"Next trading session: {next_day}. "
                f"Sleeping for {wait_seconds/3600:.1f} hours."
            )
            await asyncio.sleep(wait_seconds)

    async def _get_latest_bar(self) -> Optional[dict]:
        """Get the latest 1-minute bar for SPY."""
        try:
            end = datetime.now(UTC)
            start = end - timedelta(minutes=5)

            request = StockBarsRequest(
                symbol_or_symbols=self.config.underlying_symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                feed=DataFeed.SIP,
            )

            bars = self.data_client.get_stock_bars(request)
            bar_data = bars.data if hasattr(bars, 'data') else bars

            if self.config.underlying_symbol in bar_data:
                bar_list = bar_data[self.config.underlying_symbol]
                if bar_list:
                    bar = bar_list[-1]
                    return {
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                    }

        except Exception as e:
            self.logger.log_error(f"Failed to get latest bar: {e}", e)

        return None

    async def _find_option_contract(
        self,
        underlying_price: float,
        option_type: str,
    ) -> Optional[dict]:
        """
        Find a 0DTE ATM option contract.

        Args:
            underlying_price: Current underlying price
            option_type: "call" or "put"

        Returns:
            Option contract info or None
        """
        try:
            from alpaca.trading.requests import GetOptionContractsRequest
            from alpaca.trading.enums import ContractType

            contract_type = (
                ContractType.CALL if option_type == "call" else ContractType.PUT
            )

            # 0DTE = today's expiration
            expiration = self.current_date

            request = GetOptionContractsRequest(
                underlying_symbols=[self.config.underlying_symbol],
                expiration_date=expiration,
                type=contract_type,
            )

            contracts = self.trading_client.get_option_contracts(request)

            if not contracts or not contracts.option_contracts:
                return None

            # Find ATM strike
            target_strike = underlying_price + self.config.strike_offset
            if option_type == "put":
                target_strike = underlying_price - self.config.strike_offset

            contract = min(
                contracts.option_contracts,
                key=lambda c: abs(float(c.strike_price) - target_strike)
            )

            return {
                "symbol": contract.symbol,
                "strike": float(contract.strike_price),
                "expiration": contract.expiration_date,
                "type": option_type,
            }

        except Exception as e:
            self.logger.log_error(f"Failed to find option contract: {e}", e)
            return None

    async def _get_option_quote(self, option_symbol: str) -> Optional[dict]:
        """
        Get current quote for an option.

        Args:
            option_symbol: Option symbol

        Returns:
            Quote dict with bid, ask, mid, spread info
        """
        try:
            request = OptionLatestQuoteRequest(symbol_or_symbols=option_symbol)
            quotes = self.option_data_client.get_option_latest_quote(request)

            if option_symbol in quotes:
                quote = quotes[option_symbol]
                bid = float(quote.bid_price) if quote.bid_price else 0.0
                ask = float(quote.ask_price) if quote.ask_price else 0.0

                if bid <= 0 or ask <= 0:
                    return None

                return {
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2,
                    "spread": ask - bid,
                    "spread_pct": (ask - bid) / ask * 100 if ask > 0 else 0,
                }

        except Exception as e:
            self.logger.log_error(f"Failed to get option quote: {e}", e)

        return None

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signal."""
        self.logger.log_warning("Shutdown signal received...")
        self.running = False

    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        self.logger.log_info("Shutting down paper trader...")

        # Force exit any open positions
        await self._force_exit_all_positions()

        self.logger.log_info("Paper trader shutdown complete.")


async def main():
    """Main entry point."""
    # Load config
    config = PaperTradingConfig.from_env()
    schedule = TradingSchedule()

    # Create and run paper trader
    trader = PaperTrader(config=config, schedule=schedule)
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
