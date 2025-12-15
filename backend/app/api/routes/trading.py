"""
Trading API Routes.

Endpoints for controlling paper trading: start/stop trading bot,
monitor positions, and manually close trades.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
import asyncio
import logging

from app.services.alpaca_client import alpaca_client
from app.services.indicators import IndicatorService
from app.services.signal_generator import SignalGenerator
from app.services.order_executor import OrderExecutor
from app.services.position_manager import PositionManager
from app.models.schemas import TradingStatus, TradeRecord
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state
trading_task = None
is_trading = False
executor = OrderExecutor(alpaca_client)
position_manager = PositionManager(alpaca_client, executor)


async def trading_loop():
    """Main trading loop - monitors market and executes strategy."""
    global is_trading

    indicator_service = IndicatorService(
        rsi_period=settings.rsi_period,
        sma_period=settings.rsi_sma_period,
    )
    signal_generator = SignalGenerator()

    # Keep track of RSI values for SMA calculation
    rsi_history = []

    logger.info("Trading loop started")

    while is_trading:
        try:
            # Check if market is open
            if not position_manager.is_market_hours():
                logger.debug("Market closed, waiting...")
                await asyncio.sleep(60)
                continue

            # Fetch latest bars (need enough for RSI calculation)
            bars = await alpaca_client.get_stock_bars(
                symbol=settings.symbol,
                timeframe="1Min",
                limit=settings.rsi_period + settings.rsi_sma_period + 5,
            )

            if len(bars) < settings.rsi_period + 1:
                logger.warning("Insufficient bars for RSI calculation")
                await asyncio.sleep(60)
                continue

            # Calculate current RSI
            closes = [bar["close"] for bar in bars]
            current_rsi = indicator_service.calculate_rsi(closes)

            # Update RSI history and calculate SMA
            rsi_history.append(current_rsi)
            if len(rsi_history) > settings.rsi_sma_period * 2:
                rsi_history = rsi_history[-settings.rsi_sma_period * 2 :]

            current_sma = indicator_service.calculate_sma(rsi_history)

            # Generate signal
            latest_bar = bars[-1]
            signal = signal_generator.evaluate(
                current_rsi=current_rsi,
                current_sma=current_sma,
                close_price=latest_bar["close"],
                timestamp=latest_bar["timestamp"],
            )

            logger.debug(
                f"RSI: {current_rsi:.1f}, SMA: {current_sma:.1f}, "
                f"Signal: {signal.signal_type.value}"
            )

            # Execute signal if we don't have an open position
            if len(executor.get_open_trades()) == 0:
                trade = executor.execute_signal(signal)
                if trade:
                    logger.info(f"Opened trade: {trade.option_symbol}")

            # Check exits for open positions
            closed = await position_manager.check_exits()
            for trade in closed:
                logger.info(f"Closed trade: {trade.option_symbol}, P&L: ${trade.pnl:.2f}")

            # Wait for next bar
            await asyncio.sleep(60)

        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await asyncio.sleep(60)

    logger.info("Trading loop stopped")


@router.post("/start")
async def start_trading(background_tasks: BackgroundTasks):
    """Start the paper trading bot."""
    global is_trading, trading_task

    if is_trading:
        raise HTTPException(status_code=400, detail="Trading already running")

    # Verify Alpaca connection
    try:
        account = alpaca_client.get_account()
        logger.info(f"Connected to Alpaca. Account: {account.account_number}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to Alpaca: {e}")

    is_trading = True
    background_tasks.add_task(trading_loop)

    return {"message": "Trading started", "symbol": settings.symbol}


@router.post("/stop")
async def stop_trading():
    """Stop the paper trading bot."""
    global is_trading

    if not is_trading:
        raise HTTPException(status_code=400, detail="Trading not running")

    is_trading = False

    # Force close any open positions
    closed = await position_manager.force_close_all(reason="manual_stop")

    return {
        "message": "Trading stopped",
        "closed_positions": len(closed),
        "total_pnl": executor.get_total_pnl(),
    }


@router.get("/status", response_model=TradingStatus)
async def get_trading_status():
    """Get current trading status."""
    return TradingStatus(
        is_running=is_trading,
        open_positions=len(executor.get_open_trades()),
        total_trades_today=len(executor.get_closed_trades()),
        pnl_today=executor.get_total_pnl(),
    )


@router.get("/positions", response_model=list[TradeRecord])
async def get_positions():
    """Get open positions."""
    return executor.get_open_trades()


@router.get("/trades", response_model=list[TradeRecord])
async def get_trades():
    """Get closed trades from today."""
    return executor.get_closed_trades()


@router.post("/close-all")
async def close_all_positions():
    """Manually close all open positions."""
    closed = await position_manager.force_close_all(reason="manual")
    return {
        "message": f"Closed {len(closed)} positions",
        "trades": closed,
    }
