"""
Status API Routes.

Endpoints for health checks, account information, market status,
and current strategy configuration.
"""

from fastapi import APIRouter

from app.services.alpaca_client import alpaca_client
from app.models.schemas import HealthResponse
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    alpaca_connected = False
    market_open = False

    try:
        account = alpaca_client.get_account()
        alpaca_connected = True
        market_open = alpaca_client.is_market_open()
    except Exception:
        pass

    return HealthResponse(
        status="ok" if alpaca_connected else "degraded",
        alpaca_connected=alpaca_connected,
        market_open=market_open,
    )


@router.get("/account")
async def get_account():
    """Get Alpaca account information."""
    try:
        account = alpaca_client.get_account()
        return {
            "account_number": account.account_number,
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/market")
async def get_market_status():
    """Get market status."""
    try:
        clock = alpaca_client.trading_client.get_clock()
        return {
            "is_open": clock.is_open,
            "next_open": clock.next_open.isoformat() if clock.next_open else None,
            "next_close": clock.next_close.isoformat() if clock.next_close else None,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/config")
async def get_config():
    """Get current strategy configuration."""
    return {
        "symbol": settings.symbol,
        "rsi_period": settings.rsi_period,
        "rsi_sma_period": settings.rsi_sma_period,
        "rsi_oversold": settings.rsi_oversold,
        "rsi_overbought": settings.rsi_overbought,
        "profit_target_pct": settings.profit_target_pct,
        "stop_loss_pct": settings.stop_loss_pct,
        "contracts_per_trade": settings.contracts_per_trade,
        "paper_trading": settings.alpaca_paper,
    }
