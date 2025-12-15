"""
Backtest API Routes.

Endpoints for running historical backtests and retrieving default
strategy parameters.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime

from app.services.backtest import BacktestService, BacktestConfig
from app.services.alpaca_client import alpaca_client
from app.models.schemas import BacktestRequest, BacktestResult

router = APIRouter()

backtest_service = BacktestService(alpaca_client)


@router.post("/run", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    """
    Run historical backtest with specified parameters.

    Returns trade list, performance metrics, and equity curve.
    """
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD",
        )

    if start_date >= end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date must be before end_date",
        )

    config = BacktestConfig(
        symbol=request.symbol,
        start_date=start_date,
        end_date=end_date,
        rsi_period=request.rsi_period,
        rsi_sma_period=request.rsi_sma_period,
        profit_target_pct=request.profit_target_pct,
        stop_loss_pct=request.stop_loss_pct,
    )

    result = await backtest_service.run_backtest(config)

    return result


@router.get("/default-params")
async def get_default_params():
    """Get default backtest parameters."""
    return {
        "symbol": "SPY",
        "rsi_period": 14,
        "rsi_sma_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "profit_target_pct": 0.20,
        "stop_loss_pct": 0.50,
    }
