"""
FastAPI Application Entry Point.

Creates and configures the Marcus Trading Bot API server with routes
for trading, backtesting, and status monitoring.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import trading, backtest, status


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting Marcus Trading Bot - Symbol: {settings.symbol}")
    print(f"Paper Trading: {settings.alpaca_paper}")
    yield
    # Shutdown
    print("Shutting down Marcus Trading Bot")


app = FastAPI(
    title="Marcus Trading Bot",
    description="RSI-based 0DTE options trading bot using Alpaca API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trading.router, prefix="/trading", tags=["Trading"])
app.include_router(backtest.router, prefix="/backtest", tags=["Backtest"])
app.include_router(status.router, prefix="/status", tags=["Status"])


@app.get("/")
async def root():
    return {"message": "Marcus Trading Bot API", "version": "1.0.0"}
