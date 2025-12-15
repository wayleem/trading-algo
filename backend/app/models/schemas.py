"""
Pydantic Data Models.

Defines request/response schemas and internal data structures for
trading signals, trade records, backtest results, and API responses.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class SignalType(str, Enum):
    BUY_CALL = "buy_call"
    BUY_PUT = "buy_put"
    NO_SIGNAL = "no_signal"


class TradeStatus(str, Enum):
    OPEN = "open"
    CLOSED_PROFIT = "closed_profit"
    CLOSED_LOSS = "closed_loss"
    CLOSED_EOD = "closed_eod"
    CLOSED_TIMEOUT = "closed_timeout"  # Max hold time reached
    CLOSED_RSI_CONVERGENCE = "closed_rsi_convergence"  # RSI converged with SMA in neutral zone


class TradingSignal(BaseModel):
    signal_type: SignalType
    timestamp: datetime
    rsi: float
    rsi_sma: float
    close_price: float
    reason: str


class TradeRecord(BaseModel):
    id: str
    timestamp: datetime
    signal_type: SignalType
    option_symbol: str
    underlying_price: float
    strike: float
    option_type: str
    quantity: int
    entry_price: float
    profit_target: float
    stop_loss: float
    status: TradeStatus = TradeStatus.OPEN
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl: Optional[float] = None
    # Averaging down support
    total_contracts: int = 1  # Total contracts including add-ons
    avg_entry_price: Optional[float] = None  # Weighted avg price after add-ons
    add_on_entries: list[tuple[float, int, datetime]] = []  # [(price, qty, time), ...]

    def get_effective_entry_price(self) -> float:
        """Get the average entry price (accounts for averaging down)."""
        return self.avg_entry_price if self.avg_entry_price else self.entry_price


class RSIData(BaseModel):
    timestamp: datetime
    close: float
    rsi: float
    rsi_sma: float

    @property
    def is_oversold(self) -> bool:
        return self.rsi < 30

    @property
    def is_overbought(self) -> bool:
        return self.rsi > 70


class BacktestRequest(BaseModel):
    start_date: str
    end_date: str
    symbol: str = "SPY"
    rsi_period: int = 14
    rsi_sma_period: int = 14
    profit_target_pct: float = 0.20
    stop_loss_pct: float = 0.50


class BacktestTrade(BaseModel):
    entry_date: datetime
    exit_date: datetime
    signal_type: SignalType
    entry_price: float
    exit_price: float
    pnl_dollars: float
    pnl_percent: float
    exit_reason: str
    entry_rsi: float = 0.0
    rsi_history: list = []  # [(timestamp, rsi, option_price), ...] for debugging
    timeframe: str = ""  # Which timeframe generated this trade (for parallel mode)
    strike: float = 0.0  # Option strike price
    settlement_underlying: float = 0.0  # Underlying price at settlement (for EOD expiration)


class BacktestMetrics(BaseModel):
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown: float
    profit_factor: float
    sharpe_ratio: float = 0.0
    # Contract-level metrics (counts each entry including add-ons)
    total_contracts: int = 0
    winning_contracts: int = 0
    losing_contracts: int = 0
    contract_win_rate: float = 0.0


class BacktestResult(BaseModel):
    trades: list[BacktestTrade]
    metrics: BacktestMetrics
    equity_curve: list[tuple[datetime, float]]


class TradingStatus(BaseModel):
    is_running: bool
    open_positions: int
    total_trades_today: int
    pnl_today: float


class HealthResponse(BaseModel):
    status: str
    alpaca_connected: bool
    market_open: bool
