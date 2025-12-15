"""
Application Configuration.

Loads settings from environment variables and .env file. Contains all
configurable parameters for API credentials, trading strategy, and
risk management.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Alpaca API Credentials
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True

    # ThetaData API (ThetaTerminal runs locally)
    theta_data_base_url: str = "http://localhost:25503"
    theta_data_api_key: str = ""  # Not needed for local ThetaTerminal

    # Strategy Parameters
    symbol: str = "SPY"
    rsi_period: int = 14
    rsi_sma_period: int = 14

    # RSI thresholds for entry signals
    rsi_oversold: float = 30.0       # Entry: RSI < 30 for calls
    rsi_overbought: float = 70.0     # Entry: RSI > 70 for puts

    # RSI-SMA gap required for entry signals
    rsi_sma_gap: float = 5.0  # RSI must be at least 5 points away from SMA

    # RSI convergence exit thresholds
    rsi_convergence_call_exit: float = 40.0  # Exit CALL when RSI >= 40
    rsi_convergence_put_exit: float = 60.0   # Exit PUT when RSI <= 60

    # Entry time restriction
    entry_cutoff_hour_utc: int = 17  # 12:00 PM ET = 17:00 UTC

    # Trading Parameters
    contracts_per_trade: int = 1
    profit_target_pct: float = 0.05  # 5% profit target
    stop_loss_pct: float = 0.50      # 50% stop loss
    max_hold_minutes: int = 3        # Maximum holding period in minutes
    strike_offset: float = 0.5       # $0.50 OTM for better risk/reward (optimized from 20.0)
    slippage_pct: float = 0.01       # 1% slippage on options

    # Averaging down configuration
    avg_down_trigger_pct: float = 0.10  # Add contract every -10% from original entry
    max_add_ons: int = 3                # Max 3 add-ons (4 contracts total)

    # P/L monitoring frequency
    pl_check_interval_seconds: float = 5.0  # Check P/L every 5 seconds

    # Virtual Capital (for paper trading simulation)
    initial_capital: float = 10000.0  # Starting capital to simulate
    daily_loss_limit: float = 200.0   # Stop trading if daily loss exceeds this
    max_risk_per_trade_pct: float = 0.02  # Max 2% risk per trade

    # Parallel mode settings (optimized)
    parallel_mode: bool = True  # Run 3-min and 5-min RSI independently

    # Pattern-based position sizing
    pattern_bonus_contracts: int = 4  # Extra contracts on strong patterns
    pattern_strength_threshold: float = 0.8  # Min strength to add bonus

    @property
    def alpaca_base_url(self) -> str:
        if self.alpaca_paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"


settings = Settings()
