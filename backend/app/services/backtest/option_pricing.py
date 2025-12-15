"""Option pricing module for backtest simulation.

Handles real option price fetching from ThetaData and estimation fallbacks.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from app.core.config import settings
from app.services.theta_data import ThetaDataError
from app.models.schemas import SignalType

logger = logging.getLogger(__name__)


class OptionPricingService:
    """
    Service for fetching and estimating option prices.

    Supports:
    - Real historical prices from ThetaData
    - Price interpolation for intraday times
    - Estimation fallbacks when real data unavailable
    """

    def __init__(self, theta_available: bool = True):
        """
        Initialize option pricing service.

        Args:
            theta_available: Whether ThetaData is available for real prices
        """
        self._theta_available = theta_available
        self._option_price_cache: dict = {}
        self._intraday_cache: dict = {}

    def clear_cache(self):
        """Clear all cached option data."""
        self._option_price_cache.clear()
        self._intraday_cache.clear()

    async def get_real_option_price(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        option_type: str,
        target_date: date,
        entry_time: bool = True,
    ) -> Optional[float]:
        """
        Get real historical option price from ThetaData.

        Uses mid price (bid+ask)/2 for accurate simulation.
        For 0DTE options, gets the first available price of the day (entry) or
        last price (exit).

        Args:
            symbol: Underlying symbol (e.g., "SPY", "SPXW")
            expiration: Option expiration date
            strike: Strike price
            option_type: "call" or "put"
            target_date: Date to get price for
            entry_time: If True, get first price of day; if False, get last price

        Returns:
            Mid price or None if unavailable
        """
        if not self._theta_available:
            return None

        # Check cache first
        right = "C" if option_type == "call" else "P"
        time_key = "entry" if entry_time else "exit"
        cache_key = (symbol, expiration, strike, right, target_date, time_key)
        if cache_key in self._option_price_cache:
            return self._option_price_cache[cache_key]

        try:
            # Fetch all intraday data for the day
            data = await self.fetch_intraday_option_data(
                symbol=symbol,
                expiration=expiration,
                strike=strike,
                right=right,
                target_date=target_date,
            )

            if data:
                # Get first or last valid price
                if entry_time:
                    # Get first valid price (market open)
                    for point in data:
                        if point.get("midpoint", 0) > 0:
                            price = float(point["midpoint"])
                            self._option_price_cache[cache_key] = price
                            return price
                else:
                    # Get last valid price (EOD)
                    for point in reversed(data):
                        if point.get("midpoint", 0) > 0:
                            price = float(point["midpoint"])
                            self._option_price_cache[cache_key] = price
                            return price

        except ThetaDataError as e:
            logger.debug(f"ThetaData error for {symbol} {strike} {option_type}: {e}")
        except Exception as e:
            logger.debug(f"Error getting option price: {e}")

        return None

    async def fetch_intraday_option_data(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
        target_date: date,
    ) -> list[dict]:
        """
        Fetch all intraday 1-minute OHLC data for an option contract.

        Uses ThetaData v3 OHLC endpoint with 1-minute intervals.

        Args:
            symbol: Option root symbol (e.g., "SPY", "SPXW")
            expiration: Option expiration date
            strike: Strike price in dollars
            right: "C" for call, "P" for put
            target_date: Date to fetch data for

        Returns:
            List of data points with timestamp, midpoint, open, high, low, close.
        """
        # Check cache
        cache_key = (symbol, expiration, strike, right, target_date)
        if cache_key in self._intraday_cache:
            return self._intraday_cache[cache_key]

        import httpx

        try:
            # v3 API uses strike in dollars with decimal (e.g., "580.000")
            # and right as "call" or "put"
            right_str = "call" if right.upper() == "C" else "put"

            url = f"{settings.theta_data_base_url}/v3/option/history/ohlc"
            params = {
                "symbol": symbol.upper(),
                "expiration": expiration.strftime("%Y%m%d"),
                "strike": f"{strike:.3f}",  # Strike in dollars with 3 decimals
                "right": right_str,
                "date": target_date.strftime("%Y%m%d"),
                "interval": "1m",  # 1 minute bars
                "format": "json",
            }
            logger.info(f"ThetaData intraday request: {url} params={params}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)

                logger.info(f"ThetaData response: status={response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"ThetaData response keys: {data.keys() if isinstance(data, dict) else type(data)}")
                    # v3 response format: {"response": [{"contract": {...}, "data": [{...}, ...]}, ...]}
                    if "response" in data and data["response"]:
                        results = []
                        # The response contains contract info + data array
                        contract_response = data["response"][0] if data["response"] else {}
                        logger.info(f"ThetaData contract_response keys: {contract_response.keys() if isinstance(contract_response, dict) else type(contract_response)}")
                        ohlc_data = contract_response.get("data", [])
                        logger.info(f"ThetaData ohlc_data count: {len(ohlc_data)}, first row: {ohlc_data[0] if ohlc_data else 'EMPTY'}")

                        for row in ohlc_data:
                            if not row:
                                continue

                            # Parse timestamp (format: "2024-10-01T09:30:00")
                            ts_str = row.get("timestamp", "")
                            if not ts_str:
                                continue

                            try:
                                bar_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            except ValueError:
                                continue

                            open_price = row.get("open", 0)
                            high_price = row.get("high", 0)
                            low_price = row.get("low", 0)
                            close_price = row.get("close", 0)

                            # Skip rows with no price data
                            if not any([open_price, high_price, low_price, close_price]):
                                continue

                            # Calculate midpoint from high/low or use close
                            midpoint = (high_price + low_price) / 2 if high_price and low_price else close_price

                            results.append({
                                "timestamp": bar_time,
                                "midpoint": midpoint,
                                "open": open_price,
                                "high": high_price,
                                "low": low_price,
                                "close": close_price,
                            })

                        # Cache the results
                        self._intraday_cache[cache_key] = results
                        return results

        except Exception as e:
            logger.debug(f"Error fetching intraday option data: {e}")

        return []

    def interpolate_option_price(
        self,
        intraday_data: list[dict],
        bar_time: datetime,
    ) -> Optional[float]:
        """
        Interpolate option price at bar_time from intraday snapshots.

        Uses linear interpolation between the two surrounding data points.
        Returns closest available price if at boundary.

        Args:
            intraday_data: List of intraday snapshots with timestamp and midpoint
            bar_time: The time to get price for (assumed UTC from Alpaca)

        Returns:
            Interpolated mid price or None if no data available
        """
        if not intraday_data:
            return None

        # Parse timestamps and filter valid data points
        valid_points = []
        for point in intraday_data:
            mid = point.get("midpoint", 0)
            if mid <= 0:
                continue

            ts = point.get("timestamp")
            if not ts:
                continue

            try:
                # Handle both string and datetime timestamps
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                valid_points.append((ts, mid))
            except (ValueError, TypeError):
                continue

        if not valid_points:
            logger.info(f"interpolate_option_price: No valid points found from {len(intraday_data)} data points")
            return None

        # Sort by timestamp
        valid_points.sort(key=lambda x: x[0])
        logger.info(f"interpolate_option_price: {len(valid_points)} valid points, first={valid_points[0]}, last={valid_points[-1]}")
        logger.info(f"interpolate_option_price: bar_time={bar_time}, tzinfo={bar_time.tzinfo}")

        # Convert bar_time from UTC to ET for comparison with ThetaData timestamps
        # ThetaData returns timestamps in ET without timezone info
        # Alpaca bars come in UTC
        if bar_time.tzinfo is not None:
            # Convert UTC to ET
            et_tz = ZoneInfo("America/New_York")
            bar_time_et = bar_time.astimezone(et_tz).replace(tzinfo=None)
        else:
            # Already naive, assume it's in the right timezone
            bar_time_et = bar_time

        # Make bar_time timezone-naive for comparison
        bar_time_naive = bar_time_et

        # Find surrounding points for interpolation
        before = None
        after = None

        for ts, mid in valid_points:
            ts_naive = ts.replace(tzinfo=None) if ts.tzinfo else ts
            if ts_naive <= bar_time_naive:
                before = (ts_naive, mid)
            elif after is None:
                after = (ts_naive, mid)
                break

        # Return closest or interpolated price
        if before is None and after is None:
            return None
        elif before is None:
            return after[1]
        elif after is None:
            return before[1]
        else:
            # Linear interpolation
            before_ts, before_price = before
            after_ts, after_price = after

            total_seconds = (after_ts - before_ts).total_seconds()
            if total_seconds <= 0:
                return before_price

            elapsed_seconds = (bar_time_naive - before_ts).total_seconds()
            fraction = elapsed_seconds / total_seconds

            interpolated = before_price + fraction * (after_price - before_price)
            return max(0.01, interpolated)  # Price can't go below $0.01

    def estimate_initial_option_price(
        self, signal_type: SignalType, underlying_price: float
    ) -> float:
        """
        Estimate initial OTM option price (simplified model).

        For 0DTE $20 OTM options, price is lower than ATM due to no intrinsic value.
        This is a rough approximation - real backtest uses historical options data.

        Args:
            signal_type: BUY_CALL or BUY_PUT
            underlying_price: Current price of underlying

        Returns:
            Estimated option price
        """
        # OTM options are cheaper - roughly 0.3-0.5% of underlying for $20 OTM
        base_pct = 0.004  # 0.4% of underlying (~$2.40 for SPY at $600)
        return underlying_price * base_pct

    def estimate_option_price(
        self,
        current_underlying: float,
        entry_underlying: float,
        entry_option_price: float,
        signal_type: SignalType,
        minutes_elapsed: float = 0,
        total_minutes_to_expiry: float = 390,  # ~6.5 hours trading day
    ) -> float:
        """
        Estimate option price change based on underlying movement AND theta decay.

        For 0DTE options, theta decay is massive - options can lose 50-80% of
        extrinsic value during the trading day. This model includes:
        - Delta: Price change from underlying movement (~0.3 for OTM)
        - Theta: Time decay using square root decay model

        Args:
            current_underlying: Current underlying price
            entry_underlying: Underlying price at entry
            entry_option_price: Option price at entry
            signal_type: BUY_CALL or BUY_PUT
            minutes_elapsed: Minutes since trade entry
            total_minutes_to_expiry: Total trading minutes in day (~390)

        Returns:
            Estimated current option price
        """
        underlying_change = current_underlying - entry_underlying

        # Delta component: $20 OTM options have ~0.3 delta
        delta = 0.3 if signal_type == SignalType.BUY_CALL else -0.3
        delta_pnl = delta * underlying_change

        # Theta decay for 0DTE options
        # Options lose 50-80% of extrinsic value during the trading day
        if total_minutes_to_expiry > 0 and minutes_elapsed > 0:
            time_remaining_pct = max(0, (total_minutes_to_expiry - minutes_elapsed) / total_minutes_to_expiry)
            # Square root decay - accelerates as expiration approaches
            decay_factor = time_remaining_pct ** 0.5

            # OTM options are ~90% extrinsic (time) value, ~10% intrinsic
            extrinsic_pct = 0.9
            intrinsic_pct = 0.1

            # New price = stable intrinsic + decayed extrinsic
            theta_adjusted_price = entry_option_price * (intrinsic_pct + extrinsic_pct * decay_factor)
        else:
            theta_adjusted_price = entry_option_price

        # Apply delta change on top of theta-adjusted price
        new_price = theta_adjusted_price + delta_pnl

        return max(0.01, new_price)

    def get_current_price(
        self,
        intraday_data: list[dict],
        bar_time: datetime,
        current_underlying: float,
        entry_underlying: float,
        entry_option_price: float,
        signal_type: SignalType,
        entry_time: datetime = None,
    ) -> Optional[float]:
        """
        Get current option price from real ThetaData only.

        IMPORTANT: No estimation fallback. Returns None if real data unavailable.
        This ensures backtest accuracy - trades without real pricing data are skipped.

        Args:
            intraday_data: Intraday option price data (REQUIRED - no fallback)
            bar_time: Current bar timestamp
            current_underlying: Current underlying price (unused - kept for API compatibility)
            entry_underlying: Underlying price at entry (unused - kept for API compatibility)
            entry_option_price: Option price at entry (unused - kept for API compatibility)
            signal_type: BUY_CALL or BUY_PUT (unused - kept for API compatibility)
            entry_time: Trade entry timestamp (unused - kept for API compatibility)

        Returns:
            Current option price from ThetaData, or None if unavailable
        """
        # ONLY use real ThetaData - no estimation fallback
        if not intraday_data:
            return None

        return self.interpolate_option_price(intraday_data, bar_time)
