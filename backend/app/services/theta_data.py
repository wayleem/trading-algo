"""Theta Data REST API client for options data (v3 API).

This client interfaces with the Theta Terminal REST API running locally.
Theta Terminal exposes REST endpoints on localhost:25503.

Key endpoints used (v3 API):
- /v3/option/list/expirations - Get available option expirations
- /v3/option/list/strikes - Get available strikes for an expiration
- /v3/option/snapshot/quote - Get current bid/ask
- /v3/option/snapshot/greeks/all - Get IV, delta, and other greeks
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class ThetaDataError(Exception):
    """Error from Theta Data API."""

    pass


@dataclass
class OptionQuote:
    """Option quote data."""

    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None


@dataclass
class OptionGreeks:
    """Option greeks and IV."""

    implied_volatility: float  # Annualized IV (e.g., 0.30 for 30%)
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    underlying_price: float


@dataclass
class HistoricalGreeks:
    """Historical Greeks data point (EOD snapshot)."""

    date: date
    implied_volatility: float
    delta: float
    underlying_price: float
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class OptionEOD:
    """End-of-day option data for P&L simulation."""

    date: date
    bid: float
    ask: float
    mid: float
    close: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None


class ThetaDataClient:
    """
    Async client for Theta Data REST API.

    The Theta Terminal must be running locally to use this client.
    It exposes REST endpoints on the configured base URL (default: localhost:25510).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize client.

        Args:
            base_url: Theta Terminal API base URL
            api_key: Theta Data API key (for authentication)
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or settings.theta_data_base_url).rstrip("/")
        self.api_key = api_key or settings.theta_data_api_key
        self.client = httpx.AsyncClient(timeout=timeout)
        # In-memory cache for historical data (immutable once fetched)
        self._cache: Dict[str, Any] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        # Sentinel value to distinguish "not in cache" from "cached None"
        self._CACHE_MISS = object()

    def _cache_key(self, method: str, **kwargs) -> str:
        """Generate unique cache key from method + params."""
        # Convert date objects to strings for consistent hashing
        processed = {}
        for k, v in sorted(kwargs.items()):
            if isinstance(v, date):
                processed[k] = v.isoformat()
            else:
                processed[k] = str(v)
        key_data = f"{method}:{processed}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached(self, key: str) -> Any:
        """Get value from cache. Returns _CACHE_MISS sentinel if not found."""
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        self._cache_misses += 1
        return self._CACHE_MISS

    def _set_cached(self, key: str, value: Any) -> None:
        """Store value in cache."""
        self._cache[key] = value

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache hit/miss statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate_pct": hit_rate,
            "cached_entries": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
    ) -> dict:
        """
        Make authenticated request to Theta Data API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            ThetaDataError: On API errors
        """
        params = params or {}

        url = f"{self.base_url}{endpoint}"

        try:
            response = await self.client.get(url, params=params)

            if response.status_code == 429:
                raise ThetaDataError("Rate limited - please retry after delay")
            elif response.status_code == 401:
                raise ThetaDataError("Invalid API key")
            elif response.status_code == 404:
                raise ThetaDataError(f"Endpoint not found: {endpoint}")

            response.raise_for_status()

            data = response.json()

            # Check for API-level errors in response
            if isinstance(data, dict) and data.get("error"):
                raise ThetaDataError(f"API error: {data.get('error')}")

            return data

        except httpx.TimeoutException:
            raise ThetaDataError(f"Request timeout for {endpoint}")
        except httpx.RequestError as e:
            raise ThetaDataError(f"Connection error: {e}")
        except httpx.HTTPStatusError as e:
            raise ThetaDataError(f"HTTP error {e.response.status_code}: {e}")

    async def get_expirations(self, symbol: str) -> List[date]:
        """
        Get available expiration dates for an option chain.

        Args:
            symbol: Underlying ticker symbol (e.g., "SPY")

        Returns:
            List of expiration dates sorted ascending (future dates only)
        """
        try:
            data = await self._make_request(
                "/v3/option/list/expirations",
                params={"symbol": symbol.upper(), "format": "json"},
            )

            # Parse v3 response format: {"response": [{"symbol": "SPY", "expiration": "2024-01-19"}, ...]}
            expirations = []
            today = date.today()

            if isinstance(data, dict):
                response_data = data.get("response", [])
            else:
                response_data = data

            for item in response_data:
                if isinstance(item, dict):
                    date_str = item.get("expiration", "")
                else:
                    date_str = str(item)

                if date_str:
                    try:
                        # v3 uses YYYY-MM-DD format
                        exp_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        # Only include future expirations
                        if exp_date > today:
                            expirations.append(exp_date)
                    except ValueError:
                        # Try YYYYMMDD format as fallback
                        try:
                            exp_date = datetime.strptime(date_str, "%Y%m%d").date()
                            if exp_date > today:
                                expirations.append(exp_date)
                        except ValueError:
                            logger.warning(f"Could not parse date: {date_str}")

            return sorted(expirations)

        except ThetaDataError:
            raise
        except Exception as e:
            raise ThetaDataError(f"Failed to get expirations for {symbol}: {e}")

    async def get_strikes(self, symbol: str, expiration: date) -> List[float]:
        """
        Get available strikes for a given expiration.

        Args:
            symbol: Underlying ticker symbol
            expiration: Expiration date

        Returns:
            List of strike prices sorted ascending
        """
        try:
            exp_str = expiration.strftime("%Y-%m-%d")

            data = await self._make_request(
                "/v3/option/list/strikes",
                params={"symbol": symbol.upper(), "expiration": exp_str, "format": "json"},
            )

            # Parse v3 response format: {"response": [{"symbol": "SPY", "strike": 450.0}, ...]}
            strikes = []

            if isinstance(data, dict):
                response_data = data.get("response", [])
            else:
                response_data = data

            for item in response_data:
                if isinstance(item, dict):
                    strike = item.get("strike")
                else:
                    strike = item

                if strike is not None:
                    strike_val = float(strike)
                    strikes.append(strike_val)

            return sorted(set(strikes))  # Remove duplicates and sort

        except ThetaDataError:
            raise
        except Exception as e:
            raise ThetaDataError(f"Failed to get strikes for {symbol} {expiration}: {e}")

    async def get_option_quote(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
    ) -> Optional[OptionQuote]:
        """
        Get current bid/ask quote for an option contract.

        Args:
            symbol: Underlying ticker symbol
            expiration: Expiration date
            strike: Strike price in dollars (e.g., 450.0)
            right: "C" for call, "P" for put

        Returns:
            OptionQuote with bid/ask data, or None if not found
        """
        try:
            exp_str = expiration.strftime("%Y-%m-%d")
            # v3 API uses strike in dollars with decimals (e.g., "450.000")
            strike_str = f"{strike:.3f}"
            # v3 uses "call" or "put" instead of "C" or "P"
            right_str = "call" if right.upper() == "C" else "put"

            data = await self._make_request(
                "/v3/option/snapshot/quote",
                params={
                    "symbol": symbol.upper(),
                    "expiration": exp_str,
                    "strike": strike_str,
                    "right": right_str,
                    "format": "json",
                },
            )

            # Parse v3 response - nested structure: response[0].data[0]
            if isinstance(data, dict):
                response_data = data.get("response", [])
            else:
                response_data = data if isinstance(data, list) else [data]

            if not response_data:
                return None

            # v3 format: response is array of {contract, data} objects
            first_response = response_data[0] if isinstance(response_data, list) else response_data

            # Extract the actual data from nested structure
            if isinstance(first_response, dict) and "data" in first_response:
                data_array = first_response.get("data", [])
                if not data_array:
                    return None
                quote_data = data_array[0]
            else:
                quote_data = first_response

            # v3 field names: bid, ask, bid_size, ask_size
            bid = self._extract_price(quote_data, ["bid", "bid_price"])
            ask = self._extract_price(quote_data, ["ask", "ask_price"])

            if bid is None or ask is None:
                return None

            return OptionQuote(
                bid=bid,
                ask=ask,
                bid_size=int(quote_data.get("bid_size", 0)),
                ask_size=int(quote_data.get("ask_size", 0)),
                last=self._extract_price(quote_data, ["last", "last_price"]),
                volume=quote_data.get("volume"),
                open_interest=quote_data.get("open_interest"),
            )

        except ThetaDataError:
            raise
        except Exception as e:
            logger.warning(f"Failed to get quote for {symbol} {expiration} {strike} {right}: {e}")
            return None

    async def get_option_greeks(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
    ) -> Optional[OptionGreeks]:
        """
        Get greeks and IV for an option contract.

        Args:
            symbol: Underlying ticker symbol
            expiration: Expiration date
            strike: Strike price in dollars
            right: "C" for call, "P" for put

        Returns:
            OptionGreeks with IV and greeks, or None if not found
        """
        try:
            exp_str = expiration.strftime("%Y-%m-%d")
            strike_str = f"{strike:.3f}"
            right_str = "call" if right.upper() == "C" else "put"

            data = await self._make_request(
                "/v3/option/snapshot/greeks/all",
                params={
                    "symbol": symbol.upper(),
                    "expiration": exp_str,
                    "strike": strike_str,
                    "right": right_str,
                    "format": "json",
                },
            )

            # Parse v3 response - nested structure: response[0].data[0]
            if isinstance(data, dict):
                response_data = data.get("response", [])
            else:
                response_data = data if isinstance(data, list) else [data]

            if not response_data:
                return None

            # v3 format: response is array of {contract, data} objects
            first_response = response_data[0] if isinstance(response_data, list) else response_data

            # Extract the actual data from nested structure
            if isinstance(first_response, dict) and "data" in first_response:
                data_array = first_response.get("data", [])
                if not data_array:
                    return None
                greeks_data = data_array[0]
            else:
                greeks_data = first_response

            # v3 field names: implied_vol (not implied_volatility), delta, gamma, theta, vega, rho
            iv = self._extract_iv(greeks_data)
            underlying = self._extract_price(
                greeks_data, ["underlying_price", "stock_price", "spot"]
            )

            if underlying is None:
                return None

            # IV can be 0 for deep ITM options - still return the data
            if iv is None:
                iv = 0.0

            return OptionGreeks(
                implied_volatility=iv,
                delta=float(greeks_data.get("delta", 0)),
                gamma=float(greeks_data.get("gamma", 0)),
                theta=float(greeks_data.get("theta", 0)),
                vega=float(greeks_data.get("vega", 0)),
                rho=float(greeks_data.get("rho", 0)),
                underlying_price=underlying,
            )

        except ThetaDataError:
            raise
        except Exception as e:
            logger.warning(f"Failed to get greeks for {symbol} {expiration} {strike} {right}: {e}")
            return None

    async def get_stock_price(self, symbol: str) -> Optional[float]:
        """
        Get current stock price.

        Tries stock quote endpoint first, then falls back to deriving
        price from option greeks (underlying_price field).

        Args:
            symbol: Ticker symbol

        Returns:
            Current stock price, or None if not available
        """
        # Try stock quote endpoint first
        try:
            data = await self._make_request(
                "/v3/stock/snapshot/quote",
                params={"symbol": symbol.upper(), "format": "json"},
            )

            if isinstance(data, dict):
                response_data = data.get("response", [])
            else:
                response_data = data if isinstance(data, list) else [data]

            if response_data:
                quote = response_data[0] if isinstance(response_data, list) else response_data
                price = self._extract_price(quote, ["last", "price", "mid", "close"])
                if price:
                    return price

        except ThetaDataError as e:
            logger.debug(f"Stock quote endpoint failed for {symbol}: {e}")
        except Exception as e:
            logger.debug(f"Failed to get stock price for {symbol}: {e}")

        # Fallback: derive price from option greeks (works with FREE subscription)
        try:
            expirations = await self.get_expirations(symbol)
            if expirations:
                strikes = await self.get_strikes(symbol, expirations[0])
                if strikes:
                    # Use middle strike as estimate
                    mid_strike = strikes[len(strikes) // 2]
                    greeks = await self.get_option_greeks(
                        symbol, expirations[0], mid_strike, "C"
                    )
                    if greeks and greeks.underlying_price:
                        logger.debug(f"Got {symbol} price from options: ${greeks.underlying_price:.2f}")
                        return greeks.underlying_price
        except Exception as e:
            logger.debug(f"Fallback stock price failed for {symbol}: {e}")

        return None

    async def get_historical_stock_price(
        self,
        symbol: str,
        target_date: date,
    ) -> Optional[float]:
        """
        Get historical stock price for a specific date.

        Derives the price from historical option greeks (underlying_price field).

        Args:
            symbol: Ticker symbol
            target_date: The date to get the price for

        Returns:
            Historical stock price, or None if not available
        """
        try:
            # Get expirations that exist at target_date
            expirations = await self.get_expirations(symbol)
            if not expirations:
                return None

            # Find an expiration that was valid on target_date
            valid_exp = None
            for exp in expirations:
                if exp > target_date:
                    valid_exp = exp
                    break

            if not valid_exp:
                # Try the last available expiration
                valid_exp = expirations[-1] if expirations else None

            if not valid_exp:
                return None

            # Get strikes for that expiration
            strikes = await self.get_strikes(symbol, valid_exp)
            if not strikes:
                return None

            # Use middle strike
            mid_strike = strikes[len(strikes) // 2]

            # Get historical greeks for that date
            greeks = await self.get_historical_greeks(
                symbol=symbol,
                expiration=valid_exp,
                strike=mid_strike,
                right="C",
                start_date=target_date,
                end_date=target_date,
            )

            if greeks and len(greeks) > 0 and greeks[0].underlying_price:
                return greeks[0].underlying_price

        except Exception as e:
            logger.debug(f"Failed to get historical stock price for {symbol} on {target_date}: {e}")

        return None

    async def get_historical_greeks(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
        start_date: date,
        end_date: date,
    ) -> List[HistoricalGreeks]:
        """
        Get historical IV/Greeks for an option contract over a date range.

        Uses EOD (end of day) data for efficiency - one data point per trading day.

        Args:
            symbol: Underlying ticker symbol
            expiration: Option expiration date
            strike: Strike price in dollars
            right: "C" for call, "P" for put
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of HistoricalGreeks for each trading day in range
        """
        try:
            exp_str = expiration.strftime("%Y-%m-%d")
            strike_str = f"{strike:.3f}"
            right_str = "call" if right.upper() == "C" else "put"
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # v3 API requires one date per request, so we need to iterate
            # For EOD data, we'll fetch each date in the range
            current_date = start_date
            all_data = []

            while current_date <= end_date:
                # Skip weekends
                if current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                    continue

                try:
                    day_data = await self._make_request(
                        "/v3/option/history/greeks/implied_volatility",
                        params={
                            "symbol": symbol.upper(),
                            "expiration": expiration.strftime("%Y%m%d"),
                            "strike": f"{strike:.3f}",  # v3 uses decimal string
                            "right": "call" if right.upper() == "C" else "put",
                            "date": current_date.strftime("%Y%m%d"),
                            "interval": "1h",  # Use hourly for EOD approximation
                            "format": "json",  # Request JSON format
                        },
                    )
                    if day_data:
                        all_data.append((current_date, day_data))
                except Exception as e:
                    logger.debug(f"No data for {current_date}: {e}")

                current_date += timedelta(days=1)

            results = []

            # Process each day's data from multi-day fetch
            for day_date, day_data in all_data:
                try:
                    # v3 JSON format: {"response": [{"contract": {...}, "data": [{...}, ...]}]}
                    if not isinstance(day_data, dict):
                        continue

                    response_list = day_data.get("response", [])
                    if not response_list:
                        continue

                    # Get the first contract's data
                    contract_data = response_list[0] if response_list else None
                    if not contract_data or not isinstance(contract_data, dict):
                        continue

                    data_points = contract_data.get("data", [])
                    if not data_points:
                        continue

                    # Get the last data point (EOD) from the day's data
                    last_point = data_points[-1] if data_points else None
                    if not last_point or not isinstance(last_point, dict):
                        continue

                    # v3 JSON format fields
                    iv = last_point.get("implied_vol")
                    underlying = last_point.get("underlying_price")
                    bid = last_point.get("bid")
                    ask = last_point.get("ask")

                    # Parse IV (v3 returns decimal format, e.g., 0.1178 = 11.78%)
                    if iv is None or iv == 0:
                        continue
                    iv_float = float(iv)

                    # Parse underlying price (v3 returns dollars)
                    if underlying is None:
                        continue
                    underlying_float = float(underlying)

                    # Parse optional bid/ask (v3 returns dollars)
                    bid_float = float(bid) if bid is not None else None
                    ask_float = float(ask) if ask is not None else None

                    results.append(
                        HistoricalGreeks(
                            date=day_date,
                            implied_volatility=iv_float,
                            delta=0.0,  # Not available in implied_volatility endpoint
                            underlying_price=underlying_float,
                            bid=bid_float,
                            ask=ask_float,
                        )
                    )

                except (ValueError, TypeError, IndexError, KeyError) as e:
                    logger.debug(f"Error parsing data for {day_date}: {e}")
                    continue

            # Sort by date
            results.sort(key=lambda x: x.date)
            return results

        except ThetaDataError:
            raise
        except Exception as e:
            logger.warning(
                f"Failed to get historical greeks for {symbol} {expiration} {strike} {right}: {e}"
            )
            return []

    async def get_historical_option_eod(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
        target_date: date,
    ) -> Optional[OptionEOD]:
        """
        Get end-of-day option data for a specific historical date.

        Uses v3 implied_volatility endpoint which returns bid/ask/midpoint.
        Results are cached in-memory since historical data is immutable.

        Args:
            symbol: Underlying ticker symbol
            expiration: Option expiration date
            strike: Strike price in dollars
            right: "C" for call, "P" for put
            target_date: Historical date to fetch EOD data for

        Returns:
            OptionEOD with bid, ask, midpoint prices, or None if not found
        """
        # Check cache first (historical data is immutable)
        cache_key = self._cache_key(
            "historical_eod",
            symbol=symbol,
            expiration=expiration,
            strike=strike,
            right=right,
            target_date=target_date,
        )
        cached = self._get_cached(cache_key)
        if cached is not self._CACHE_MISS:
            return cached

        try:
            data = await self._make_request(
                "/v3/option/history/greeks/implied_volatility",
                params={
                    "symbol": symbol.upper(),
                    "expiration": expiration.strftime("%Y%m%d"),
                    "strike": f"{strike:.3f}",
                    "right": "call" if right.upper() == "C" else "put",
                    "date": target_date.strftime("%Y%m%d"),
                    "interval": "1h",
                    "format": "json",
                },
            )

            if not data or not isinstance(data, dict):
                self._set_cached(cache_key, None)
                return None

            # v3 JSON format: {"response": [{"contract": {...}, "data": [{...}, ...]}]}
            response_list = data.get("response", [])
            if not response_list:
                self._set_cached(cache_key, None)
                return None

            contract_data = response_list[0] if response_list else None
            if not contract_data or not isinstance(contract_data, dict):
                self._set_cached(cache_key, None)
                return None

            data_points = contract_data.get("data", [])
            if not data_points:
                self._set_cached(cache_key, None)
                return None

            # Get the last data point (EOD) from the day's data
            last_point = data_points[-1] if data_points else None
            if not last_point or not isinstance(last_point, dict):
                self._set_cached(cache_key, None)
                return None

            try:
                bid = float(last_point.get("bid", 0))
                ask = float(last_point.get("ask", 0))
                midpoint = float(last_point.get("midpoint", 0))

                # Validate we have actual prices
                if bid == 0 and ask == 0:
                    self._set_cached(cache_key, None)
                    return None

                # Use midpoint if provided, otherwise calculate
                if midpoint == 0 and bid > 0 and ask > 0:
                    midpoint = (bid + ask) / 2

                result = OptionEOD(
                    date=target_date,
                    bid=bid,
                    ask=ask,
                    mid=midpoint,
                    close=midpoint,  # Use midpoint as close approximation
                    open=None,
                    high=None,
                    low=None,
                    volume=None,
                )
                # Cache successful result
                self._set_cached(cache_key, result)
                return result

            except (ValueError, TypeError) as e:
                logger.debug(f"Error parsing EOD data for {target_date}: {e}")
                # Cache None for failed lookups to avoid re-querying
                self._set_cached(cache_key, None)
                return None

        except ThetaDataError:
            raise
        except Exception as e:
            logger.debug(
                f"Failed to get historical EOD for {symbol} {expiration} {strike} {right} on {target_date}: {e}"
            )
            # Cache None for failed lookups
            self._set_cached(cache_key, None)
            return None

    async def get_historical_option_eod_range(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
        start_date: date,
        end_date: date,
    ) -> List[OptionEOD]:
        """
        Get EOD option data for a date range.

        More efficient than calling get_historical_option_eod for each date.

        Args:
            symbol: Underlying ticker symbol
            expiration: Option expiration date
            strike: Strike price in dollars
            right: "C" for call, "P" for put
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of OptionEOD for trading days in range
        """
        try:
            strike_int = int(strike * 1000)

            data = await self._make_request(
                "/v2/hist/option/eod",
                params={
                    "root": symbol.upper(),
                    "exp": expiration.strftime("%Y%m%d"),
                    "strike": strike_int,
                    "right": right.upper(),
                    "start_date": start_date.strftime("%Y%m%d"),
                    "end_date": end_date.strftime("%Y%m%d"),
                },
            )

            if not data:
                return []

            if isinstance(data, dict):
                response_data = data.get("response", [])
            else:
                response_data = data if isinstance(data, list) else []

            results = []
            for row in response_data:
                if not row or not isinstance(row, list) or len(row) < 6:
                    continue

                try:
                    # Parse date from row[0] (format: YYYYMMDD)
                    date_val = row[0]
                    if isinstance(date_val, int):
                        date_str = str(date_val)
                        row_date = datetime.strptime(date_str, "%Y%m%d").date()
                    else:
                        row_date = datetime.strptime(str(date_val), "%Y%m%d").date()

                    # Parse prices
                    open_price = float(row[2]) / 100 if row[2] else None
                    high_price = float(row[3]) / 100 if row[3] else None
                    low_price = float(row[4]) / 100 if row[4] else None
                    close_price = float(row[5]) / 100 if row[5] else 0.0
                    volume = int(row[6]) if len(row) > 6 and row[6] else None

                    bid = float(row[8]) / 100 if len(row) > 8 and row[8] else close_price
                    ask = float(row[9]) / 100 if len(row) > 9 and row[9] else close_price

                    mid = (bid + ask) / 2 if bid and ask else close_price

                    results.append(OptionEOD(
                        date=row_date,
                        bid=bid,
                        ask=ask,
                        mid=mid,
                        close=close_price,
                        open=open_price,
                        high=high_price,
                        low=low_price,
                        volume=volume,
                    ))

                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Error parsing EOD row: {e}")
                    continue

            results.sort(key=lambda x: x.date)
            return results

        except ThetaDataError:
            raise
        except Exception as e:
            logger.warning(
                f"Failed to get historical EOD range for {symbol} {expiration} {strike} {right}: {e}"
            )
            return []

    def _extract_price(self, data: dict, keys: List[str]) -> Optional[float]:
        """Extract price from response data, trying multiple possible keys."""
        for key in keys:
            value = data.get(key)
            if value is not None:
                try:
                    price = float(value)
                    # Prices in cents need conversion
                    if price > 10000:
                        price = price / 100
                    return price
                except (ValueError, TypeError):
                    continue
        return None

    def _extract_iv(self, data: dict) -> Optional[float]:
        """Extract implied volatility from response data."""
        iv_keys = ["implied_vol", "implied_volatility", "impliedVolatility", "iv", "impl_vol"]

        for key in iv_keys:
            value = data.get(key)
            if value is not None:
                try:
                    iv = float(value)
                    # IV might be in percentage (30) or decimal (0.30)
                    if iv > 5:  # Likely percentage
                        iv = iv / 100
                    return iv
                except (ValueError, TypeError):
                    continue
        return None


# Singleton instance for convenience
_client: Optional[ThetaDataClient] = None


def get_theta_client() -> ThetaDataClient:
    """Get or create singleton Theta Data client."""
    global _client
    if _client is None:
        _client = ThetaDataClient()
    return _client
