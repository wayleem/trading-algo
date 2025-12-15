"""
Technical Indicators Module.

Provides RSI, SMA, MACD, Bollinger Bands, and other technical indicator
calculations for trading signals and analysis.
"""

from typing import Optional
import numpy as np


class IndicatorService:
    """Calculate technical indicators from price data."""

    def __init__(self, rsi_period: int = 14, sma_period: int = 14):
        self.rsi_period = rsi_period
        self.sma_period = sma_period

    def calculate_rsi(self, closes: list[float]) -> float:
        """
        Calculate RSI from closing prices using Wilder's smoothing.

        Args:
            closes: List of closing prices (oldest first)

        Returns:
            RSI value between 0 and 100
        """
        if len(closes) < self.rsi_period + 1:
            return 50.0  # Neutral if insufficient data

        prices = np.array(closes)
        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Use Wilder's smoothing (exponential moving average)
        alpha = 1.0 / self.rsi_period

        avg_gain = self._wilder_smooth(gains, self.rsi_period)
        avg_loss = self._wilder_smooth(losses, self.rsi_period)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return float(rsi)

    def _wilder_smooth(self, values: np.ndarray, period: int) -> float:
        """Calculate Wilder's smoothed average."""
        if len(values) < period:
            return np.mean(values) if len(values) > 0 else 0.0

        # Initial SMA for first period
        initial_avg = np.mean(values[:period])

        # Apply Wilder's smoothing for remaining values
        smoothed = initial_avg
        for i in range(period, len(values)):
            smoothed = (smoothed * (period - 1) + values[i]) / period

        return float(smoothed)

    def calculate_sma(self, values: list[float]) -> float:
        """
        Calculate Simple Moving Average.

        Args:
            values: List of values (oldest first)

        Returns:
            SMA value
        """
        if not values:
            return 0.0

        if len(values) < self.sma_period:
            return sum(values) / len(values)

        return sum(values[-self.sma_period :]) / self.sma_period

    def detect_crossover(
        self,
        current_rsi: float,
        previous_rsi: float,
        current_sma: float,
        previous_sma: float,
    ) -> Optional[str]:
        """
        Detect RSI crossing its SMA.

        Args:
            current_rsi: Current RSI value
            previous_rsi: Previous RSI value
            current_sma: Current SMA of RSI
            previous_sma: Previous SMA of RSI

        Returns:
            "bullish" - RSI crossed ABOVE SMA
            "bearish" - RSI crossed BELOW SMA
            None - No crossover
        """
        # Bullish crossover: RSI was at/below SMA, now above
        if previous_rsi <= previous_sma and current_rsi > current_sma:
            return "bullish"

        # Bearish crossover: RSI was at/above SMA, now below
        if previous_rsi >= previous_sma and current_rsi < current_sma:
            return "bearish"

        return None


def calculate_rsi_series(closes: list[float], period: int = 14) -> list[float]:
    """
    Calculate RSI for each bar in the series.

    Args:
        closes: List of closing prices
        period: RSI period

    Returns:
        List of RSI values (same length as closes, with NaN for insufficient data)
    """
    if len(closes) < period + 1:
        return [float("nan")] * len(closes)

    prices = np.array(closes)
    deltas = np.diff(prices)

    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi_values = [float("nan")] * (period)

    # First RSI value using SMA
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

    # Subsequent values using Wilder's smoothing
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

    return rsi_values


def calculate_sma_series(values: list[float], period: int = 14) -> list[float]:
    """
    Calculate SMA for each point in the series.

    Args:
        values: List of values
        period: SMA period

    Returns:
        List of SMA values (same length as values, with NaN for insufficient data)
    """
    if len(values) < period:
        return [float("nan")] * len(values)

    sma_values = [float("nan")] * (period - 1)

    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        valid_values = [v for v in window if not np.isnan(v)]
        if len(valid_values) == period:
            sma_values.append(sum(valid_values) / period)
        else:
            sma_values.append(float("nan"))

    return sma_values


def calculate_ema_series(values: list[float], period: int) -> list[float]:
    """
    Calculate Exponential Moving Average for each point in the series.

    Uses standard EMA formula: EMA_t = price * k + EMA_{t-1} * (1-k)
    where k = 2 / (period + 1)

    Handles leading NaN values by finding the first valid window of 'period' values.

    Args:
        values: List of values (oldest first)
        period: EMA period

    Returns:
        List of EMA values (same length as values, with NaN for insufficient data)
    """
    n = len(values)
    if n < period:
        return [float("nan")] * n

    ema_values = [float("nan")] * n

    # Find the first index where we have 'period' consecutive valid values
    first_valid_start = None
    for start_idx in range(n - period + 1):
        window = values[start_idx:start_idx + period]
        if all(not np.isnan(v) for v in window):
            first_valid_start = start_idx
            break

    if first_valid_start is None:
        return ema_values  # Not enough valid data

    # First EMA is SMA of first valid window
    first_window = values[first_valid_start:first_valid_start + period]
    ema = sum(first_window) / period
    ema_values[first_valid_start + period - 1] = ema

    # Multiplier for EMA
    k = 2.0 / (period + 1)

    # Calculate remaining EMA values
    for i in range(first_valid_start + period, n):
        if np.isnan(values[i]):
            ema_values[i] = float("nan")
        else:
            ema = values[i] * k + ema * (1 - k)
            ema_values[i] = ema

    return ema_values


def calculate_macd_series(
    closes: list[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[list[float], list[float], list[float]]:
    """
    Calculate MACD indicator series.

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(signal_period) of MACD Line
    Histogram = MACD Line - Signal Line

    Args:
        closes: List of closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram), each same length as closes
    """
    n = len(closes)

    if n < slow_period:
        return ([float("nan")] * n, [float("nan")] * n, [float("nan")] * n)

    # Calculate EMAs
    ema_fast = calculate_ema_series(closes, fast_period)
    ema_slow = calculate_ema_series(closes, slow_period)

    # MACD Line = EMA_fast - EMA_slow
    macd_line = []
    for i in range(n):
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            macd_line.append(float("nan"))
        else:
            macd_line.append(ema_fast[i] - ema_slow[i])

    # Signal Line = EMA of MACD Line
    signal_line = calculate_ema_series(macd_line, signal_period)

    # Histogram = MACD - Signal
    histogram = []
    for i in range(n):
        if np.isnan(macd_line[i]) or np.isnan(signal_line[i]):
            histogram.append(float("nan"))
        else:
            histogram.append(macd_line[i] - signal_line[i])

    return (macd_line, signal_line, histogram)


def calculate_bollinger_bands_series(
    closes: list[float],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[list[float], list[float], list[float]]:
    """
    Calculate Bollinger Bands series.

    Middle Band = SMA(period)
    Upper Band = Middle + num_std * StdDev(period)
    Lower Band = Middle - num_std * StdDev(period)

    Args:
        closes: List of closing prices
        period: SMA period for middle band (default 20)
        num_std: Number of standard deviations for bands (default 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band), each same length as closes
    """
    n = len(closes)

    if n < period:
        return ([float("nan")] * n, [float("nan")] * n, [float("nan")] * n)

    upper_band = [float("nan")] * (period - 1)
    middle_band = [float("nan")] * (period - 1)
    lower_band = [float("nan")] * (period - 1)

    prices = np.array(closes)

    for i in range(period - 1, n):
        window = prices[i - period + 1 : i + 1]
        sma = np.mean(window)
        std = np.std(window, ddof=0)  # Population std dev

        middle_band.append(float(sma))
        upper_band.append(float(sma + num_std * std))
        lower_band.append(float(sma - num_std * std))

    return (upper_band, middle_band, lower_band)


def calculate_band_width_series(
    upper: list[float],
    lower: list[float],
    middle: list[float],
) -> list[float]:
    """
    Calculate Bollinger Band width as percentage.

    Band Width = (Upper - Lower) / Middle * 100

    Args:
        upper: Upper band values
        lower: Lower band values
        middle: Middle band values

    Returns:
        List of band width percentages
    """
    width = []
    for i in range(len(middle)):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or middle[i] == 0:
            width.append(float("nan"))
        else:
            width.append((upper[i] - lower[i]) / middle[i] * 100)

    return width


def calculate_true_range(
    high: float,
    low: float,
    prev_close: float,
) -> float:
    """
    Calculate True Range for a single bar.

    TR = max(High - Low, |High - PrevClose|, |Low - PrevClose|)

    Args:
        high: Current bar high
        low: Current bar low
        prev_close: Previous bar close

    Returns:
        True Range value
    """
    return max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    )


def calculate_atr_series(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float]:
    """
    Calculate Average True Range (ATR) series.

    ATR = Wilder's smoothed average of True Range over N periods.

    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: ATR period (default 14)

    Returns:
        List of ATR values (same length as inputs, with NaN for insufficient data)
    """
    n = len(closes)
    if n < period + 1:
        return [float("nan")] * n

    # Calculate True Range series
    tr_values = [float("nan")]  # First bar has no previous close
    for i in range(1, n):
        tr = calculate_true_range(highs[i], lows[i], closes[i - 1])
        tr_values.append(tr)

    # Calculate ATR using Wilder's smoothing
    atr_values = [float("nan")] * period

    # First ATR is SMA of first 'period' TR values (skip first NaN)
    first_tr_window = tr_values[1:period + 1]
    if len(first_tr_window) == period:
        atr = sum(first_tr_window) / period
        atr_values.append(atr)

        # Subsequent ATR values using Wilder's smoothing
        for i in range(period + 1, n):
            atr = (atr * (period - 1) + tr_values[i]) / period
            atr_values.append(atr)

    return atr_values


def calculate_adx_series(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> tuple[list[float], list[float], list[float]]:
    """
    Calculate Average Directional Index (ADX) series.

    ADX measures trend strength (not direction).
    - ADX < 25: Weak or no trend (range-bound)
    - ADX 25-50: Strong trend
    - ADX > 50: Very strong trend

    Also returns +DI and -DI for directional analysis:
    - +DI > -DI: Bullish trend
    - -DI > +DI: Bearish trend

    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of close prices
        period: ADX period (default 14)

    Returns:
        Tuple of (adx_values, plus_di_values, minus_di_values)
        Each same length as inputs, with NaN for insufficient data
    """
    n = len(closes)
    if n < period * 2:
        return ([float("nan")] * n, [float("nan")] * n, [float("nan")] * n)

    # Calculate +DM and -DM
    plus_dm = [float("nan")]
    minus_dm = [float("nan")]

    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        # +DM: positive up move that's greater than down move
        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)

        # -DM: positive down move that's greater than up move
        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)

    # Calculate ATR (needed for DI calculation)
    atr_values = calculate_atr_series(highs, lows, closes, period)

    # Smooth +DM and -DM using Wilder's method
    smoothed_plus_dm = [float("nan")] * period
    smoothed_minus_dm = [float("nan")] * period

    # First smoothed value is sum of first 'period' values
    first_plus_dm = sum(plus_dm[1:period + 1])
    first_minus_dm = sum(minus_dm[1:period + 1])
    smoothed_plus_dm.append(first_plus_dm)
    smoothed_minus_dm.append(first_minus_dm)

    # Subsequent values using Wilder's smoothing
    for i in range(period + 1, n):
        smoothed_plus_dm.append(
            smoothed_plus_dm[-1] - (smoothed_plus_dm[-1] / period) + plus_dm[i]
        )
        smoothed_minus_dm.append(
            smoothed_minus_dm[-1] - (smoothed_minus_dm[-1] / period) + minus_dm[i]
        )

    # Calculate +DI and -DI
    plus_di = [float("nan")] * n
    minus_di = [float("nan")] * n

    for i in range(period, n):
        if not np.isnan(atr_values[i]) and atr_values[i] > 0:
            plus_di[i] = 100 * smoothed_plus_dm[i] / (atr_values[i] * period)
            minus_di[i] = 100 * smoothed_minus_dm[i] / (atr_values[i] * period)

    # Calculate DX
    dx_values = [float("nan")] * n
    for i in range(period, n):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx_values[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

    # Calculate ADX as smoothed DX
    adx_values = [float("nan")] * n

    # First ADX is average of first 'period' DX values after DX starts
    dx_start = period
    first_valid_dx = []
    for i in range(dx_start, min(dx_start + period, n)):
        if not np.isnan(dx_values[i]):
            first_valid_dx.append(dx_values[i])

    if len(first_valid_dx) == period:
        adx = sum(first_valid_dx) / period
        adx_idx = dx_start + period - 1
        if adx_idx < n:
            adx_values[adx_idx] = adx

            # Subsequent ADX values using Wilder's smoothing
            for i in range(adx_idx + 1, n):
                if not np.isnan(dx_values[i]):
                    adx = (adx * (period - 1) + dx_values[i]) / period
                    adx_values[i] = adx

    return (adx_values, plus_di, minus_di)
