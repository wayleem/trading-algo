import pytest
from app.services.indicators import IndicatorService, calculate_rsi_series, calculate_sma_series


class TestIndicatorService:
    def test_calculate_rsi_neutral(self, sample_closes):
        """Test RSI calculation returns value between 0-100."""
        service = IndicatorService()
        rsi = service.calculate_rsi(sample_closes)

        assert 0 <= rsi <= 100

    def test_calculate_rsi_oversold(self, oversold_closes):
        """Test RSI is low for downtrending prices."""
        service = IndicatorService()
        rsi = service.calculate_rsi(oversold_closes)

        assert rsi < 40  # Should be relatively low

    def test_calculate_rsi_overbought(self, overbought_closes):
        """Test RSI is high for uptrending prices."""
        service = IndicatorService()
        rsi = service.calculate_rsi(overbought_closes)

        assert rsi > 60  # Should be relatively high

    def test_calculate_rsi_insufficient_data(self):
        """Test RSI returns neutral with insufficient data."""
        service = IndicatorService()
        rsi = service.calculate_rsi([100.0, 101.0, 102.0])

        assert rsi == 50.0  # Default neutral

    def test_calculate_sma(self):
        """Test SMA calculation."""
        service = IndicatorService(sma_period=5)
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        sma = service.calculate_sma(values)

        assert sma == 30.0  # Average of [10, 20, 30, 40, 50]

    def test_detect_bullish_crossover(self):
        """Test bullish crossover detection."""
        service = IndicatorService()

        result = service.detect_crossover(
            current_rsi=32.0,
            previous_rsi=28.0,
            current_sma=30.0,
            previous_sma=30.0,
        )

        assert result == "bullish"

    def test_detect_bearish_crossover(self):
        """Test bearish crossover detection."""
        service = IndicatorService()

        result = service.detect_crossover(
            current_rsi=68.0,
            previous_rsi=72.0,
            current_sma=70.0,
            previous_sma=70.0,
        )

        assert result == "bearish"

    def test_detect_no_crossover(self):
        """Test no crossover when RSI stays on same side."""
        service = IndicatorService()

        result = service.detect_crossover(
            current_rsi=55.0,
            previous_rsi=52.0,
            current_sma=50.0,
            previous_sma=50.0,
        )

        assert result is None


class TestRSISeries:
    def test_calculate_rsi_series_length(self, sample_closes):
        """Test RSI series has same length as input."""
        rsi_series = calculate_rsi_series(sample_closes, period=14)

        assert len(rsi_series) == len(sample_closes)

    def test_calculate_sma_series_length(self):
        """Test SMA series has same length as input."""
        values = [50.0] * 20
        sma_series = calculate_sma_series(values, period=14)

        assert len(sma_series) == len(values)
