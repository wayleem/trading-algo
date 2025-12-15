import pytest
from datetime import datetime
from app.services.signal_generator import SignalGenerator
from app.models.schemas import SignalType


class TestSignalGenerator:
    def test_no_signal_on_first_bar(self):
        """Test no signal generated on first bar (no previous data)."""
        generator = SignalGenerator()

        signal = generator.evaluate(
            current_rsi=25.0,
            current_sma=30.0,
            close_price=450.0,
            timestamp=datetime.now(),
        )

        assert signal.signal_type == SignalType.NO_SIGNAL

    def test_buy_call_signal(self):
        """Test BUY_CALL signal on oversold bullish crossover."""
        generator = SignalGenerator(rsi_oversold=30.0)

        # First bar - establishes previous values
        generator.evaluate(
            current_rsi=25.0,
            current_sma=28.0,
            close_price=450.0,
            timestamp=datetime.now(),
        )

        # Second bar - bullish crossover while oversold
        signal = generator.evaluate(
            current_rsi=29.0,  # Still below 30 (oversold)
            current_sma=27.0,  # RSI crossed above SMA
            close_price=451.0,
            timestamp=datetime.now(),
        )

        assert signal.signal_type == SignalType.BUY_CALL

    def test_buy_put_signal(self):
        """Test BUY_PUT signal on overbought bearish crossover."""
        generator = SignalGenerator(rsi_overbought=70.0)

        # First bar - establishes previous values
        generator.evaluate(
            current_rsi=75.0,
            current_sma=72.0,
            close_price=450.0,
            timestamp=datetime.now(),
        )

        # Second bar - bearish crossover while overbought
        signal = generator.evaluate(
            current_rsi=71.0,  # Still above 70 (overbought)
            current_sma=73.0,  # RSI crossed below SMA
            close_price=449.0,
            timestamp=datetime.now(),
        )

        assert signal.signal_type == SignalType.BUY_PUT

    def test_no_signal_not_oversold(self):
        """Test no signal when RSI not in extreme zone."""
        generator = SignalGenerator()

        # First bar
        generator.evaluate(
            current_rsi=45.0,
            current_sma=48.0,
            close_price=450.0,
            timestamp=datetime.now(),
        )

        # Second bar - crossover but not oversold/overbought
        signal = generator.evaluate(
            current_rsi=50.0,
            current_sma=47.0,
            close_price=451.0,
            timestamp=datetime.now(),
        )

        assert signal.signal_type == SignalType.NO_SIGNAL

    def test_reset(self):
        """Test reset clears previous values."""
        generator = SignalGenerator()

        # Establish previous values
        generator.evaluate(
            current_rsi=50.0,
            current_sma=50.0,
            close_price=450.0,
            timestamp=datetime.now(),
        )

        generator.reset()

        # After reset, should have no previous values
        assert generator._prev_rsi is None
        assert generator._prev_sma is None

    def test_signal_includes_metadata(self):
        """Test signal includes all metadata."""
        generator = SignalGenerator()

        timestamp = datetime.now()
        signal = generator.evaluate(
            current_rsi=50.0,
            current_sma=50.0,
            close_price=450.0,
            timestamp=timestamp,
        )

        assert signal.rsi == 50.0
        assert signal.rsi_sma == 50.0
        assert signal.close_price == 450.0
        assert signal.timestamp == timestamp
        assert signal.reason is not None
