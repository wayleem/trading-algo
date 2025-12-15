import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_closes():
    """Sample closing prices for indicator tests."""
    return [
        100.0, 101.0, 102.0, 101.5, 100.5,
        99.5, 98.5, 97.0, 96.0, 95.0,
        94.5, 95.5, 96.5, 97.5, 98.5,
        99.0, 100.0, 101.0, 102.0, 103.0,
    ]


@pytest.fixture
def oversold_closes():
    """Prices that create oversold RSI condition."""
    # Start high and trend down sharply
    return [
        100.0, 99.5, 99.0, 98.5, 98.0,
        97.0, 96.0, 95.0, 94.0, 93.0,
        92.0, 91.0, 90.0, 89.0, 88.0,
        87.5, 87.0, 86.5, 86.0, 85.5,
    ]


@pytest.fixture
def overbought_closes():
    """Prices that create overbought RSI condition."""
    # Start low and trend up sharply
    return [
        100.0, 100.5, 101.0, 101.5, 102.0,
        103.0, 104.0, 105.0, 106.0, 107.0,
        108.0, 109.0, 110.0, 111.0, 112.0,
        112.5, 113.0, 113.5, 114.0, 114.5,
    ]
