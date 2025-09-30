"""
Shared fixtures and configuration for py-TIM library tests
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_close_data():
    """Sample closing price data for testing"""
    return np.array([100, 102, 98, 105, 108, 110, 112, 115, 118, 120])


@pytest.fixture
def sample_volume_data():
    """Sample volume data for testing"""
    return np.array([1000, 1200, 800, 1500, 1100, 1300, 1400, 1600])


@pytest.fixture
def sample_ohlc_data():
    """Sample OHLC (Open, High, Low, Close) data"""
    return {
        'open': np.array([99, 101, 97, 104, 107, 109, 111, 114, 117, 119]),
        'high': np.array([101, 103, 99, 106, 109, 111, 113, 116, 119, 121]),
        'low': np.array([98, 100, 96, 103, 106, 108, 110, 113, 116, 118]),
        'close': np.array([100, 102, 98, 105, 108, 110, 112, 115, 118, 120])
    }


@pytest.fixture
def trending_up_data():
    """Data that trends strongly upward"""
    base_price = 100
    noise = np.random.normal(0, 0.01, 50)  # Small noise
    trend = np.linspace(0, 10, 50)  # Strong upward trend
    prices = base_price + trend + noise
    return prices


@pytest.fixture
def oscillating_data():
    """Data with strong oscillations around a mean"""
    mean_price = 100
    amplitude = 5
    periods = 10
    t = np.linspace(0, periods, 50)
    prices = mean_price + amplitude * np.sin(2 * np.pi * t / (50/periods))
    prices += np.random.normal(0, 1, 50)  # Add some noise
    return prices


@pytest.fixture
def volatile_data():
    """High volatility data"""
    base_price = 100
    volatility = 3
    changes = np.random.normal(0, volatility, 50)
    prices = base_price + np.cumsum(changes)
    return prices


@pytest.fixture
def flat_market_data():
    """Sideways/flat market data"""
    base_price = 100
    noise = np.random.normal(0, 0.5, 50)  # Low volatility
    prices = base_price + noise
    return prices


@pytest.fixture
def large_test_dataset():
    """Large dataset for performance testing (1000 data points)"""
    np.random.seed(42)  # For reproducible results
    base_price = 100
    changes = np.random.normal(0.001, 0.015, 1000)  # Small daily changes
    prices = base_price * np.exp(np.cumsum(changes))
    return prices


@pytest.fixture
def test_dataframe(sample_ohlc_data):
    """Pandas DataFrame version of OHLC data"""
    df = pd.DataFrame({
        'Open': sample_ohlc_data['open'],
        'High': sample_ohlc_data['high'],
        'Low': sample_ohlc_data['low'],
        'Close': sample_ohlc_data['close']
    })
    return df


@pytest.fixture
def test_series(sample_close_data):
    """Pandas Series version of close data"""
    return pd.Series(sample_close_data, dtype=float)
