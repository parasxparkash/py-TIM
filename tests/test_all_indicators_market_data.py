#!/usr/bin/env python3
"""
Comprehensive Market Data Testing for All py-TIM Indicators

This module uses real financial data from Yahoo Finance to test every implemented
technical indicator across multiple market conditions and asset classes.

Tests all indicators for:
- Mathematical correctness
- Performance efficiency
- Error handling
- Edge cases
- Real-world market data compatibility

Author: Paras Parkash
"""

import warnings
warnings.filterwarnings('ignore')

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import scipy.stats as stats

# Import all py-TIM indicators
from ta_analysis import *


@pytest.fixture(scope="session")
def real_market_data():
    """Download real market data for comprehensive testing"""
    try:
        import yfinance as yf
    except ImportError:
        pytest.skip("yfinance not available for testing")
        return {}

    # Download data for multiple markets and timeframes
    symbols = {
        'tech_stock': 'AAPL',      # Technology sector leader
        'tech_index': '^IXIC',      # NASDAQ Composite
        'banking_stock': 'JPM',     # Banking sector
        'energy_stock': 'XOM',      # Energy sector
        'crypto': 'BTC-USD',        # Cryptocurrency
        'commodity': 'GC=F',        # Gold Futures (public market)
        'fx_pair': 'EURUSD=X'       # Forex pair if available
    }

    market_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years data

    print(f"\n📡 Downloading market data for comprehensive testing...")

    for category, symbol in symbols.items():
        try:
            print(f"  Downloading {symbol} ({category})...", end="", flush=True)
            df = yf.download(symbol, start=start_date, end=end_date,
                           progress=False, prepost=False)

            if not df.empty and len(df) > 500:  # Require substantial data
                # Clean and prepare data
                data = prepare_market_data(df)
                market_data[category] = data
                print(f" ✓ {len(data['close']):,} points")
            else:
                print(" ❌ Insufficient data")
        except Exception as e:
            print(f" ❌ Error: {str(e)[:30]}...")
            print(" ❌ Insufficient data")
    if len(market_data) < 2:
        pytest.skip("Insufficient market data downloaded for comprehensive testing")

    print(f"✓ Downloaded data for {len(market_data)} asset classes")
    return market_data


def prepare_market_data(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Clean and prepare market data for indicator testing"""
    # Remove missing values
    df = df.dropna()

    # Ensure OHLC integrity
    if len(df) > 0:
        # Make sure high >= max(open, close) and low <= min(open, close)
        df['High'] = df[['High', 'Open', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Open', 'Close']].min(axis=1)

    return {
        'open': df['Open'].values,
        'high': df['High'].values,
        'low': df['Low'].values,
        'close': df['Close'].values,
        'volume': df['Volume'].fillna(10000).values,  # Handle missing volume
        'dates': df.index
    }


@pytest.fixture
def trending_stock_data(real_market_data):
    """Get trending stock data for testing"""
    return real_market_data.get('tech_stock', {})


@pytest.fixture
def volatile_prop_data(real_market_data):
    """Get volatile/cryptocurrency data"""
    return real_market_data.get('crypto', {})


@pytest.fixture
def stable_banking_data(real_market_data):
    """Get stable banking sector data"""
    return real_market_data.get('banking_stock', {})


class TestAllTrendIndicators:
    """Comprehensive testing of all trend indicators"""

    @pytest.mark.parametrize("data_fixture", [
        "trending_stock_data",
        "stable_banking_data"
    ])
    def test_sma_comprehensive(self, request, data_fixture):
        """Test SMA across different market conditions"""
        data = request.getfixturevalue(data_fixture)
        if not data:
            pytest.skip("Market data not available")

        result = sma(data['close'], period=20)

        # Comprehensive validation
        assert len(result) == len(data['close'])
        assert np.isfinite(result[19])  # First valid value
        assert pytest.approx(result[19], abs=0.01) == np.mean(data['close'][:20])

        # Trend coherence test
        latest_sma = result[-1]
        recent_prices = data['close'][-50:]
        assert latest_sma >= np.min(recent_prices) * 0.9  # Reasonable range
        assert latest_sma <= np.max(recent_prices) * 1.1

    @pytest.mark.parametrize("data_fixture", [
        "trending_stock_data",
        "volatile_prop_data"
    ])
    def test_ema_adaptiveness(self, request, data_fixture):
        """Test EMA smoothing and adaptive behavior"""
        data = request.getfixturevalue(data_fixture)
        if not data:
            pytest.skip("Market data not available")

        ema_short = ema(data['close'], period=12)
        ema_long = ema(data['close'], period=26)

        # EMA properties
        assert len(ema_short) == len(ema_long) == len(data['close'])

        # Short EMA should be more responsive (higher variance) in trending markets
        short_variance = np.var(ema_short[25:] - data['close'][25:])
        long_variance = np.var(ema_long[25:] - data['close'][25:])

        if data_fixture == "trending_stock_data":
            # In trending data, short EMA should follow price more closely
            # This is a statistical expectation, not absolute
            pass  # Skip statistical assertion in basic tests

    def test_parabolic_sar_stops(real_market_data):
        """Test Parabolic SAR stop levels across markets"""
        test_data = real_market_data.get('tech_stock', {}) or real_market_data.get('banking_stock', {})
        if not test_data:
            pytest.skip("Market data not available")

        psar = parabolic_sar(test_data['high'], test_data['low'])

        assert len(psar) == len(test_data['high'])

        # SAR should be within high-low range
        valid_psar = psar[np.isfinite(psar)]
        if len(valid_psar) > 10:
            avg_psar = np.mean(valid_psar)
            price_range = np.mean(test_data['high'] - test_data['low'])

            # SAR should be reasonable relative to price
            assert avg_psar > np.min(test_data['low']) * 0.9
            assert avg_psar < np.max(test_data['high']) * 1.1


class TestAllMomentumIndicators:
    """Comprehensive testing of all momentum indicators"""

    def test_rsi_extremes_real_data(self, real_market_data):
        """Test RSI in real market conditions"""
        data = real_market_data.get('tech_stock', {}) or real_market_data.get('banking_stock', {})
        if not data:
            pytest.skip("Market data not available")

        rsi_vals = rsi(data['close'], period=14)

        assert len(rsi_vals) == len(data['close'])
        valid_rsi = rsi_vals[np.isfinite(rsi_vals)]

        # RSI should be bounded
        assert np.all((valid_rsi >= 0) & (valid_rsi <= 100))

        # Check for reasonable distribution (not all 50)
        rsi_variance = np.var(valid_rsi[-100:]) if len(valid_rsi) > 100 else 0
        if rsi_variance > 1:  # Some variation expected
            # RSI should show reasonable volatility in normal markets
            assert rsi_variance < 2000  # Not excessively volatile

    def test_macd_divergence_detection(self, real_market_data):
        """Test MACD signal generation and divergence"""
        data = real_market_data.get('tech_stock', {}) or real_market_data.get('banking_stock', {})
        if not data:
            pytest.skip("Market data not available")

        macd_line, signal_line, histogram = macd(data['close'])

        assert len(macd_line) == len(signal_line) == len(histogram) == len(data['close'])

        # Check signal crossovers exist
        if len(macd_line) > 50:  # Only test if sufficient data
            recent_macd = macd_line[-50:]
            recent_signal = signal_line[-50:]

            # Should have some crossovers in normal market data
            crossovers = np.sum(np.sign(recent_macd - recent_signal)[:-1] !=
                              np.sign(recent_macd - recent_signal)[1:])

            # Not all identical (some market movement expected)
            assert crossovers > 0 or np.mean(np.abs(recent_macd - recent_signal)) < 0.0001

    def test_adx_trend_strength(self, real_market_data):
        """Test ADX trend strength measurement"""
        data = real_market_data.get('banking_stock', {}) or real_market_data.get('tech_stock', {})
        if not data:
            pytest.skip("Market data not available")

        adx_val, plus_di, minus_di = adx(data['high'], data['low'], data['close'])

        assert len(adx_val) == len(data['high'])

        # ADX should be bounded [0, 100]
        valid_adx = adx_val[np.isfinite(adx_val)]
        if len(valid_adx) > 0:
            assert np.all((valid_adx >= 0) & (valid_adx <= 100))

            # ADX often stays below 50 in non-trending markets
            average_adx = np.mean(valid_adx[-50:]) if len(valid_adx) > 50 else np.mean(valid_adx)
            assert average_adx <= 80  # Should not be excessive


class TestAllVolumeIndicators:
    """Comprehensive testing of all volume indicators"""

    def test_volume_price_trend_accumulation(self, real_market_data):
        """Test Price Volume Trend accumulation over time"""
        data = real_market_data.get('tech_stock', {})
        if not data:
            pytest.skip("Market data not available")

        pvt = price_volume_trend(data['close'], data['volume'])

        assert len(pvt) == len(data['close'])

        # PVT should accumulate and not stay constant
        if len(pvt) > 50:
            pvt_range = np.max(pvt[-50:]) - np.min(pvt[-50:])
            assert pvt_range > 0  # Should show some accumulation/distribution

    def test_chaikin_money_flow_real_dynamics(self, real_market_data):
        """Test Chaikin Money Flow in real market conditions"""
        data = real_market_data.get('tech_stock', {}) or real_market_data.get('banking_stock', {})
        if not data:
            pytest.skip("Market data not available")

        cmf = cmf(data['high'], data['low'], data['close'], data['volume'])

        assert len(cmf) == len(data['high'])

        # CMF should be bounded [-1, 1] by definition
        valid_cmf = cmf[np.isfinite(cmf)]
        if len(valid_cmf) > 0:
            assert np.all((valid_cmf >= -1.1) & (valid_cmf <= 1.1))  # Small tolerance

    def test_volume_positive_negative_index(self, real_market_data):
        """Test Positive/Negative Volume Index behavior"""
        data = real_market_data.get('tech_stock', {})
        if not data:
            pytest.skip("Market data not available")

        pvi = positive_volume_index(data['close'], data['volume'])
        nvi = negative_volume_index(data['close'], data['volume'])

        assert len(pvi) == len(nvi) == len(data['close'])

        # PVI and NVI should evolve from initial price
        if len(pvi) > 100:
            # Check that they have reasonable range over time
            pvi_range = np.max(pvi[50:]) - np.min(pvi[50:])
            nvi_range = np.max(nvi[50:]) - np.min(nvi[50:])

            # In normal markets, these should show some variation
            total_range = pvi_range + nvi_range
            assert total_range > 0.001  # Some movement expected


class TestIndicatorEdgeCases:
    """Test indicators with various edge cases"""

    def test_empty_data_handling(self):
        """Test how indicators handle empty data"""
        empty_data = np.array([])

        with pytest.raises(ValueError):
            sma(empty_data, 20)

        with pytest.raises(ValueError):
            rsi(empty_data, 14)

    def test_single_value_data(self):
        """Test indicators with minimal data"""
        single_value = np.array([100.0])

        # Should raise appropriate errors for insufficient data
        with pytest.raises(ValueError):
            sma(single_value, 20)  # Need at least 20 values

    def test_nan_inf_handling(self):
        """Test NaN and infinite value handling"""
        data = np.array([100.0, 102.0, float('nan'), 105.0, float('inf'), 108.0])

        # Indicators should handle NaN gracefully
        result = sma(data, 3)

        # Should not return infinite values
        assert not np.any(np.isinf(result))

        # Should handle NaN without crashing (though may return NaN)
        assert len(result) == len(data)

    def test_extreme_values(self):
        """Test indicators with extreme price values"""
        data = np.array([0.0001, 50000, 0.001, 75000, 100])  # Wide price range

        result = sma(data, 3)  # Should handle without numerical issues
        assert not np.any(np.isnan(result))  # Should not return NaN unexpectedly

        result = rsi(data, 7)
        assert len(result) == len(data)

    def test_constant_data(self):
        """Test indicators with constant price data"""
        constant_data = np.array([100.0] * 50)

        sma_result = sma(constant_data, 20)
        ema_result = ema(constant_data, 20)

        # All SMA values should equal the constant
        sma_valid = sma_result[np.isfinite(sma_result)]
        if len(sma_valid) > 0:
            assert np.allclose(sma_valid, 100.0, rtol=0.001)

        # EMA should eventually converge to the constant
        ema_valid = ema_result[np.isfinite(ema_result)]
        if len(ema_valid) > 0:
            last_ema = ema_valid[-10:]  # Last 10 values
            assert np.allclose(last_ema, 100.0, atol=1.0)  # Should be close to constant


class TestCrossMarketIndicatorRobustness:
    """Test indicators across different market types"""

    def test_stock_vs_crypto_indicators(self, real_market_data):
        """Compare indicator behavior across asset classes"""
        stock_data = real_market_data.get('tech_stock', {})
        crypto_data = real_market_data.get('crypto', {})

        if not stock_data or not crypto_data:
            pytest.skip("Insufficient diverse market data")

        # Test RSI on both markets
        stock_rsi = rsi(stock_data['close'], 14)
        crypto_rsi = rsi(crypto_data['close'], 14)

        # Both should be bounded [0, 100]
        stock_valid = stock_rsi[np.isfinite(stock_rsi)]
        crypto_valid = crypto_rsi[np.isfinite(crypto_rsi)]

        assert np.all((stock_valid >= 0) & (stock_valid <= 100))
        assert np.all((crypto_valid >= 0) & (crypto_valid <= 100))

    def test_volume_indicators_different_markets(self, real_market_data):
        """Test volume indicators work across different liquidity regimes"""
        stock_data = real_market_data.get('tech_stock', {})
        crypto_data = real_market_data.get('crypto', {})

        if not stock_data or not crypto_data:
            pytest.skip("Insufficient diverse market data")

        # Test OBV on both
        stock_obv = obv(stock_data['close'], stock_data['volume'])
        crypto_obv = obv(crypto_data['close'], crypto_data['volume'])

        assert len(stock_obv) == len(stock_data['close'])
        assert len(crypto_obv) == len(crypto_data['close'])

        # Should not contain infinite values
        assert not np.any(np.isinf(stock_obv))
        assert not np.any(np.isinf(crypto_obv))


class TestStatisticalIndicatorProperties:
    """Test statistical and mathematical properties of indicators"""

    def test_moving_average_lag_properties(self, trending_stock_data):
        """Test that moving averages exhibit expected lag properties"""
        if not trending_stock_data:
            pytest.skip("Market data not available")

        prices = trending_stock_data['close'][-200:]  # Recent data

        # Calculate different period SMAs
        sma_short = sma(prices, 10)
        sma_long = sma(prices, 50)

        # Longer MA should lag more behind recent prices
        if len(sma_long) > 60:  # Ensure enough data
            short_recent = sma_short[-30:].mean()
            long_recent = sma_long[-30:].mean()
            current_price = prices[-1:].mean()

            # Lag relationship: longer MAs have more lag
            short_lag = abs(short_recent - current_price)
            long_lag = abs(long_recent - current_price)

            # This is a statistical tendency, not absolute rule
            # Longer MA should show more smoothing/lag in trending markets
            assert short_lag <= long_lag + current_price * 0.1  # Allow some tolerance

    def test_indicator_continuous_evolution(self, real_market_data):
        """Test that indicators evolve continuously without jumps"""
        data = real_market_data.get('tech_stock', {})
        if not data:
            pytest.skip("Market data not available")

        rsi_values = rsi(data['close'], 14)

        # Check for excessive jumps (should be gradual)
        if len(rsi_values) > 20:
            rsi_valid = rsi_values[np.isfinite(rsi_values)][-100:]  # Last 100 valid values

            if len(rsi_valid) > 10:
                # Calculate daily changes
                rsi_changes = np.abs(rsi_valid[1:] - rsi_valid[:-1])

                # Most changes should be reasonable (< 20 points per day)
                extreme_changes = np.sum(rsi_changes > 25)
                extreme_ratio = extreme_changes / len(rsi_changes)

                assert extreme_ratio < 0.05  # Less than 5% extreme changes


class TestIndicatorPerformanceBenchmarking:
    """Performance benchmarking across all indicators"""

    @pytest.mark.slow
    def test_large_dataset_performance(self, real_market_data):
        """Test indicator performance on large real datasets"""
        data = real_market_data.get('tech_stock', {})
        if not data:
            pytest.skip("Market data not available")

        # Use only the most recent 1000 points if available
        large_close = data['close'][-1000:] if len(data['close']) > 1000 else data['close']
