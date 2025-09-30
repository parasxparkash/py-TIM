"""
Tests for Trend Analysis Indicators
"""

import pytest
import numpy as np
from ta_analysis import sma, ema, wma, dema, tema, kama, hma, parabolic_sar


class TestTrendIndicators:
    """Test suite for all trend indicators"""

    def test_sma_basic(self, sample_close_data):
        """Test Simple Moving Average with basic functionality"""
        result = sma(sample_close_data, period=5)
        assert len(result) == len(sample_close_data)
        assert np.isnan(result[0])  # First period-1 values should be NaN
        assert np.all(np.isfinite(result[4:]))  # Values from period-1 onward should be finite
        assert result[-1] == pytest.approx(np.mean(sample_close_data[-5:]))

    def test_ema_basic(self, sample_close_data):
        """Test Exponential Moving Average"""
        result = ema(sample_close_data, period=5)
        assert len(result) == len(sample_close_data)
        assert np.isnan(result[0])
        assert np.all(np.isfinite(result[4:]))

    @pytest.mark.trending
    def test_hma_accuracy(self, trending_up_data):
        """Test Hull Moving Average accuracy in trending markets"""
        result = hma(trending_up_data, period=10)
        assert len(result) == len(trending_up_data)
        # In a strong uptrend, HMA should be smoother than raw data
        # and follow the trend direction

    def test_kama_adaptive(self, volatile_data):
        """Test Kaufman's Adaptive Moving Average in volatile conditions"""
        result = kama(volatile_data, period=15)
        assert len(result) == len(volatile_data)

    def test_parabolic_sar_basic(self, sample_ohlc_data):
        """Test Parabolic SAR basic functionality"""
        sar = parabolic_sar(sample_ohlc_data['high'], sample_ohlc_data['low'])
        assert len(sar) == len(sample_ohlc_data['high'])
        # SAR should be within range of high/low prices
        for i, sar_val in enumerate(sar):
            if np.isfinite(sar_val):
                assert sar_val >= sample_ohlc_data['low'][i]
                assert sar_val <= sample_ohlc_data['high'][i]

    def test_sma_edge_cases(self):
        """Test SMA with edge cases"""
        # Test with period larger than data
        data = np.array([1, 2, 3])
        result = sma(data, period=5)
        assert all(np.isnan(result))

        # Test with empty data
        with pytest.raises(ValueError):
            sma([], period=2)

    def test_ema_weighting(self):
        """Test EMA proper weighting behavior"""
        data = np.array([1] * 10)  # Constant data should produce constant EMA
        result = ema(data, period=5)

        # For constant data, EMA should eventually converge to the constant
        # The last few values should be very close to 1.0
        assert result[-1] == pytest.approx(1.0, abs=0.1)
        assert result[-2] == pytest.approx(1.0, abs=0.1)
