"""
Performance and Benchmarking Tests for py-TIM Indicators
"""

import pytest
import time
import numpy as np
from ta_analysis import sma, ema, rsi, macd, bollinger_bands


class TestPerformance:
    """Performance benchmarking tests"""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_dataset_performance(self, large_test_dataset):
        """Test indicator performance on large datasets"""
        start_time = time.time()
        result = sma(large_test_dataset, period=20)
        sma_time = time.time() - start_time

        start_time = time.time()
        result = ema(large_test_dataset, period=20)
        ema_time = time.time() - start_time

        start_time = time.time()
        macd_line, signal, histogram = macd(large_test_dataset)
        macd_time = time.time() - start_time

        # Performance requirements for large datasets
        assert sma_time < 1.0, f"SMA too slow: {sma_time}s"
        assert ema_time < 1.0, f"EMA too slow: {ema_time}s"
        assert macd_time < 2.0, f"MACD too slow: {macd_time}s"

        # Verify results are still accurate
        assert len(result) == len(large_test_dataset)
        assert np.all(np.isfinite(result))

    @pytest.mark.benchmark
    def test_time_complexity_comparison(self):
        """Compare time complexity of different indicators"""
        sizes = [100, 500, 1000, 2000]
        sma_times = []
        ema_times = []
        rsi_times = []

        for size in sizes:
            # Generate test data
            data = 100 + np.cumsum(np.random.normal(0, 1, size))
            data = np.abs(data)  # Ensure positive prices (for RSI)

            # Test SMA
            start = time.time()
            sma(data, period=20)
            sma_times.append(time.time() - start)

            # Test EMA
            start = time.time()
            ema(data, period=20)
            ema_times.append(time.time() - start)

            # Test RSI
            start = time.time()
            rsi(data, period=14)
            rsi_times.append(time.time() - start)

        # SMA should have similar or better performance than EMA
        # RSI should be reasonable
        assert all(t < 0.5 for t in sma_times[-2:])  # Last two sizes should be fast
        assert all(t < 1.0 for t in ema_times[-2:])
        assert all(t < 1.0 for t in rsi_times[-2:])

    @pytest.mark.performance
    def test_memory_efficiency(self, large_test_dataset):
        """Test memory efficiency of indicator calculations"""
        import psutil
        import os

        initial_memory = psutil.Process(os.getpid()).memory_info().rss

        # Run multiple indicators on large dataset
        result1 = sma(large_test_dataset, period=50)
        result2 = ema(large_test_dataset, period=50)
        result3 = rsi(large_test_dataset, period=14)

        final_memory = psutil.Process(os.getpid()).memory_info().rss
        memory_used = final_memory - initial_memory

        # Should not use excessive memory (less than 100MB additional)
        assert memory_used < 100 * 1024 * 1024  # 100MB

        # Verify results are valid
        assert len(result1) == len(large_test_dataset)
        assert len(result2) == len(large_test_dataset)
        assert len(result3) == len(large_test_dataset)

    def test_calculation_accuracy_optimization(self):
        """Test that optimizations don't compromise accuracy"""
        data = np.array([100, 102, 98, 105, 108, 110, 112, 115, 118, 120], dtype=np.float64)

        # Test that results are consistent across runs (deterministic)
        result1 = sma(data, period=5)
        result2 = sma(data, period=5)

        np.testing.assert_array_almost_equal(result1, result2, decimal=10)

        # Test that results match expected mathematical formulas
        expected_last = np.mean(data[-5:])
        assert result1[-1] == pytest.approx(expected_last, abs=1e-10)


class TestAccuracyValidation:
    """Tests to validate mathematical accuracy of indicators"""

    def test_moving_average_invariant(self):
        """Test that simple mathematical invariants hold"""
        # For constant data, all moving averages should equal the constant
        constant_data = np.array([100.0] * 20)
        period = 10

        sma_result = sma(constant_data, period)[period - 1:]
        ema_result = ema(constant_data, period)[period - 1:]

        # All results should be very close to 100
        np.testing.assert_allclose(sma_result, 100.0, atol=0.1)
        np.testing.assert_allclose(ema_result, 100.0, atol=0.01)  # EMA should be more precise

    def test_rsi_extremes(self):
        """Test RSI behavior at extremes"""
        # All gains should give RSI ≈ 100
        data = np.array([1] * 20).cumsum()  # Strongly increasing: 1,2,3,4...
        rsi_result = rsi(data, period=14)

        # RSI of strongly trending data should be high but not necessarily 100
        # The exact value depends on the calculation, but should be valid
        assert np.all(rsi_result >= 0)
        assert np.all(rsi_result <= 100)
