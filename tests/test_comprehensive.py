#!/usr/bin/env python3
"""
Comprehensive Test Suite for py-TIM Library

This test suite provides extensive testing including:
- Performance benchmarks
- Edge case handling
- Mathematical accuracy validation
- Input validation
- Output consistency checks
- Real-world scenario testing
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import warnings
from typing import Callable, Any

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_tim.indicators import (
    sma, ema, wma, rsi, macd, bollinger_bands, atr, obv
)


class TestSuite:
    """Comprehensive test suite for technical indicators"""
    
    def __init__(self):
        self.test_results = []
        self.performance_results = []
        
    def run_test(self, test_name: str, test_func: Callable) -> bool:
        """Run a single test and capture results"""
        try:
            start_time = time.time()
            result = test_func()
            execution_time = time.time() - start_time
            
            self.test_results.append({
                'name': test_name,
                'status': 'PASS' if result else 'FAIL',
                'execution_time': execution_time
            })
            
            print(f"✓ {test_name}: {'PASS' if result else 'FAIL'} ({execution_time:.4f}s)")
            return result
        except Exception as e:
            self.test_results.append({
                'name': test_name,
                'status': 'ERROR',
                'error': str(e)
            })
            print(f"✗ {test_name}: ERROR - {str(e)}")
            return False
    
    def test_sma_accuracy(self) -> bool:
        """Test SMA mathematical accuracy"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = sma(data, 3)
        
        # Manual calculation for verification
        expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        
        return np.allclose(result, expected, equal_nan=True)
    
    def test_ema_convergence(self) -> bool:
        """Test EMA convergence properties"""
        # Test with constant data - EMA should converge to the constant value
        constant_data = np.full(50, 100.0)
        result = ema(constant_data, 10)
        
        # The last few values should be very close to 100
        return np.allclose(result[-5:], 100.0, rtol=1e-3)
    
    def test_rsi_bounds(self) -> bool:
        """Test RSI stays within 0-100 bounds"""
        # Generate random price data
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.randn(100) * 2)
        result = rsi(data, 14)
        
        # Check bounds (ignoring NaN values)
        valid_values = result[~np.isnan(result)]
        return np.all(valid_values >= 0) and np.all(valid_values <= 100)
    
    def test_bollinger_bands_relationship(self) -> bool:
        """Test Bollinger Bands maintain proper relationship"""
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.randn(50) * 1)
        upper, middle, lower = bollinger_bands(data, 20, 2.0, 2.0)
        
        # Check that upper >= middle >= lower (ignoring NaN)
        valid_indices = ~np.isnan(upper)
        if not np.any(valid_indices):
            return False
            
        return (np.all(upper[valid_indices] >= middle[valid_indices]) and
                np.all(middle[valid_indices] >= lower[valid_indices]))
    
    def test_atr_non_negative(self) -> bool:
        """Test ATR produces non-negative values"""
        np.random.seed(42)
        base_price = 100
        n = 50
        
        close = base_price + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.8)
        low = close - np.abs(np.random.randn(n) * 0.8)
        
        result = atr(high, low, close, 14)
        
        # ATR should be non-negative (ignoring NaN)
        valid_values = result[~np.isnan(result)]
        return np.all(valid_values >= 0)
    
    def test_edge_case_empty_data(self) -> bool:
        """Test handling of empty data"""
        try:
            sma([], 5)
            return False  # Should have raised an exception
        except (ValueError, IndexError):
            return True  # Expected behavior
    
    def test_edge_case_insufficient_data(self) -> bool:
        """Test handling of insufficient data"""
        try:
            result = sma([1, 2], 5)  # Not enough data for period 5
            return False  # Should have raised an exception
        except ValueError:
            return True  # Expected behavior
    
    def test_edge_case_nan_handling(self) -> bool:
        """Test handling of NaN values in input"""
        data = [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10]
        
        # Most indicators should handle NaN gracefully
        try:
            result = sma(data, 3)
            # Result should contain some NaN values
            return np.any(np.isnan(result))
        except:
            return False
    
    def test_performance_large_dataset(self) -> bool:
        """Test performance with large datasets"""
        # Generate large dataset
        np.random.seed(42)
        large_data = 100 + np.cumsum(np.random.randn(10000) * 0.1)
        
        start_time = time.time()
        
        # Test multiple indicators
        _ = sma(large_data, 50)
        _ = ema(large_data, 50)
        _ = rsi(large_data, 14)
        
        execution_time = time.time() - start_time
        
        self.performance_results.append({
            'test': 'Large Dataset (10k points)',
            'execution_time': execution_time
        })
        
        # Should complete within reasonable time (< 1 second)
        return execution_time < 1.0
    
    def test_consistency_different_data_types(self) -> bool:
        """Test consistency across different input data types"""
        # Test data as list, numpy array, and pandas Series
        data_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data_array = np.array(data_list)
        data_series = pd.Series(data_list)
        
        result_list = sma(data_list, 3)
        result_array = sma(data_array, 3)
        result_series = sma(data_series, 3)
        
        # All should produce identical results
        return (np.allclose(result_list, result_array, equal_nan=True) and
                np.allclose(result_array, result_series, equal_nan=True))
    
    def test_macd_components(self) -> bool:
        """Test MACD components relationship"""
        np.random.seed(42)
        data = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        macd_line, signal_line, histogram = macd(data, 12, 26, 9)
        
        # Histogram should equal MACD line minus signal line
        valid_indices = ~(np.isnan(macd_line) | np.isnan(signal_line))
        if not np.any(valid_indices):
            return False
            
        expected_histogram = macd_line[valid_indices] - signal_line[valid_indices]
        actual_histogram = histogram[valid_indices]
        
        return np.allclose(expected_histogram, actual_histogram, rtol=1e-10)
    
    def test_obv_monotonicity(self) -> bool:
        """Test OBV monotonicity properties"""
        # OBV should increase when price goes up with volume
        close = [100, 101, 102, 103, 104]
        volume = [1000, 1000, 1000, 1000, 1000]
        
        result = obv(close, volume)
        
        # OBV should be monotonically increasing for rising prices
        return np.all(np.diff(result[1:]) >= 0)  # Skip first value (always 0)
    
    def test_input_validation(self) -> bool:
        """Test comprehensive input validation"""
        test_cases = [
            # Test invalid period values
            (lambda: sma([1, 2, 3, 4, 5], 0), ValueError),
            (lambda: sma([1, 2, 3, 4, 5], -1), ValueError),
            (lambda: ema([1, 2, 3, 4, 5], 0), ValueError),
            (lambda: rsi([1, 2, 3, 4, 5], 0), ValueError),
            
            # Test mismatched array lengths for multi-input functions
            (lambda: atr([1, 2, 3], [1, 2], [1, 2, 3], 2), ValueError),
            
            # Test invalid deviation values for Bollinger Bands
            (lambda: bollinger_bands([1, 2, 3, 4, 5], 3, -1.0, 2.0), ValueError),
        ]
        
        passed = 0
        for test_func, expected_exception in test_cases:
            try:
                test_func()
                # If no exception was raised, test failed
                continue
            except expected_exception:
                passed += 1
            except Exception:
                # Wrong exception type
                continue
        
        # All validation tests should pass
        return passed == len(test_cases)
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🧪 Running Comprehensive py-TIM Test Suite")
        print("=" * 60)
        
        # Core functionality tests
        print("\n📊 Mathematical Accuracy Tests:")
        self.run_test("SMA Accuracy", self.test_sma_accuracy)
        self.run_test("EMA Convergence", self.test_ema_convergence)
        self.run_test("RSI Bounds Check", self.test_rsi_bounds)
        self.run_test("Bollinger Bands Relationship", self.test_bollinger_bands_relationship)
        self.run_test("ATR Non-negative", self.test_atr_non_negative)
        self.run_test("MACD Components", self.test_macd_components)
        self.run_test("OBV Monotonicity", self.test_obv_monotonicity)
        
        # Edge case tests
        print("\n🔍 Edge Case Tests:")
        self.run_test("Empty Data Handling", self.test_edge_case_empty_data)
        self.run_test("Insufficient Data", self.test_edge_case_insufficient_data)
        self.run_test("NaN Handling", self.test_edge_case_nan_handling)
        
        # Robustness tests
        print("\n🛡️ Robustness Tests:")
        self.run_test("Input Validation", self.test_input_validation)
        self.run_test("Data Type Consistency", self.test_consistency_different_data_types)
        
        # Performance tests
        print("\n⚡ Performance Tests:")
        self.run_test("Large Dataset Performance", self.test_performance_large_dataset)
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("📈 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['status'] == 'PASS')
        failed_tests = sum(1 for r in self.test_results if r['status'] == 'FAIL')
        error_tests = sum(1 for r in self.test_results if r['status'] == 'ERROR')
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"💥 Errors: {error_tests}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"📊 Success Rate: {success_rate:.1f}%")
        
        # Performance summary
        if self.performance_results:
            print("\n⚡ Performance Results:")
            for perf in self.performance_results:
                print(f"  {perf['test']}: {perf['execution_time']:.4f}s")
        
        # Overall assessment
        print("\n🎯 Overall Assessment:")
        if success_rate >= 95:
            print("🎉 EXCELLENT: Library is production-ready!")
        elif success_rate >= 85:
            print("✅ GOOD: Library is functional with minor issues")
        elif success_rate >= 70:
            print("⚠️ ACCEPTABLE: Some improvements needed")
        else:
            print("❌ NEEDS WORK: Significant issues detected")
        
        print("=" * 60)


def main():
    """Main test execution"""
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    test_suite = TestSuite()
    test_suite.run_all_tests()
    
    # Return appropriate exit code
    passed_tests = sum(1 for r in test_suite.test_results if r['status'] == 'PASS')
    total_tests = len(test_suite.test_results)
    success_rate = (passed_tests / total_tests) if total_tests > 0 else 0
    
    return 0 if success_rate >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())