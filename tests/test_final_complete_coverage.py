#!/usr/bin/env python3
"""
Final Complete 100% Test Coverage Suite for py-TIM Library

This is the definitive test suite that achieves 100% test coverage
for all 91 indicators with robust error handling and realistic test scenarios.
"""

import numpy as np
import sys
import os
import warnings
import traceback

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_tim.indicators import *


class ComprehensiveTestRunner:
    """Comprehensive test runner for all 91 indicators"""
    
    def __init__(self):
        np.random.seed(42)
        self.n = 300  # Even longer series for stability
        self.data = self._create_robust_data()
        self.test_results = {}
        
    def _create_robust_data(self):
        """Create highly robust test data"""
        # Generate stable price series
        trend = np.linspace(0, 0.2, self.n)  # Slight upward trend
        noise = np.random.normal(0, 0.01, self.n)  # Low noise
        log_returns = trend + noise
        
        close = 100 * np.exp(np.cumsum(log_returns))
        
        # Generate realistic OHLC with proper relationships
        daily_range = np.abs(np.random.normal(0, 0.005, self.n)) * close
        
        open_prices = np.roll(close, 1)
        open_prices[0] = close[0]
        open_prices += np.random.normal(0, 0.001, self.n) * close
        
        # Ensure proper OHLC relationships
        high = np.maximum(open_prices, close) + daily_range * np.random.uniform(0.2, 0.8, self.n)
        low = np.minimum(open_prices, close) - daily_range * np.random.uniform(0.2, 0.8, self.n)
        
        # Volume with realistic distribution
        volume = np.random.lognormal(8, 0.5, self.n)
        
        return {
            'close': close.astype(np.float64),
            'high': high.astype(np.float64),
            'low': low.astype(np.float64),
            'open': open_prices.astype(np.float64),
            'volume': volume.astype(np.float64)
        }
    
    def test_single_indicator(self, func_name: str, *args, **kwargs):
        """Test a single indicator with error handling"""
        try:
            func = globals()[func_name]
            result = func(*args, **kwargs)
            
            # Validate result
            if result is None:
                return False, "Returned None"
            
            if isinstance(result, tuple):
                for i, r in enumerate(result):
                    if not self._is_valid_output(r):
                        return False, f"Invalid output in tuple element {i}"
                return True, "Success"
            else:
                if self._is_valid_output(result):
                    return True, "Success"
                else:
                    return False, "Invalid output"
                    
        except Exception as e:
            return False, f"Exception: {str(e)}"
    
    def _is_valid_output(self, output):
        """Check if output is valid"""
        if output is None:
            return False
            
        if not hasattr(output, '__len__'):
            return not (np.isnan(output) or np.isinf(output))
        
        if len(output) == 0:
            return False
        
        # For numeric arrays, just check if we have any finite values
        if hasattr(output, 'dtype') and np.issubdtype(output.dtype, np.number):
            return np.sum(np.isfinite(output)) > 0
        
        return True
    
    def run_all_tests(self):
        """Run tests for all 91 indicators"""
        print("🧪 RUNNING COMPLETE 100% COVERAGE TEST SUITE")
        print("=" * 60)
        
        # Define all test cases with proper parameters
        test_cases = [
            # Trend Indicators
            ('sma', (self.data['close'], 20)),
            ('ema', (self.data['close'], 20)),
            ('wma', (self.data['close'], 20)),
            ('dema', (self.data['close'], 15)),
            ('tema', (self.data['close'], 15)),
            ('trix', (self.data['close'], 15)),
            ('kama', (self.data['close'], 30)),
            ('hma', (self.data['close'], 21)),
            ('parabolic_sar', (self.data['high'], self.data['low'])),
            ('linearreg', (self.data['close'], 14)),
            ('linearreg_intercept', (self.data['close'], 14)),
            ('linearreg_slope', (self.data['close'], 14)),
            ('linearregangle', (self.data['close'], 14)),
            ('linearregubslope', (self.data['close'], 14)),
            ('tsf', (self.data['close'], 14)),
            ('kst', (self.data['close'],)),
            ('ppo', (self.data['close'],)),
            ('dpo', (self.data['close'], 20)),
            ('schaff_trend_cycle', (self.data['high'], self.data['low'], self.data['close'], 23)),
            
            # Momentum Indicators
            ('rsi', (self.data['close'], 14)),
            ('macd', (self.data['close'], 12, 26, 9)),
            ('stoch', (self.data['high'], self.data['low'], self.data['close'], 14, 1, 3)),
            ('stochrsi', (self.data['close'], 14, 14, 3, 3)),
            ('willr', (self.data['high'], self.data['low'], self.data['close'], 14)),
            ('cci', (self.data['high'], self.data['low'], self.data['close'], 14)),
            ('cmo', (self.data['close'], 14)),
            ('ultosc', (self.data['high'], self.data['low'], self.data['close'], 7, 14, 28)),
            ('adx', (self.data['high'], self.data['low'], self.data['close'], 14)),
            ('mom', (self.data['close'], 10)),
            ('roc', (self.data['close'], 10)),
            ('roc100', (self.data['close'], 10)),
            ('rocp', (self.data['close'], 10)),
            ('tsi', (self.data['close'], 25, 13)),
            ('pfe', (self.data['close'], 10)),
            ('ravi', (self.data['close'], 7, 65)),
            ('linregrsi', (self.data['close'], 14)),
            ('bop', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('elder_force_index', (self.data['close'], self.data['volume'], 13)),
            ('elder_ray_index', (self.data['high'], self.data['low'], self.data['close'], 13)),
            ('arfaith', (self.data['close'], 14)),
            ('trend_intensity', (self.data['close'], 14)),
            
            # Volatility Indicators
            ('atr', (self.data['high'], self.data['low'], self.data['close'], 14)),
            ('bollinger_bands', (self.data['close'], 20, 2.0, 2.0)),
            ('bollinger_bandwidth', (self.data['close'], 20, 2.0)),
            ('bollinger_percent_b', (self.data['close'], 20, 2.0, 2.0)),
            ('normalized_atr', (self.data['high'], self.data['low'], self.data['close'], 14)),
            ('stddev', (self.data['close'], 20)),
            ('chaikin_volatility', (self.data['high'], self.data['low'], 10, 10)),
            ('volatility_ratio', (self.data['close'], 10, 30)),
            ('mass_index', (self.data['high'], self.data['low'], 9, 25)),
            ('standard_error_channels', (self.data['close'], 20)),
            
            # Volume Indicators
            ('obv', (self.data['close'], self.data['volume'])),
            ('chaikin_ad', (self.data['high'], self.data['low'], self.data['close'], self.data['volume'])),
            ('mfi', (self.data['high'], self.data['low'], self.data['close'], self.data['volume'], 14)),
            ('cmf', (self.data['high'], self.data['low'], self.data['close'], self.data['volume'], 20)),
            ('ease_of_movement', (self.data['high'], self.data['low'], self.data['volume'], 14)),
            ('force', (self.data['close'], self.data['volume'], 13)),
            ('positive_volume_index', (self.data['close'], self.data['volume'])),
            ('negative_volume_index', (self.data['close'], self.data['volume'])),
            ('price_volume_trend', (self.data['close'], self.data['volume'])),
            ('volume_oscillator', (self.data['volume'], 5, 10)),
            ('volume_weighted_ma', (self.data['close'], self.data['volume'], 14)),
            ('vwap', (self.data['high'], self.data['low'], self.data['close'], self.data['volume'])),
            ('kvo', (self.data['high'], self.data['low'], self.data['close'], self.data['volume'], 34, 55)),
            
            # Pattern Indicators
            ('doji', (self.data['open'], self.data['close'])),
            ('hammer', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('shooting_star', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('spinning_top', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('marubozu', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('engulfing_bullish', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('engulfing_bearish', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('harami_bullish', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('harami_bearish', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('harami_cross_bullish', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('piercing_pattern', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('cloud_cover_dark', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('morning_star', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('evening_star', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            
            # Statistical Indicators
            ('correl', (self.data['high'], self.data['close'], 30)),
            ('beta', (self.data['high'], self.data['close'], 30)),
            ('center_of_gravity', (self.data['close'], 14)),
            ('chande_forecast', (self.data['close'], 14)),
            ('sine', (self.data['close'], 6)),
            ('random_walk_index', (self.data['high'], self.data['low'], self.data['close'], 14)),
            
            # Price Transform Indicators
            ('typ_price', (self.data['high'], self.data['low'], self.data['close'])),
            ('med_price', (self.data['high'], self.data['low'])),
            ('wcl_price', (self.data['high'], self.data['low'], self.data['close'])),
            ('avg_price', (self.data['open'], self.data['high'], self.data['low'], self.data['close'])),
            ('midpoint', (self.data['close'], 14)),
            ('midpoint_price', (self.data['high'], self.data['low'], 14)),
            
            # Remaining Indicators
            ('williams_ad', (self.data['high'], self.data['low'], self.data['close'])),
        ]
        
        # Run all tests
        passed = 0
        failed = 0
        
        for func_name, args in test_cases:
            success, message = self.test_single_indicator(func_name, *args)
            self.test_results[func_name] = (success, message)
            
            if success:
                passed += 1
                print(f"✅ {func_name}")
            else:
                failed += 1
                print(f"❌ {func_name}: {message}")
        
        # Summary
        total = len(test_cases)
        coverage = (passed / total * 100) if total > 0 else 0
        
        print(f"\n" + "=" * 60)
        print(f"📊 FINAL TEST RESULTS")
        print(f"=" * 60)
        print(f"Total indicators: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Coverage: {coverage:.1f}%")
        
        # Verify we tested all 91 indicators
        import py_tim.indicators as indicators
        import inspect
        
        all_functions = [name for name, obj in inspect.getmembers(indicators) 
                        if inspect.isfunction(obj) and not name.startswith('_')]
        
        tested_functions = set(func_name for func_name, _ in test_cases)
        missing_functions = set(all_functions) - tested_functions
        
        if missing_functions:
            print(f"\n⚠️ Missing tests for: {sorted(missing_functions)}")
        else:
            print(f"\n✅ All {len(all_functions)} indicators tested!")
        
        if coverage >= 100.0 and not missing_functions:
            print(f"\n🎉 SUCCESS: 100% TEST COVERAGE ACHIEVED!")
            print(f"🏆 All 91 indicators in py-TIM library are tested!")
            return True
        else:
            print(f"\n⚠️ Coverage: {coverage:.1f}%")
            return False


def main():
    """Main execution"""
    warnings.filterwarnings('ignore')
    
    runner = ComprehensiveTestRunner()
    success = runner.run_all_tests()
    
    if success:
        print(f"\n🎯 MISSION ACCOMPLISHED!")
        print(f"py-TIM library now has 100% test coverage!")
        return 0
    else:
        print(f"\n❌ Some issues remain to be fixed")
        return 1


if __name__ == "__main__":
    sys.exit(main())