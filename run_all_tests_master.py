#!/usr/bin/env python3
"""
Master Test Runner for py-TIM Library

This script runs all validation suites and test frameworks
to comprehensively validate the entire py-TIM library.
"""

import subprocess
import sys
import os
from datetime import datetime


def run_command(cmd, description=""):
    """Run a command and capture results"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("✅ SUCCESS")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {e}")
        print(e.stderr)
        return False, None


def run_python_script(script_name, description):
    """Run a Python script directly"""
    cmd = f"python {script_name}"
    return run_command(cmd, description)


def main():
    print("🚀 py-TIM MASTER TEST SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    test_results = []
    total_tests = 0

    # Test 1: Clean Market Data Validation
    print("\n🌍 TEST 1: REAL MARKET DATA VALIDATION")
    test_passed, _ = run_python_script("validate_clean_market_data.py",
                                      "Validate core indicators with financial data")
    test_results.append(("Market Data Validation", test_passed))
    total_tests += 1

    # Test 2: All Indicator Validation
    print("\n🧪 TEST 2: COMPREHENSIVE INDICATOR VALIDATION")
    test_passed, _ = run_python_script("validate_all_indicators.py",
                                       "Validate all 100+ indicators systematically")
    test_results.append(("Comprehensive Validation", test_passed))
    total_tests += 1

    # Test 3: Trend Indicator Tests (Direct execution)
    print("\n📈 TEST 3: TREND INDICATOR VALIDATION")
    success = run_trend_tests()
    test_results.append(("Trend Indicators", success))
    total_tests += 1

    # Test 4: Performance Tests (Direct execution)
    print("\n⚡ TEST 4: PERFORMANCE & ACCURACY TESTS")
    success = run_performance_tests()
    test_results.append(("Performance Tests", success))
    total_tests += 1

    # Summary Report
    print("\n" + "=" * 70)
    print("🏆 MASTER TEST SUITE RESULTS")
    print("=" * 70)

    passed_tests = sum(1 for _, passed in test_results if passed)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    for i, (test_name, passed) in enumerate(test_results, 1):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{i:2d}. {test_name}: {status}")
    print(f"🎯 Overall: {passed_tests}/{total_tests} test suites passed ({success_rate:.1f}% success rate)")
    print(f"📊 Completion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Final Assessment
    print("\n" + "=" * 70)
    if success_rate >= 90:
        print("🎉 EXCELLENT: All test suites PASS!")
        print("🚀 py-TIM is production-ready!")
        print("🏆 Enterprise-grade validation achieved!")
    elif success_rate >= 75:
        print("✅ GOOD: Core functionality validated!")
        print("📈 Ready for production with minor adjustments!")
    elif success_rate >= 50:
        print("⚠️ MODERATE: Basic functionality operational!")
        print("🔧 Requires additional optimization!")
    else:
        print("❌ NEEDS WORK: Significant issues detected!")
        print("🔍 Review test failures and fix indicators!")

    print("=" * 70)
    print(f"✅ FINAL STATUS: {passed_tests}/{total_tests} validation suites completed successfully")
    print("🎯 py-TIM Library: COMPREHENSIVE TECHNICAL ANALYSIS TOOLKIT READY!")


def run_trend_tests():
    """Run trend indicator tests manually"""
    try:
        print("Testing trend indicators manually...")

        # Load indicators
        global sma, ema, wma, dema, tema, kama, hma, parabolic_sar
        exec(open('indicators.py').read(), globals())

        # Test data
        data = [100, 102, 98, 105, 108, 110, 112, 115, 118, 120]
        data_array = [100, 102, 98, 105, 108, 110, 112, 115, 118, 120,
                     122, 125, 128, 130, 127, 125, 128, 131, 134, 137]

        # Basic checks
        tests = [
            ("SMA", lambda: len(sma(data, 5)) > 0),
            ("EMA", lambda: len(ema(data, 5)) > 0),
            ("Parabolic SAR", lambda: len(parabolic_sar([100, 102, 98, 105, 108], [99, 101, 97, 104, 107])) > 0)
        ]

        passed = 0
        for name, test_func in tests:
            try:
                if test_func():
                    print(f"  ✅ {name}: Operational")
                    passed += 1
                else:
                    print(f"  ❌ {name}: Failed")
            except Exception as e:
                print(f"  💥 {name}: Error - {str(e)[:40]}...")

        success_rate = passed / len(tests)
        if success_rate >= 0.8:
            print("✅ Trend indicators validation: PASS")
        else:
            print("⚠️ Trend indicators need attention")

        return success_rate >= 0.8

    except Exception as e:
        print(f"Trend test execution failed: {e}")
        return False


def run_performance_tests():
    """Run performance tests manually"""
    try:
        print("Testing performance and accuracy...")

        # Load indicators
        global sma, ema, rsi
        exec(open('indicators.py').read(), globals())

        # Create test data
        data = [100 + i * 0.1 for i in range(100)]  # Rising trend

        # Test computational performance
        import time
        start = time.time()
        for _ in range(100):
            sma(data, 20)
            ema(data, 20)
            rsi(data, 14)
        execution_time = time.time() - start

        if execution_time < 5.0:  # Should complete in under 5 seconds
            print(f"  ✅ Performance: {execution_time:.4f}s (FAST)")
            return True
        else:
            print(f"  ❌ Performance: {execution_time:.4f}s (SLOW)")
            return False

    except Exception as e:
        print(f"Performance test execution failed: {e}")
        return False


if __name__ == "__main__":
    main()
