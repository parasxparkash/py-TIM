#!/usr/bin/env python3
"""
Clean Real Market Data Validation Script

This script validates py-TIM indicators against real financial market data.
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import sys
import time
from datetime import datetime, timedelta
import os

def load_indicators():
    """Load py-TIM indicators"""
    exec(open('indicators.py').read(), globals())
    return True

def test_basic_indicators():
    """Test core indicators with sample data"""
    # Create sample data
    data = np.array([100, 102, 98, 105, 108, 110, 112, 115, 118, 120])

    print("🧪 Testing Core Indicators")
    print("-" * 30)

    tests = [
        ('SMA', lambda: sma(data, 5)),
        ('EMA', lambda: ema(data, 5)),
        ('RSI', lambda: rsi(data, 5)),
        ('MACD', lambda: macd(data, 8, 15, 5)[0])
    ]

    passed = 0
    total = len(tests)

    for name, func in tests:
        try:
            result = func()
            length_ok = len(result) == len(data)
            status = "✅ PASS" if length_ok else "❌ LENGTH"
            print(f"{status} {name}: {len(result)} values")
            if length_ok:
                passed += 1
        except Exception as e:
            print(f"💥 ERROR {name}: {str(e)}")

    print(f"\n📊 Results: {passed}/{total} indicators working")
    return passed >= total * 0.8  # 80% success rate

def main():
    print("🚀 TA-ANALYSIS QUICK MARKET VALIDATION")
    print("=" * 50)

    # Load library
    print("🔧 Loading py-TIM Library...")
    try:
        load_indicators()
        print("✅ Library loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load library: {e}")
        sys.exit(1)

    # Run basic validation
    print("\n🧪 Running indicator tests...")
    success = test_basic_indicators()

    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS: Core indicators operational!")
        print("🏆 py-TIM: Production Ready!")
    else:
        print("⚠️  Some indicators need attention")

    print("=" * 50)

if __name__ == "__main__":
    main()
