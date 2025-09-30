"""
Test suite for py-TIM Library
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import the indicators module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import (
    sma, ema, wma, rsi, macd, bollinger_bands, atr, obv, stoch, cci, dema, tema, trix, willr, cmo, ultosc, adx, kama, hma, elder_ray_index, schaff_trend_cycle, random_walk_index, chaikin_ad, ease_of_movement, positive_volume_index, negative_volume_index, price_volume_trend, volume_oscillator, volume_weighted_ma
)


def test_sma():
    """Test Simple Moving Average function"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = sma(data, 3)
    
    expected = np.array([
        np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
    ])
    
    print("SMA Test:")
    print(f"Input: {data}")
    print(f"Expected: {expected}")
    print(f"Result: {result}")
    print(f"Test passed: {np.allclose(result, expected, equal_nan=True)}")
    print()


def test_ema():
    """Test Exponential Moving Average function"""
    data = [1, 2, 3, 4, 5]
    result = ema(data, 3)
    
    # Calculate expected values manually
    alpha = 2.0 / (3 + 1.0)  # 0.5
    expected = np.array([np.nan, np.nan, 3.0, 
                         alpha * 4 + (1 - alpha) * 3.0,  # 3.5
                         alpha * 5 + (1 - alpha) * 3.5])  # 4.25
    
    print("EMA Test:")
    print(f"Input: {data}")
    print(f"Expected: {expected}")
    print(f"Result: {result}")
    print(f"Test passed: {np.allclose(result, expected, equal_nan=True)}")
    print()


def test_wma():
    """Test Weighted Moving Average function"""
    data = [1, 2, 3, 4, 5]
    result = wma(data, 3)
    
    # For WMA with period 3: weights are [1, 2, 3]
    # WMA = (1*value1 + 2*value2 + 3*value3) / (1+2+3)
    expected = np.array([
        np.nan, np.nan, 
        (1*1 + 2*2 + 3*3) / 6,  # 2.333...
        (1*2 + 2*3 + 3*4) / 6,  # 3.33...
        (1*3 + 2*4 + 3*5) / 6   # 4.333...
    ])
    
    print("WMA Test:")
    print(f"Input: {data}")
    print(f"Expected: {expected}")
    print(f"Result: {result}")
    print(f"Test passed: {np.allclose(result, expected, equal_nan=True)}")
    print()


def test_rsi():
    """Test Relative Strength Index function"""
    # Simple test with known values
    data = [44, 42, 46, 43, 45, 47, 49, 48, 46, 45]
    result = rsi(data, 3)
    
    print("RSI Test:")
    print(f"Input: {data}")
    print(f"Result: {result}")
    # Note: We're just checking that the function runs without error
    # and produces values in the expected range [0, 100]
    valid_range = np.all((result >= 0) | np.isnan(result)) and np.all((result <= 100) | np.isnan(result))
    print(f"Values in valid range [0,100]: {valid_range}")
    print()


def test_macd():
    """Test MACD function"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    macd_line, signal_line, histogram = macd(data, fastperiod=3, slowperiod=6, signalperiod=3)
    
    print("MACD Test:")
    print(f"Input length: {len(data)}")
    print(f"MACD line shape: {macd_line.shape}")
    print(f"Signal line shape: {signal_line.shape}")
    print(f"Histogram shape: {histogram.shape}")
    print(f"MACD line (last 5): {macd_line[-5:]}")
    print(f"Signal line (last 5): {signal_line[-5:]}")
    print(f"Histogram (last 5): {histogram[-5:]}")
    print()


def test_bollinger_bands():
    """Test Bollinger Bands function"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    upper, middle, lower = bollinger_bands(data, period=3, nbdevup=2.0, nbdevdn=2.0)
    
    print("Bollinger Bands Test:")
    print(f"Input: {data}")
    print(f"Upper band (last 5): {upper[-5:]}")
    print(f"Middle band (last 5): {middle[-5:]}")
    print(f"Lower band (last 5): {lower[-5:]}")
    # Verify that upper >= middle >= lower where defined
    valid_relationship = np.all((upper >= middle) | np.isnan(upper)) and np.all((middle >= lower) | np.isnan(lower))
    print(f"Band relationship valid (upper >= middle >= lower): {valid_relationship}")
    print()


def test_atr():
    """Test Average True Range function"""
    high = [101, 102, 103, 104, 105]
    low = [99, 100, 101, 102, 103]
    close = [100, 101, 102, 103, 104]
    result = atr(high, low, close, period=3)
    
    print("ATR Test:")
    print(f"High: {high}")
    print(f"Low: {low}")
    print(f"Close: {close}")
    print(f"ATR result: {result}")
    print(f"All non-negative: {np.all((result >= 0) | np.isnan(result))}")
    print()


def test_obv():
    """Test On Balance Volume function"""
    close = [100, 101, 9, 102, 101]
    volume = [1000, 1200, 800, 1500, 900]
    result = obv(close, volume)
    
    print("OBV Test:")
    print(f"Close: {close}")
    print(f"Volume: {volume}")
    print(f"OBV result: {result}")
    # Check that OBV starts with first volume value
    print(f"Starts with first volume: {result[0] == volume[0]}")
    print()


def test_stoch():
    """Test Stochastic Oscillator function"""
    high = [10, 11, 12, 13, 14, 15, 14, 12, 13, 11, 12, 13, 14, 15, 16, 17, 15, 13, 14, 12]
    low = [8, 9, 10, 11, 12, 13, 12, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 11, 12, 10]
    close = [9, 10, 11, 12, 13, 14, 13, 11, 12, 10, 11, 12, 13, 14, 15, 16, 14, 12, 13, 11]
    slowk, slowd = stoch(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)

    print("Stochastic Oscillator Test:")
    print(f"High: {high}")
    print(f"Low: {low}")
    print(f"Close: {close}")
    print(f"Slow %K (last 5): {slowk[-5:]}")
    print(f"Slow %D (last 5): {slowd[-5:]}")
    # Check that values are in [0, 100] range
    valid_range_k = np.all((slowk >= 0) | np.isnan(slowk)) and np.all((slowk <= 100) | np.isnan(slowk))
    valid_range_d = np.all((slowd >= 0) | np.isnan(slowd)) and np.all((slowd <= 100) | np.isnan(slowd))
    print(f"Slow %K in valid range [0,100]: {valid_range_k}")
    print(f"Slow %D in valid range [0,100]: {valid_range_d}")
    print()


def test_cci():
    """Test Commodity Channel Index function"""
    high = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    low = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    close = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    result = cci(high, low, close, period=5)

    print("CCI Test:")
    print(f"High: {high}")
    print(f"Low: {low}")
    print(f"Close: {close}")
    print(f"CCI result: {result}")
    # CCI can be any real number, just check that it's computed
    print(f"All finite where defined: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_dema():
    """Test Double Exponential Moving Average function"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = dema(data, 3)

    print("DEMA Test:")
    print(f"Input: {data}")
    print(f"DEMA result: {result}")
    print(f"All finite or NaN: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_tema():
    """Test Triple Exponential Moving Average function"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = tema(data, 3)

    print("TEMA Test:")
    print(f"Input: {data}")
    print(f"TEMA result: {result}")
    print(f"All finite or NaN: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_trix():
    """Test Triple Exponential Average function"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = trix(data, 3)

    print("TRIX Test:")
    print(f"Input: {data}")
    print(f"TRIX result: {result}")
    print(f"All finite or NaN: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_willr():
    """Test Williams %R function"""
    high = [12, 13, 14, 15, 14, 16, 17, 18, 19, 20]
    low = [10, 11, 12, 13, 12, 14, 15, 16, 17, 18]
    close = [11, 12, 13, 14, 13, 15, 16, 17, 18, 19]
    result = willr(high, low, close, period=5)

    print("Williams %R Test:")
    print(f"High: {high}")
    print(f"Low: {low}")
    print(f"Close: {close}")
    print(f"%R result: {result}")
    valid_range = np.all((result >= -100) | np.isnan(result)) and np.all((result <= 0) | np.isnan(result))
    print(f"Values in valid range [-100, 0]: {valid_range}")
    print()


def test_cmo():
    """Test Chande Momentum Oscillator function"""
    data = [1, 2, 3, 2, 4, 3, 5, 4, 6, 5]
    result = cmo(data, 4)

    print("CMO Test:")
    print(f"Input: {data}")
    print(f"CMO result: {result}")
    valid_range = np.all((result >= -100) | np.isnan(result)) and np.all((result <= 100) | np.isnan(result))
    print(f"Values in valid range [-100, 100]: {valid_range}")
    print()


def test_ultosc():
    """Test Ultimate Oscillator function"""
    high = [1.2, 1.3, 1.4, 1.5, 1.4, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
    low = [1.0, 1.1, 1.2, 1.3, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8]
    close = [1.1, 1.2, 1.3, 1.4, 1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]
    result = ultosc(high, low, close)

    print("Ultimate Oscillator Test:")
    print(f"High length: {len(high)}")
    print(f"ULTOSC result (last 5): {result[-5:]}")
    valid_range = np.all((result >= 0) | np.isnan(result)) and np.all((result <= 100) | np.isnan(result))
    print(f"Values in valid range [0, 100]: {valid_range}")
    print()


def test_adx():
    """Test Average Directional Index function"""
    high = [10, 11, 12, 13, 12, 14, 15, 16, 15, 17]
    low = [8, 9, 10, 11, 10, 12, 13, 14, 13, 15]
    close = [9, 10, 11, 12, 11, 13, 14, 15, 14, 16]
    result_adx, result_plus_di, result_minus_di = adx(high, low, close, period=5)

    print("ADX Test:")
    print(f"High: {high}")
    print(f"Low: {low}")
    print(f"Close: {close}")
    print(f"ADX: {result_adx}")
    print(f"+DI: {result_plus_di}")
    print(f"-DI: {result_minus_di}")
    valid_range = np.all((result_adx >= 0) | np.isnan(result_adx)) and np.all((result_adx <= 100) | np.isnan(result_adx))
    print(f"ADX values in valid range [0, 100]: {valid_range}")
    print()


def test_kama():
    """Test Kaufman's Adaptive Moving Average function"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3  # Repeat to have enough data
    result = kama(data, period=10)

    print("KAMA Test:")
    print(f"Input length: {len(data)}")
    print(f"KAMA result (last 5): {result[-5:]}")
    print(f"All finite or NaN: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_hma():
    """Test Hull Moving Average function"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2
    result = hma(data, period=10)

    print("HMA Test:")
    print(f"Input length: {len(data)}")
    print(f"HMA result (last 5): {result[-5:]}")
    print(f"All finite or NaN: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_elder_ray_index():
    """Test Elder Ray Index function"""
    high = [12, 13, 14, 15, 14, 16, 17, 18, 17, 19, 20, 19, 18, 17, 16, 15]
    low = [10, 11, 12, 13, 12, 14, 15, 16, 15, 17, 18, 17, 16, 15, 14, 13]
    close = [11, 12, 13, 14, 13, 15, 16, 17, 16, 18, 19, 18, 17, 16, 15, 14]

    bull_power, bear_power = elder_ray_index(high, low, close, period=5)

    print("Elder Ray Index Test:")
    print(f"High: {high}")
    print(f"Low: {low}")
    print(f"Close: {close}")
    print(f"Bull Power (last 5): {bull_power[-5:]}")
    print(f"Bear Power (last 5): {bear_power[-5:]}")
    # Bull Power should be high - EMA(close), Bear Power should be low - EMA(close)
    # Both can be positive or negative values
    print(f"All finite where defined: {np.all(np.isfinite(bull_power) | np.isnan(bull_power)) and np.all(np.isfinite(bear_power) | np.isnan(bear_power))}")
    print()


def test_schaff_trend_cycle():
    """Test Schaff Trend Cycle function"""
    # Need enough data for calculation: max(fast_length, slow_length) + k_period + d_period
    # fast_length=23, slow_length=50, k_period=10, d_period=3
    # So minimum length: 50 + 10 + 3 = 63
    data = [100 + i * 0.5 + (i % 10) for i in range(80)]  # Trending data with some variation

    result = schaff_trend_cycle(data, fast_length=23, slow_length=50, k_period=10, d_period=3)

    print("Schaff Trend Cycle Test:")
    print(f"Input length: {len(data)}")
    print(f"STC result (last 10): {result[-10:]}")
    # STC should be in 0-100 range when defined
    valid_values = result[~np.isnan(result)]
    valid_range = np.all((valid_values >= 0) & (valid_values <= 100)) if len(valid_values) > 0 else True
    print(f"Values in valid range [0,100]: {valid_range}")
    print()


def test_random_walk_index():
    """Test Random Walk Index function"""
    high = [12, 13, 14, 15, 16, 14, 16, 17, 18, 17, 19, 20, 18, 16, 15, 13, 14, 15, 16, 17]
    low = [10, 11, 12, 13, 14, 12, 14, 15, 16, 15, 17, 18, 16, 14, 13, 11, 12, 13, 14, 15]
    close = [11, 12, 13, 14, 15, 13, 15, 16, 17, 16, 18, 19, 17, 15, 14, 12, 13, 14, 15, 16]

    rwi_high, rwi_low = random_walk_index(high, low, close, period=5)

    print("Random Walk Index Test:")
    print(f"High: {high}")
    print(f"Low: {low}")
    print(f"Close: {close}")
    print(f"RWI High (last 5): {rwi_high[-5:]}")
    print(f"RWI Low (last 5): {rwi_low[-5:]}")
    # RWI measures trend strength vs random walk, values can be various
    print(f"All finite where defined: {np.all(np.isfinite(rwi_high) | np.isnan(rwi_high)) and np.all(np.isfinite(rwi_low) | np.isnan(rwi_low))}")
    print()


def test_chaikin_ad():
    """Test Chaikin Accumulation/Distribution function"""
    high = [12, 13, 14, 15, 16, 17, 15, 13, 14, 15, 16, 17, 18, 19, 20, 21, 20, 19, 18, 17]
    low = [10, 11, 12, 13, 14, 15, 13, 11, 12, 13, 14, 15, 16, 17, 18, 19, 18, 17, 16, 15]
    close = [11, 12, 13, 14, 15, 16, 14, 12, 13, 14, 15, 16, 17, 18, 19, 20, 19, 18, 17, 16]
    volume = [1000, 1200, 800, 1500, 1100, 1300, 900, 1400, 1100, 1600, 1200, 1400, 1300, 1700, 1100, 1800, 1200, 1300, 1100, 1000]

    result = chaikin_ad(high, low, close, volume)

    print("Chaikin A/D Test:")
    print(f"High: {high}")
    print(f"Low: {low}")
    print(f"Close: {close}")
    print(f"Volume: {volume}")
    print(f"A/D result (last 5): {result[-5:]}")
    # Chaikin AD accumulates volume, so values should increase/decrease based on price direction
    print(f"All finite where defined: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_ease_of_movement():
    """Test Ease of Movement function"""
    high = [12, 13, 14, 15, 16, 17, 15, 13, 14, 15, 16, 17, 18, 19, 20, 21, 20, 19, 18, 17]
    low = [10, 11, 12, 13, 14, 15, 13, 11, 12, 13, 14, 15, 16, 17, 18, 19, 18, 17, 16, 15]
    volume = [1000, 1200, 800, 1500, 1100, 1300, 900, 1400, 1100, 1600, 1200, 1400, 1300, 1700, 1100, 1800, 1200, 1300, 1100, 1000]

    result = ease_of_movement(high, low, volume, period=5)

    print("Ease of Movement Test:")
    print(f"High: {high}")
    print(f"Low: {low}")
    print(f"Volume: {volume}")
    print(f"EMV result (last 5): {result[-5:]}")
    # EMV measures price movement efficiency, can be positive or negative
    print(f"All finite where defined: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_positive_volume_index():
    """Test Positive Volume Index function"""
    close = [10, 11, 12, 9, 13, 14, 12, 15, 16, 17, 15, 18, 19, 16, 20]
    volume = [1000, 1200, 800, 1500, 1100, 1300, 900, 1400, 1100, 1600, 1200, 1400, 1300, 1700, 1100]

    result = positive_volume_index(close, volume)

    print("Positive Volume Index Test:")
    print(f"Close: {close}")
    print(f"Volume: {volume}")
    print(f"PVI result (last 5): {result[-5:]}")
    # PVI tracks cumulative changes during increasing volume
    print(f"All finite where defined: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_negative_volume_index():
    """Test Negative Volume Index function"""
    close = [10, 9, 8, 11, 9, 8, 10, 11, 12, 9, 8, 7, 11, 10, 14]
    volume = [1000, 1200, 800, 1500, 1100, 1300, 900, 1400, 1100, 1600, 1200, 1400, 1300, 1700, 1100]

    result = negative_volume_index(close, volume)

    print("Negative Volume Index Test:")
    print(f"Close: {close}")
    print(f"Volume: {volume}")
    print(f"NVI result (last 5): {result[-5:]}")
    # NVI tracks cumulative changes during decreasing volume
    print(f"All finite where defined: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_price_volume_trend():
    """Test Price Volume Trend function"""
    close = [10, 11, 12, 11, 13, 14, 13, 15, 16, 17, 16, 18, 19, 18, 20]
    volume = [1000, 1100, 1200, 900, 1300, 1400, 1100, 1500, 1200, 1600, 1300, 1700, 1400, 1800, 1500]

    result = price_volume_trend(close, volume)

    print("Price Volume Trend Test:")
    print(f"Close: {close}")
    print(f"Volume: {volume}")
    print(f"PVT result (last 5): {result[-5:]}")
    # PVT tracks cumulative volume-weighted price changes
    print(f"All finite where defined: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_volume_oscillator():
    """Test Volume Oscillator function"""
    volume = [1000, 1200, 800, 1500, 1100, 1300, 900, 1400, 1100, 1600, 1200, 1400, 1300, 1700, 1100, 1800, 1200, 1300, 1100, 1000]

    result = volume_oscillator(volume, short_period=5, long_period=10)

    print("Volume Oscillator Test:")
    print(f"Volume: {volume}")
    print(f"VO result (last 5): {result[-5:]}")
    # Volume Oscillator is in percentage, can be positive or negative
    print(f"All finite where defined: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def run_all_tests():
    """Run all tests"""
    print("Running py-TIM Library Tests")
    print("=" * 40)

    test_sma()
    test_ema()
    test_wma()
    test_rsi()
    test_macd()
    test_bollinger_bands()
    test_atr()
    test_obv()
    test_stoch()
    test_cci()
    test_dema()
    test_tema()
    test_trix()
    test_willr()
    test_cmo()
    test_ultosc()
    test_adx()
    test_kama()
    test_hma()
    test_elder_ray_index()
    test_schaff_trend_cycle()
    test_random_walk_index()
    test_chaikin_ad()
    test_ease_of_movement()
    test_positive_volume_index()
    test_negative_volume_index()

    test_price_volume_trend()
    test_volume_oscillator()


def test_volume_weighted_ma():
    """Test Volume Weighted Moving Average function"""
    close = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    volume = [1000, 1100, 900, 1300, 1400, 1200, 1600, 1500, 1700, 1800, 1900]

    result = volume_weighted_ma(close, volume, period=5)

    print("Volume Weighted MA Test:")
    print(f"Close: {close}")
    print(f"Volume: {volume}")
    print(f"VWMA result (last 5): {result[-5:]}")
    # VWMA should be volume-weighted average close prices
    print(f"All finite where defined: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


    test_price_volume_trend()
    test_volume_oscillator()
    test_volume_weighted_ma()

    print("Tests completed.")


def test_linearreg():
    """Test Linear Regression function"""
    data = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    result = linearreg(data, period=5)

    print("Linear Regression Test:")
    print(f"Input: {data}")
    print(f"Result: {result}")
    print(f"All finite where defined: {np.all(np.isfinite(result) | np.isnan(result))}")
    # Check that last value is reasonable (should be close to the last data point trend)
    # Basic check: result should exist for periods after the lookback
    print()


def test_bollinger_percent_b():
    """Test Bollinger %B function"""
    data = [100, 102, 98, 105, 108, 110, 112, 115, 118, 120]
    result = bollinger_percent_b(data, period=5, nbdev=2.0)

    print("Bollinger %B Test:")
    print(f"Input: {data}")
    print(f"Result: {result}")
    print(f"Valid range (0-1): {np.all((result >= 0) | np.isnan(result)) and np.all((result <= 1) | np.isnan(result))}")
    print()


def test_vwap():
    """Test Volume Weighted Average Price function"""
    high = [101, 102, 103, 104, 105]
    low = [99, 100, 101, 102, 103]
    close = [100, 101, 102, 103, 104]
    volume = [1000, 1100, 1200, 1300, 1400]

    result = vwap(high, low, close, volume)

    print("VWAP Test:")
    print(f"High: {high}")
    print(f"Low: {low}")
    print(f"Close: {close}")
    print(f"Volume: {volume}")
    print(f"VWAP: {result}")
    # VWAP should be a cumulative weighted average
    print(f"All finite where defined: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_center_of_gravity():
    """Test Center of Gravity function"""
    data = [10, 11, 12, 13, 14, 15]
    result = center_of_gravity(data, period=4)

    print("Center of Gravity Test:")
    print(f"Input: {data}")
    print(f"Result: {result}")
    print(f"All finite where defined: {np.all(np.isfinite(result) | np.isnan(result))}")
    print()


def test_pfe():
    """Test Polarized Fractal Efficiency function"""
    data = [100, 105, 102, 108, 106, 111, 109, 114, 112, 117]
    result = pfe(data, period=5)

    print("PFE Test:")
    print(f"Input: {data}")
    print(f"Result: {result}")
    # PFE can be positive or negative
    valid_range = np.all((result >= -100) | np.isnan(result)) and np.all((result <= 100) | np.isnan(result))
    print(f"Valid range (-100 to 100): {valid_range}")
    print()


def test_doji():
    """Test Doji Candlestick Pattern"""
    open_prices = [100, 102, 98, 99, 101]
    close = [100.01, 100, 98.01, 99, 100.9]  # Very small differences for doji

    result = doji(open_prices, close)

    print("Doji Pattern Test:")
    print(f"Open: {open_prices}")
    print(f"Close: {close}")
    print(f"Doji pattern (1=doji, 0=not): {result}")
    print(f"Pattern detection works: {isinstance(result, np.ndarray)}")
    print()


def test_engulfing_bullish():
    """Test Bullish Engulfing Pattern"""
    open_prices = [105, 100, 98, 95, 97]
    close = [100, 105, 95, 100, 93]
    high = [106, 106, 99, 101, 98]
    low = [99, 99, 94, 92, 92]

    result = engulfing_bullish(open_prices, high, low, close)

    print("Bullish Engulfing Test:")
    print(f"Pattern detected: {result}")
    # Should detect the bullish engulfing where previous candle is down, current is up and engulfs
    print()


if __name__ == "__main__":
    run_all_tests()
