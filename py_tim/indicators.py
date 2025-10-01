"""
Technical Analysis Indicators
Implementation of various technical analysis indicators

Type Hints: Full typing annotations for better code quality and IDE support
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List


def _verify_data(data: Union[list, np.ndarray, pd.Series], min_length: int = 1) -> np.ndarray:
    """
    Verify and convert input data to numpy array
    """
    if isinstance(data, pd.Series):
        data = data.values
    elif isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError("Data must be a list, numpy array, or pandas Series")
    
    if len(data) < min_length:
        raise ValueError(f"Data length must be at least {min_length}")
    
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Data must contain numeric values only")

    return data


def sma(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Simple Moving Average - Vectorized Implementation

    Args:
        data: Input data (list, numpy array, or pandas Series)
        period: Period for the moving average

    Returns:
        numpy array with SMA values
    """
    data = _verify_data(data, period)
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    # Vectorized rolling window calculation using stride tricks
    # This is significantly faster than manual loops
    return np.concatenate([
        np.full(period - 1, np.nan),  # Initial NaN values
        np.convolve(data, np.ones(period), mode='valid') / period
    ])


def ema(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Exponential Moving Average - Optimized vectorized implementation
    
    Args:
        data: Input data (list, numpy array, or pandas Series)
        period: Period for the EMA
        
    Returns:
        numpy array with EMA values
    """
    data = _verify_data(data, period)
    if period <= 0:
        raise ValueError("Period must be greater than 0")
    
    alpha = 2.0 / (period + 1.0)
    ema_values = np.full_like(data, np.nan, dtype=float)
    
    # Use SMA for initial value to reduce warm-up period
    ema_values[period - 1] = np.mean(data[:period])
    
    # Vectorized calculation using cumulative operations where possible
    for i in range(period, len(data)):
        ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i - 1]
    
    return ema_values


def wma(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Weighted Moving Average
    
    Args:
        data: Input data (list, numpy array, or pandas Series)
        period: Period for the WMA
        
    Returns:
        numpy array with WMA values
    """
    data = _verify_data(data, period)
    if period <= 0:
        raise ValueError("Period must be greater than 0")
    
    wma_values = np.full_like(data, np.nan, dtype=float)
    
    # Create weights: [1, 2, 3, ..., period]
    weights = np.arange(1, period + 1, dtype=float)
    sum_weights = np.sum(weights)
    
    for i in range(period - 1, len(data)):
        subset = data[i - period + 1:i + 1]
        wma_values[i] = np.sum(subset * weights) / sum_weights
    
    return wma_values


def rsi(data: Union[list, np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
    """
    Relative Strength Index - Optimized implementation
    
    Args:
        data: Input data (list, numpy array, or pandas Series)
        period: Period for the RSI (default 14)
        
    Returns:
        numpy array with RSI values
    """
    data = _verify_data(data, period + 1)
    if period <= 0:
        raise ValueError("Period must be greater than 0")
    
    # Vectorized price changes calculation
    changes = np.diff(data, prepend=data[0])
    rsi_values = np.full_like(data, np.nan, dtype=float)
    
    # Vectorized gains and losses
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    
    # Use EMA-style smoothing for better performance
    alpha = 1.0 / period
    
    # Initialize first values
    avg_gain = np.mean(gains[1:period+1])  # Skip first element (always 0)
    avg_loss = np.mean(losses[1:period+1])
    
    # Calculate RSI for the first period
    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi_values[period] = 100 - (100 / (1 + rs))
    else:
        rsi_values[period] = 100
    
    # Optimized loop for remaining values
    for i in range(period + 1, len(data)):
        # EMA-style updating
        avg_gain = (1 - alpha) * avg_gain + alpha * gains[i]
        avg_loss = (1 - alpha) * avg_loss + alpha * losses[i]
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))
        else:
            rsi_values[i] = 100
    
    return rsi_values


def macd(
    data: Union[list, np.ndarray, pd.Series], 
    fastperiod: int = 12, 
    slowperiod: int = 26, 
    signalperiod: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Moving Average Convergence Divergence
    
    Args:
        data: Input data (list, numpy array, or pandas Series)
        fastperiod: Fast EMA period (default 12)
        slowperiod: Slow EMA period (default 26)
        signalperiod: Signal line period (default 9)
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    data = _verify_data(data, max(fastperiod, slowperiod, signalperiod) + 1)
    
    if fastperiod <= 0 or slowperiod <= 0 or signalperiod <= 0:
        raise ValueError("All periods must be greater than 0")
    
    if fastperiod >= slowperiod:
        raise ValueError("Fast period must be less than slow period")
    
    # Calculate fast and slow EMAs
    fast_ema = ema(data, fastperiod)
    slow_ema = ema(data, slowperiod)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = ema(macd_line, signalperiod)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bollinger_bands(
    data: Union[list, np.ndarray, pd.Series], 
    period: int = 20, 
    nbdevup: float = 2.0, 
    nbdevdn: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands - Optimized vectorized implementation
    
    Args:
        data: Input data (list, numpy array, or pandas Series)
        period: Period for the middle band (default 20)
        nbdevup: Number of standard deviations for upper band (default 2.0)
        nbdevdn: Number of standard deviations for lower band (default 2.0)
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    data = _verify_data(data, period)
    
    if period <= 0:
        raise ValueError("Period must be greater than 0")
    
    if nbdevup < 0 or nbdevdn < 0:
        raise ValueError("Number of standard deviations must be non-negative")
    
    # Calculate middle band (SMA)
    middle_band = sma(data, period)
    
    # Vectorized rolling standard deviation calculation
    std_values = np.full_like(data, np.nan, dtype=float)
    
    # Calculate rolling standard deviation using numpy operations
    for i in range(period - 1, len(data)):
        subset = data[i - period + 1:i + 1]
        std_values[i] = np.std(subset, ddof=0)
    
    # Vectorized band calculations
    upper_band = middle_band + (nbdevup * std_values)
    lower_band = middle_band - (nbdevdn * std_values)
    
    return upper_band, middle_band, lower_band


def atr(
    high: Union[list, np.ndarray, pd.Series], 
    low: Union[list, np.ndarray, pd.Series], 
    close: Union[list, np.ndarray, pd.Series], 
    period: int = 14
) -> np.ndarray:
    """
    Average True Range - Optimized vectorized implementation
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for the ATR (default 14)
        
    Returns:
        numpy array with ATR values
    """
    high = _verify_data(high, period + 1)
    low = _verify_data(low, period + 1)
    close = _verify_data(close, period + 1)
    
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("All input arrays must have the same length")
    
    if period <= 0:
        raise ValueError("Period must be greater than 0")
    
    # Vectorized True Range calculation
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # Handle first element
    
    # Calculate all three components vectorized
    hl = high - low
    hc = np.abs(high - prev_close)
    lc = np.abs(low - prev_close)
    
    # True Range is the maximum of the three components
    tr_values = np.maximum(hl, np.maximum(hc, lc))
    tr_values[0] = hl[0]  # First value is just high-low
    
    # Calculate ATR using exponential moving average approach
    atr_values = np.full_like(close, np.nan, dtype=float)
    
    # Initialize first ATR value
    atr_values[period - 1] = np.mean(tr_values[:period])
    
    # Use EMA-style calculation for efficiency
    alpha = 1.0 / period
    for i in range(period, len(close)):
        atr_values[i] = (1 - alpha) * atr_values[i-1] + alpha * tr_values[i]
    
    return atr_values


def obv(
    close: Union[list, np.ndarray, pd.Series], 
    volume: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    On Balance Volume
    
    Args:
        close: Close prices
        volume: Volume data
        
    Returns:
        numpy array with OBV values
    """
    close = _verify_data(close)
    volume = _verify_data(volume)
    
    if len(close) != len(volume):
        raise ValueError("Close and volume arrays must have the same length")
    
    obv_values = np.full_like(close, np.nan, dtype=float)
    
    # Initialize first OBV value
    obv_values[0] = volume[0]
    
    # Calculate OBV for the rest of the data
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            # Price went up, add volume
            obv_values[i] = obv_values[i-1] + volume[i]
        elif close[i] < close[i-1]:
            # Price went down, subtract volume
            obv_values[i] = obv_values[i-1] - volume[i]
        else:
            # Price unchanged, OBV remains the same
            obv_values[i] = obv_values[i-1]
    
    return obv_values


def stoch(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series],
    fastk_period: int = 14,
    slowk_period: int = 1,
    slowd_period: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        fastk_period: Period for %K (default 14)
        slowk_period: Smoothing period for %K to get slow %K (default 1, often 3)
        slowd_period: Period for %D (SMA of slow %K) (default 3)

    Returns:
        Tuple of (slowk, slowd) - often referred to as %K and %D
    """
    high = _verify_data(high, fastk_period)
    low = _verify_data(low, fastk_period)
    close = _verify_data(close, fastk_period)

    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("All input arrays must have the same length")

    if fastk_period <= 0 or slowk_period <= 0 or slowd_period <= 0:
        raise ValueError("All periods must be greater than 0")

    # Calculate %K (fast %K)
    fastk = np.full_like(close, np.nan, dtype=float)
    for i in range(fastk_period - 1, len(close)):
        h = np.max(high[i - fastk_period + 1:i + 1])
        l = np.min(low[i - fastk_period + 1:i + 1])
        c = close[i]
        if h != l:
            fastk[i] = ((c - l) / (h - l)) * 100
        else:
            fastk[i] = 50  # Neutral when no price movement

    # Slow %K: Simple Moving Average of fast %K (often slowk_period=1 means no smoothing)
    slowk = sma(fastk, slowk_period)

    # Slow %D: Simple Moving Average of slow %K
    slowd = sma(slowk, slowd_period)

    return slowk, slowd


def cci(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Commodity Channel Index

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for the CCI (default 14)

    Returns:
        numpy array with CCI values
    """
    high = _verify_data(high, period)
    low = _verify_data(low, period)
    close = _verify_data(close, period)

    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("All input arrays must have the same length")

    if period <= 0:
        raise ValueError("Period must be greater than 0")

    # Calculate Typical Price
    tp = (high + low + close) / 3.0

    cci_values = np.full_like(tp, np.nan, dtype=float)

    for i in range(period - 1, len(tp)):
        subset = tp[i - period + 1:i + 1]
        mean_tp = np.mean(subset)
        mad = np.mean(np.abs(subset - mean_tp))
        if mad != 0:
            cci_values[i] = (tp[i] - mean_tp) / (0.015 * mad)

    return cci_values


def dema(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Double Exponential Moving Average

    Args:
        data: Input data
        period: Period for the DEMA

    Returns:
        numpy array with DEMA values
    """
    data = _verify_data(data, period)
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    ema1 = ema(data, period)
    ema_of_ema1 = ema(ema1, period)
    dema_values = 2 * ema1 - ema_of_ema1

    return dema_values


def tema(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Triple Exponential Moving Average

    Args:
        data: Input data
        period: Period for the TEMA

    Returns:
        numpy array with TEMA values
    """
    data = _verify_data(data, period)
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    ema1 = ema(data, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    tema_values = 3 * ema1 - 3 * ema2 + ema3

    return tema_values


def trix(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Triple Exponential Average

    Args:
        data: Input data
        period: Period for the TRIX

    Returns:
        numpy array with TRIX values
    """
    data = _verify_data(data, period)
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    ema1 = ema(data, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)

    trix_values = np.full_like(ema3, np.nan)
    trix_values[1:] = 100 * (ema3[1:] / ema3[:-1] - 1)

    return trix_values


def willr(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Williams %R

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for the %R

    Returns:
        numpy array with %R values
    """
    high = _verify_data(high, period)
    low = _verify_data(low, period)
    close = _verify_data(close, period)
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("All input arrays must have the same length")
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    willr_values = np.full_like(close, np.nan, dtype=float)
    for i in range(period - 1, len(close)):
        h = np.max(high[i - period + 1:i + 1])
        l = np.min(low[i - period + 1:i + 1])
        c = close[i]
        denominator = h - l
        if abs(denominator) > 1e-10:
            willr_values[i] = -100 * (h - c) / denominator
        else:
            willr_values[i] = -50  # Neutral when no range

    return willr_values


def cmo(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Chande Momentum Oscillator

    Args:
        data: Input data
        period: Period for the CMO

    Returns:
        numpy array with CMO values
    """
    data = _verify_data(data, period + 1)
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    changes = np.diff(data)
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)

    cmo_values = np.full_like(data, np.nan, dtype=float)
    for i in range(period, len(data)):
        su = np.sum(gains[i - period:i])
        sd = np.sum(losses[i - period:i])
        denominator = su + sd
        if abs(denominator) > 1e-10:
            cmo_values[i] = 100 * (su - sd) / denominator
        else:
            cmo_values[i] = 0  # Neutral when no momentum

    return cmo_values


def ultosc(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Ultimate Oscillator

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with Ultimate Oscillator values
    """
    high = _verify_data(high, 29)
    low = _verify_data(low, 29)
    close = _verify_data(close, 29)
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("All input arrays must have the same length")

    # Buying Pressure BP
    bp = np.full_like(close, np.nan)
    for i in range(1, len(close)):
        bp[i] = close[i] - np.maximum(low[i-1], close[i-1])

    # True Range TR
    tr_values = np.full_like(close, np.nan)
    for i in range(1, len(close)):
        tr_values[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    # Averages over 7,14,28
    bp7 = sma(bp, 7) / sma(tr_values, 7)
    bp14 = sma(bp, 14) / sma(tr_values, 14)
    bp28 = sma(bp, 28) / sma(tr_values, 28)

    ultosc_values = 100 * (4 * bp7 + 2 * bp14 + bp28) / 7

    return ultosc_values


def adx(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series],
    period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average Directional Index, +DI, -DI

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for the ADX

    Returns:
        Tuple of (adx, plus_di, minus_di)
    """
    high = _verify_data(high, period + 1)
    low = _verify_data(low, period + 1)
    close = _verify_data(close, period + 1)
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("All input arrays must have the same length")
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    # Directional Movement
    dm_plus = np.zeros_like(high)
    dm_minus = np.zeros_like(high)
    for i in range(1, len(high)):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > 0 and high_diff > low_diff:
            dm_plus[i] = high_diff
        if low_diff > 0 and low_diff > high_diff:
            dm_minus[i] = low_diff

    tr_values = np.full_like(close, np.nan)
    for i in range(1, len(close)):
        tr_values[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    plus_di = 100 * ema(dm_plus, period) / ema(tr_values, period)
    minus_di = 100 * ema(dm_minus, period) / ema(tr_values, period)

    dx = np.full_like(plus_di, np.nan)
    ax = np.logical_and(plus_di + minus_di != 0, np.logical_not(np.isnan(plus_di + minus_di)))
    dx[ax] = 100 * np.abs(plus_di[ax] - minus_di[ax]) / (plus_di[ax] + minus_di[ax])
    adx_values = ema(dx, period)

    return adx_values, plus_di, minus_di


def kama(data: Union[list, np.ndarray, pd.Series], period: int = 30) -> np.ndarray:
    """
    Kaufman's Adaptive Moving Average

    Args:
        data: Input data
        period: Period for the KAMA (default 30)

    Returns:
        numpy array with KAMA values
    """
    data = _verify_data(data, period + 1)
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    kama_values = np.full_like(data, np.nan, dtype=float)

    # Initialize with first available average
    kama_values[period - 1] = np.mean(data[:period])

    fast = 2  # Default
    slow = 30  # Default

    for i in range(period, len(data)):
        # Efficiency Ratio
        change = abs(data[i] - data[i - period])
        vol = np.sum(np.abs(np.diff(data[i - period:i + 1])))  # Volatility sum of absolute changes
        er = change / vol if vol > 1e-10 else 0

        # Smoothing constant
        sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2

        # KAMA update
        kama_values[i] = kama_values[i-1] + sc * (data[i] - kama_values[i-1])

    return kama_values


def hma(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Hull Moving Average

    Args:
        data: Input data
        period: Period for the HMA

    Returns:
        numpy array with HMA values
    """
    data = _verify_data(data, int(2 * period))
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    half_period = period // 2
    sqrt_period = int(np.sqrt(period))

    wma_half = wma(data, half_period)
    wma_full = wma(data, period)

    raw_hma = 2 * wma_half - wma_full

    # WMA of the raw HMA with sqrt(period)
    hma_values = wma(raw_hma, sqrt_period)

    return hma_values


def mom(data: Union[list, np.ndarray, pd.Series], period: int = 10) -> np.ndarray:
    """
    Momentum

    Args:
        data: Input data (typically close prices)
        period: Period for the momentum calculation (default 10)

    Returns:
        numpy array with momentum values (current value - value 'period' periods ago)
    """
    data = _verify_data(data, period + 1)
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    mom_values = np.full_like(data, np.nan, dtype=float)
    for i in range(period, len(data)):
        mom_values[i] = data[i] - data[i - period]

    return mom_values


def roc(data: Union[list, np.ndarray, pd.Series], period: int = 10) -> np.ndarray:
    """
    Rate of Change

    Args:
        data: Input data (typically close prices)
        period: Period for the ROC calculation (default 10)

    Returns:
        numpy array with ROC values as percentage ((current / past - 1) * 100)
    """
    data = _verify_data(data, period + 1)
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    roc_values = np.full_like(data, np.nan, dtype=float)
    for i in range(period, len(data)):
        if data[i - period] != 0:
            roc_values[i] = ((data[i] / data[i - period]) - 1) * 100
        else:
            roc_values[i] = np.nan  # Avoid division by zero

    return roc_values


def rocp(data: Union[list, np.ndarray, pd.Series], period: int = 10) -> np.ndarray:
    """
    Rate of Change Percentage

    Args:
        data: Input data
        period: Period for the ROCP calculation

    Returns:
        numpy array with ROCP values ((current / past) - 1)
    """
    data = _verify_data(data, period + 1)
    if period <= 0:
        raise ValueError("Period must be greater than 0")

    rocp_values = np.full_like(data, np.nan, dtype=float)
    for i in range(period, len(data)):
        if data[i - period] != 0:
            rocp_values[i] = (data[i] / data[i - period]) - 1
        else:
            rocp_values[i] = np.nan  # Avoid division by zero

    return rocp_values


def bop(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Balance Of Power

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with BOP values ((close - open) / (high - low))
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    bop_values = np.full_like(close, np.nan, dtype=float)
    for i in range(len(close)):
        denominator = high[i] - low[i]
        if abs(denominator) > 1e-10:  # Avoid division by zero or very small numbers
            bop_values[i] = (close[i] - open_prices[i]) / denominator
        else:
            bop_values[i] = 0  # Neutral when no range

    return bop_values


def stochrsi(
    data: Union[list, np.ndarray, pd.Series],
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic RSI

    Args:
        data: Input price data
        rsi_period: Period for RSI calculation (default 14)
        stoch_period: Period for Stochastic calculation on RSI (default 14)
        k_period: Period for %K smoothing (default 3)
        d_period: Period for %D smoothing (default 3)

    Returns:
        Tuple of (stoch_k, stoch_d) - Stochastic %K and %D applied to RSI
    """
    data = _verify_data(data, rsi_period + stoch_period + 1)

    # First calculate RSI
    rsi_values = rsi(data, rsi_period)

    # Apply stochastic oscillator to RSI values
    stoch_k, stoch_d = stoch(rsi_values, rsi_values, rsi_values, stoch_period, k_period, d_period)

    return stoch_k, stoch_d


def tsf(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Time Series Forecast (Linear Regression Forecast)

    Args:
        data: Input data
        period: Period for linear regression

    Returns:
        numpy array with TSF values (forecast for next period)
    """
    data = _verify_data(data, period)
    if period <= 1:
        raise ValueError("Period must be greater than 1")

    tsf_values = np.full_like(data, np.nan, dtype=float)

    for i in range(period - 1, len(data)):
        # Get the slice of data for this period
        y = data[i - period + 1:i + 1]

        # Create x values (0 to period-1)
        x = np.arange(period)

        # Calculate linear regression coefficients
        slope = np.cov(x, y)[0, 1] / np.var(x)
        intercept = np.mean(y) - slope * np.mean(x)

        # Forecast next value (x = period)
        tsf_values[i] = intercept + slope * period

    return tsf_values


def roc100(data: Union[list, np.ndarray, pd.Series], period: int = 10) -> np.ndarray:
    """
    Rate of Change Ratio 100 Scale

    Args:
        data: Input data
        period: Period for the ROC calculation

    Returns:
        numpy array with ROC values multiplied by 100 for better scaling
    """
    return roc(data, period)


def correl(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    period: int = 30
) -> np.ndarray:
    """
    Pearson's Correlation Coefficient between High and Low prices

    Args:
        high: High prices
        low: Low prices
        period: Period for correlation calculation

    Returns:
        numpy array with correlation coefficients
    """
    high = _verify_data(high, period)
    low = _verify_data(low, period)

    if len(high) != len(low):
        raise ValueError("High and low arrays must have the same length")

    correl_values = np.full_like(high, np.nan, dtype=float)

    for i in range(period - 1, len(high)):
        high_slice = high[i - period + 1:i + 1]
        low_slice = low[i - period + 1:i + 1]
        correl_values[i] = np.corrcoef(high_slice, low_slice)[0, 1]

    return correl_values


def mfi(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series],
    volume: Union[list, np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Money Flow Index

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        period: Period for the MFI calculation (default 14)

    Returns:
        numpy array with MFI values (0-100 scale)
    """
    high = _verify_data(high, period)
    low = _verify_data(low, period)
    close = _verify_data(close, period)
    volume = _verify_data(volume, period)

    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("All input arrays must have the same length")

    # Calculate Typical Price
    tp = (high + low + close) / 3.0

    # Calculate Money Flow
    mf = tp * volume

    mfi_values = np.full_like(tp, np.nan, dtype=float)

    for i in range(period, len(tp)):
        # Get the period's money flow values
        pos_mf = 0  # Positive money flow
        neg_mf = 0  # Negative money flow

        for j in range(i - period + 1, i + 1):
            if tp[j] > tp[j - 1]:
                pos_mf += mf[j]
            elif tp[j] < tp[j - 1]:
                neg_mf += mf[j]
            # If tp[j] == tp[j - 1], it's neutral and ignored

        # Calculate Money Flow Ratio
        if neg_mf != 0:
            mfr = pos_mf / neg_mf
            mfi_values[i] = 100 - (100 / (1 + mfr))
        else:
            mfi_values[i] = 100  # All positive flow

    return mfi_values


def cmf(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series],
    volume: Union[list, np.ndarray, pd.Series],
    period: int = 21
) -> np.ndarray:
    """
    Chaikin Money Flow

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        period: Period for the CMF calculation (default 21)

    Returns:
        numpy array with CMF values
    """
    high = _verify_data(high, period)
    low = _verify_data(low, period)
    close = _verify_data(close, period)
    volume = _verify_data(volume, period)

    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("All input arrays must have the same length")

    # Calculate Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where(np.isnan(mfm), 0, mfm)  # Handle division by zero

    # Calculate Money Flow Volume
    mfv = mfm * volume

    # Calculate Chaikin Money Flow (SMA of MFV / SMA of Volume)
    cmf_values = np.full_like(close, np.nan, dtype=float)

    for i in range(period - 1, len(close)):
        sum_mfv = np.sum(mfv[i - period + 1:i + 1])
        sum_volume = np.sum(volume[i - period + 1:i + 1])

        if abs(sum_volume) > 1e-10:  # Avoid division by zero or very small numbers
            cmf_values[i] = sum_mfv / sum_volume

    return cmf_values


def vwap(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series],
    volume: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Volume Weighted Average Price

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data

    Returns:
        numpy array with VWAP values
    """
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)
    volume = _verify_data(volume)

    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("All input arrays must have the same length")

    # Calculate Typical Price
    tp = (high + low + close) / 3.0

    # Calculate cumulative volume and cumulative volume-weighted price
    cum_vol = np.cumsum(volume)
    cum_vol_price = np.cumsum(tp * volume)

    vwap_values = np.full_like(tp, np.nan, dtype=float)

    # VWAP is cumulative by nature (typically reset daily)
    # For simplicity, we'll calculate cumulative VWAP
    for i in range(len(tp)):
        if cum_vol[i] > 0:
            vwap_values[i] = cum_vol_price[i] / cum_vol[i]

    return vwap_values


def force(
    close: Union[list, np.ndarray, pd.Series],
    volume: Union[list, np.ndarray, pd.Series],
    period: int = 13
) -> np.ndarray:
    """
    Force Index

    Args:
        close: Close prices
        volume: Volume data
        period: Period for EMA smoothing (default 13)

    Returns:
        numpy array with Force Index values
    """
    close = _verify_data(close)
    volume = _verify_data(volume)

    if len(close) != len(volume):
        raise ValueError("Close and volume arrays must have the same length")

    # Calculate force index (volume * (close - previous close))
    force_raw = np.full_like(close, np.nan, dtype=float)
    force_raw[1:] = volume[1:] * (close[1:] - close[:-1])

    # Apply EMA smoothing
    force_values = ema(force_raw, period)

    return force_values


def bollinger_percent_b(
    data: Union[list, np.ndarray, pd.Series],
    period: int = 20,
    nbdev: float = 2.0
) -> np.ndarray:
    """
    Bollinger %B (Position within Bollinger Bands)

    Args:
        data: Input price data (typically close)
        period: Period for moving average
        nbdev: Number of standard deviations

    Returns:
        numpy array with %B values (0-1 scale, where 0=touching lower band, 1=touching upper band)
    """
    upper, middle, lower = bollinger_bands(data, period, nbdev, nbdev)

    # Calculate %B: (price - lower) / (upper - lower)
    percent_b = np.full_like(data, np.nan, dtype=float)

    for i in range(len(data)):
        if not np.isnan(upper[i]) and not np.isnan(lower[i]) and upper[i] != lower[i]:
            percent_b[i] = (data[i] - lower[i]) / (upper[i] - lower[i])

    return percent_b


def bollinger_bandwidth(
    data: Union[list, np.ndarray, pd.Series],
    period: int = 20,
    nbdev: float = 2.0
) -> np.ndarray:
    """
    Bollinger Bandwidth

    Args:
        data: Input price data (typically close)
        period: Period for moving average
        nbdev: Number of standard deviations

    Returns:
        numpy array with bandwidth values ((upper - lower) / middle)
    """
    upper, middle, lower = bollinger_bands(data, period, nbdev, nbdev)

    # Calculate bandwidth: (upper - lower) / middle
    bandwidth = np.full_like(data, np.nan, dtype=float)

    for i in range(len(data)):
        if not np.isnan(upper[i]) and not np.isnan(lower[i]) and not np.isnan(middle[i]) and middle[i] != 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]

    return bandwidth


def mass_index(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    ema_period: int = 9,
    sum_period: int = 25
) -> np.ndarray:
    """
    Mass Index

    Args:
        high: High prices
        low: Low prices
        ema_period: Period for EMA calculation (default 9)
        sum_period: Period for the sum of ranges (default 25)

    Returns:
        numpy array with Mass Index values
    """
    high = _verify_data(high)
    low = _verify_data(low)

    if len(high) != len(low):
        raise ValueError("High and low arrays must have the same length")

    # Calculate single EMA of high-low range
    hl_range = high - low
    ema_range = ema(hl_range, ema_period)

    # Calculate double EMA of high-low range
    ema_ema_range = ema(ema_range, ema_period)

    # Calculate ratio
    ratio = np.where(ema_range != 0, ema_range / ema_ema_range, 1.0)

    # Sum the ratio over the specified period
    mass_values = np.full_like(ratio, np.nan, dtype=float)

    for i in range(sum_period - 1, len(ratio)):
        mass_values[i] = np.sum(ratio[i - sum_period + 1:i + 1])

    return mass_values


def elder_force_index(close: Union[list, np.ndarray, pd.Series], volume: Union[list, np.ndarray, pd.Series], period: int = 13) -> np.ndarray:
    """
    Elder Force Index

    Args:
        close: Close prices
        volume: Volume data
        period: Period for EMA smoothing (default 13)

    Returns:
        numpy array with Elder Force Index values
    """
    close = _verify_data(close)
    volume = _verify_data(volume)

    if len(close) != len(volume):
        raise ValueError("Close and volume arrays must have the same length")

    # Force Index = Volume * (Close - Previous Close)
    force_raw = np.full_like(close, np.nan, dtype=float)
    force_raw[1:] = volume[1:] * (close[1:] - close[:-1])

    # Apply EMA smoothing (same as regular Force Index)
    efi_values = ema(force_raw, period)

    return efi_values


def elder_ray_index(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series],
    period: int = 13
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Elder Ray Index (Bull Power and Bear Power)

    Args:
        high: High prices
        low: Low prices
        close: Close prices (used for EMA calculation)
        period: Period for EMA calculation (default 13)

    Returns:
        Tuple of (bull_power, bear_power) where:
        - bull_power: High - EMA(Close)
        - bear_power: Low - EMA(Close)
    """
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("All input arrays must have the same length")

    # Calculate EMA of close
    ema_close = ema(close, period)

    # Calculate Bull Power (High - EMA)
    bull_power = np.full_like(high, np.nan, dtype=float)
    for i in range(len(high)):
        if not np.isnan(ema_close[i]):
            bull_power[i] = high[i] - ema_close[i]

    # Calculate Bear Power (Low - EMA)
    bear_power = np.full_like(low, np.nan, dtype=float)
    for i in range(len(low)):
        if not np.isnan(ema_close[i]):
            bear_power[i] = low[i] - ema_close[i]

    return bull_power, bear_power


def schaff_trend_cycle(
    data: Union[list, np.ndarray, pd.Series],
    fast_length: int = 23,
    slow_length: int = 50,
    k_period: int = 10,
    d_period: int = 3
) -> np.ndarray:
    """
    Schaff Trend Cycle (STC) - a momentum oscillator that cycles between 0-100

    Args:
        data: Input price data (typically close)
        fast_length: Fast cycle length (default 23)
        slow_length: Slow cycle length (default 50)
        k_period: Stochastic K period (default 10)
        d_period: Stochastic D period (default 3)

    Returns:
        numpy array with STC values (0-100 scale)
    """
    data = _verify_data(data, max(fast_length, slow_length) + k_period + d_period)

    # Step 1: Calculate MACD (fast cycle - slow cycle)
    # Using fast_length and slow_length as EMA periods for MACD calculation
    fast_ema = ema(data, fast_length)
    slow_ema = ema(data, slow_length)
    macd_cycle = fast_ema - slow_ema

    # Step 2: Calculate stochastic on the MACD cycle
    # Find highest/lowest over k_period of the MACD
    k_values = np.full_like(macd_cycle, np.nan, dtype=float)

    for i in range(k_period - 1, len(macd_cycle)):
        macd_window = macd_cycle[i - k_period + 1:i + 1]
        highest = np.max(macd_window)
        lowest = np.min(macd_window)
        current = macd_cycle[i]

        if highest != lowest:
            k_values[i] = 100 * (current - lowest) / (highest - lowest)

    # Step 3: Smooth K with EMA (stochastic D component)
    d_values = ema(k_values, d_period)

    # Step 4: Apply another stochastic calculation to smooth further
    # Find highest/lowest over k_period of the smoothed values
    stc_values = np.full_like(d_values, np.nan, dtype=float)

    for i in range(k_period - 1, len(d_values)):
        d_window = d_values[i - k_period + 1:i + 1]
        highest = np.max(d_window)
        lowest = np.min(d_window)
        current = d_values[i]

        if highest != lowest:
            stc_values[i] = 100 * (current - lowest) / (highest - lowest)

    return stc_values


def random_walk_index(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series],
    period: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random Walk Index (RWI) - measures trend strength vs random walk

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for calculation (default 10)

    Returns:
        Tuple of (rwi_high, rwi_low) where:
        - rwi_high: Bullish random walk component
        - rwi_low: Bearish random walk component
    """
    high = _verify_data(high, period + 1)
    low = _verify_data(low, period + 1)
    close = _verify_data(close, period + 1)

    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("All input arrays must have the same length")

    # Calculate True Range over the period for normalization
    tr_values = np.full_like(close, np.nan, dtype=float)
    for i in range(1, len(close)):
        tr_values[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    rwi_high = np.full_like(close, np.nan, dtype=float)
    rwi_low = np.full_like(close, np.nan, dtype=float)

    for i in range(period, len(close)):
        # Get period start price
        period_start_price = close[i - period]

        # Calculate max true range over the period
        max_tr = np.max(tr_values[i - period + 1:i + 1])

        if max_tr > 0:
            # RWI High: measures upward trend strength relative to random walk
            rwi_high[i] = period * (close[i] - period_start_price) / max_tr

            # RWI Low: measures downward trend strength relative to random walk
            rwi_low[i] = period * (period_start_price - close[i]) / max_tr

    return rwi_high, rwi_low


def chaikin_ad(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series],
    volume: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Chaikin Accumulation/Distribution Line (AD)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data

    Returns:
        numpy array with Chaikin AD values (accumulated volume)
    """
    high = _verify_data(high, 2)  # Need at least 2 periods for AD calculation
    low = _verify_data(low, 2)
    close = _verify_data(close, 2)
    volume = _verify_data(volume, 2)

    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("All input arrays must have the same length")

    ad_values = np.full_like(close, np.nan, dtype=float)

    # Start with the first period's volume
    if len(close) > 0:
        ad_values[0] = volume[0]

    # Calculate Money Flow Multiplier (MFM) and Accumulation/Distribution (AD)
    for i in range(1, len(close)):
        # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
        price_range = high[i] - low[i]

        if abs(price_range) > 1e-10:  # Avoid division by zero or very small numbers
            mfm = ((close[i] - low[i]) - (high[i] - close[i])) / price_range
        else:
            mfm = 0  # Neutral when no price range

        # Money Flow Volume = MFM * Volume
        mfv = mfm * volume[i]

        # Accumulation/Distribution = Previous AD + MFV
        ad_values[i] = ad_values[i-1] + mfv

    return ad_values


def ease_of_movement(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    volume: Union[list, np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Ease of Movement (EMV) - measures the "ease" with which price moves

    Args:
        high: High prices
        low: Low prices
        volume: Volume data
        period: Period for EMA smoothing (default 14)

    Returns:
        numpy array with EMV values
    """
    high = _verify_data(high, 2)  # Need at least 2 periods for EMV calculation
    low = _verify_data(low, 2)
    volume = _verify_data(volume, 2)

    if len(high) != len(low) or len(high) != len(volume):
        raise ValueError("All input arrays must have the same length")

    emv_values = np.full_like(high, np.nan, dtype=float)

    # Calculate midpoint move = (High + Low)/2 - (High_prev + Low_prev)/2
    midpoints = (high + low) / 2
    midpoint_move = np.full_like(midpoints, np.nan)
    midpoint_move[1:] = midpoints[1:] - midpoints[:-1]

    # Calculate box ratio = Volume / (High - Low) * constant
    # Traditionally, EMV uses a constant of 1000000 for scaling
    hl_range = high - low
    box_ratio = np.full_like(hl_range, np.nan)

    for i in range(len(hl_range)):
        if abs(hl_range[i]) > 1e-10 and not np.isnan(volume[i]):
            box_ratio[i] = volume[i] / hl_range[i] * 1000000  # Scale for readability
        else:
            box_ratio[i] = 0

    # Calculate raw EMV = Midpoint Move / Box Ratio (when Box Ratio != 0)
    raw_emv = np.full_like(midpoint_move, np.nan)

    for i in range(len(midpoint_move)):
        if not np.isnan(midpoint_move[i]) and not np.isnan(box_ratio[i]) and abs(box_ratio[i]) > 1e-10:
            raw_emv[i] = midpoint_move[i] / box_ratio[i]
        else:
            raw_emv[i] = 0

    # Apply EMA smoothing to get final EMV
    emv_values = ema(raw_emv, period)

    return emv_values


def positive_volume_index(
    close: Union[list, np.ndarray, pd.Series],
    volume: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Positive Volume Index (PVI) - tracks cumulative price changes during increasing volume

    Args:
        close: Close prices
        volume: Volume data

    Returns:
        numpy array with PVI values (cumulative index starting at close[0])
    """
    close = _verify_data(close)
    volume = _verify_data(volume)

    if len(close) != len(volume):
        raise ValueError("Close and volume arrays must have the same length")

    pvi_values = np.full_like(close, np.nan, dtype=float)

    if len(close) > 0:
        pvi_values[0] = close[0]  # Initialize with first close price

    # Calculate PVI
    for i in range(1, len(close)):
        if volume[i] >= volume[i-1]:  # Volume increased or equal
            # Add the percentage change to the index
            if abs(close[i-1]) > 1e-10:  # Avoid division by zero
                percent_change = (close[i] - close[i-1]) / close[i-1]
                pvi_values[i] = pvi_values[i-1] * (1 + percent_change)
            else:
                pvi_values[i] = pvi_values[i-1]
        else:
            # Volume decreased, PVI stays the same
            pvi_values[i] = pvi_values[i-1]

    return pvi_values


def negative_volume_index(
    close: Union[list, np.ndarray, pd.Series],
    volume: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Negative Volume Index (NVI) - tracks cumulative price changes during decreasing volume

    Args:
        close: Close prices
        volume: Volume data

    Returns:
        numpy array with NVI values (cumulative index starting at close[0])
    """
    close = _verify_data(close)
    volume = _verify_data(volume)

    if len(close) != len(volume):
        raise ValueError("Close and volume arrays must have the same length")

    nvi_values = np.full_like(close, np.nan, dtype=float)

    if len(close) > 0:
        nvi_values[0] = close[0]  # Initialize with first close price

    # Calculate NVI - opposite of PVI
    for i in range(1, len(close)):
        if volume[i] < volume[i-1]:  # Volume decreased (note: strict less than for NVI)
            # Add the percentage change to the index on decreasing volume
            if abs(close[i-1]) > 1e-10:  # Avoid division by zero
                percent_change = (close[i] - close[i-1]) / close[i-1]
                nvi_values[i] = nvi_values[i-1] * (1 + percent_change)
            else:
                nvi_values[i] = nvi_values[i-1]
        else:
            # Volume increased or stayed same, NVI stays the same
            nvi_values[i] = nvi_values[i-1]

    return nvi_values


def price_volume_trend(
    close: Union[list, np.ndarray, pd.Series],
    volume: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Price Volume Trend (PVT) - tracks cumulative price changes weighted by volume

    Args:
        close: Close prices
        volume: Volume data

    Returns:
        numpy array with PVT values (cumulative volume-weighted price changes)
    """
    close = _verify_data(close)
    volume = _verify_data(volume)

    if len(close) != len(volume):
        raise ValueError("Close and volume arrays must have the same length")

    pvt_values = np.full_like(close, np.nan, dtype=float)

    # Initialize first value (typically starts at 0)
    pvt_values[0] = 0

    # Calculate PVT
    for i in range(1, len(close)):
        if not np.isnan(close[i-1]) and close[i-1] != 0:
            # Calculate price change percentage and multiply by volume
            price_change_pct = (close[i] - close[i-1]) / close[i-1]
            pvt_values[i] = pvt_values[i-1] + (price_change_pct * volume[i])
        else:
            pvt_values[i] = pvt_values[i-1]  # No change if invalid data

    return pvt_values


def volume_oscillator(
    volume: Union[list, np.ndarray, pd.Series],
    short_period: int = 5,
    long_period: int = 10
) -> np.ndarray:
    """
    Volume Oscillator - momentum oscillator measuring volume changes

    Args:
        volume: Volume data
        short_period: Short period for fast SMA (default 5)
        long_period: Long period for slow SMA (default 10)

    Returns:
        numpy array with Volume Oscillator values as percentage difference
    """
    volume = _verify_data(volume, long_period)

    # Calculate fast and slow SMAs of volume
    fast_sma = sma(volume, short_period)
    slow_sma = sma(volume, long_period)

    # Volume Oscillator = ((Fast SMA - Slow SMA) / Slow SMA) * 100
    vol_osc = np.full_like(volume, np.nan, dtype=float)

    for i in range(len(volume)):
        if not np.isnan(fast_sma[i]) and not np.isnan(slow_sma[i]) and slow_sma[i] != 0:
            vol_osc[i] = ((fast_sma[i] - slow_sma[i]) / slow_sma[i]) * 100

    return vol_osc


def volume_weighted_ma(
    close: Union[list, np.ndarray, pd.Series],
    volume: Union[list, np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Volume Weighted Moving Average (VWMA)

    Args:
        close: Close prices
        volume: Volume data
        period: Period for the VWMA calculation (default 14)

    Returns:
        numpy array with VWMA values (volume-weighted moving average)
    """
    close = _verify_data(close, period)
    volume = _verify_data(volume, period)

    if len(close) != len(volume):
        raise ValueError("Close and volume arrays must have the same length")

    vwma_values = np.full_like(close, np.nan, dtype=float)

    for i in range(period - 1, len(close)):
        # Get the period's data slice
        close_slice = close[i - period + 1:i + 1]
        volume_slice = volume[i - period + 1:i + 1]

        # Calculate volume-weighted price for this window
        total_volume = np.sum(volume_slice)
        if total_volume > 0:
            weighted_sum = np.sum(close_slice * volume_slice)
            vwma_values[i] = weighted_sum / total_volume

    return vwma_values


def williams_ad(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Williams Accumulation/Distribution

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with Williams A/D values
    """
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("All input arrays must have the same length")

    wad_values = np.full_like(close, np.nan, dtype=float)

    # Williams A/D starts with 0
    if len(close) > 0:
        wad_values[0] = 0

    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            # If close up, add True Range to previous A/D
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            wad_values[i] = wad_values[i-1] + tr
        elif close[i] < close[i-1]:
            # If close down, subtract True Range from previous A/D
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            wad_values[i] = wad_values[i-1] - tr
        else:
            # If close unchanged, A/D remains the same
            wad_values[i] = wad_values[i-1]

    return wad_values


def chande_forecast(
    data: Union[list, np.ndarray, pd.Series],
    period: int = 5,
    smooth_period: int = 3
) -> np.ndarray:
    """
    Chande Forecast Oscillator

    Args:
        data: Input price data
        period: Period for the forecast calculation
        smooth_period: Period for EMA smoothing

    Returns:
        numpy array with Chande Forecast Oscillator values
    """
    data = _verify_data(data, period + smooth_period)

    fc_values = np.full_like(data, np.nan, dtype=float)

    for i in range(period + smooth_period - 2, len(data)):
        # Get the data slice for calculation
        slice_data = data[i - period - smooth_period + 2:i + 1]

        if len(slice_data) >= period + smooth_period - 1:
            # Calculate linear regression forecast
            y = slice_data[-(period + smooth_period - 1):]
            x = np.arange(len(y))
            slope = np.cov(x, y)[0, 1] / np.var(x)
            intercept = np.mean(y) - slope * np.mean(x)

            # Forecast next value
            forecast = intercept + slope * len(y)

            # Calculate forecast error as percentage
            if data[i] != 0:
                fc_values[i] = 100 * (forecast - data[i]) / data[i]

    return fc_values


def ravi(
    data: Union[list, np.ndarray, pd.Series],
    short_period: int = 7,
    long_period: int = 65
) -> np.ndarray:
    """
    Relative Strength Index

    Args:
        data: Input price data
        short_period: Short EMA period
        long_period: Long EMA period

    Returns:
        numpy array with RAVI values ((short_ema / long_ema - 1) * 100)
    """
    data = _verify_data(data, long_period)

    # Calculate short and long EMAs
    short_ema = ema(data, short_period)
    long_ema = ema(data, long_period)

    # RAVI = ((short EMA / long EMA) - 1) * 100
    ravi_values = np.full_like(data, np.nan, dtype=float)

    for i in range(len(data)):
        if not np.isnan(short_ema[i]) and not np.isnan(long_ema[i]) and long_ema[i] != 0:
            ravi_values[i] = ((short_ema[i] / long_ema[i]) - 1) * 100

    return ravi_values


def tsi(
    data: Union[list, np.ndarray, pd.Series],
    short_period: int = 25,
    long_period: int = 13
) -> np.ndarray:
    """
    True Strength Index

    Args:
        data: Input price data
        short_period: Short EMA period (default 25)
        long_period: Long EMA period (default 13)

    Returns:
        numpy array with TSI values
    """
    data = _verify_data(data, short_period + long_period)

    # Calculate momentum (price change)
    momentum = np.full_like(data, np.nan, dtype=float)
    momentum[1:] = data[1:] - data[:-1]

    # Calculate absolute momentum
    abs_momentum = np.abs(momentum)

    # Double smooth momentum and absolute momentum
    ema_short_momentum = ema(momentum, short_period)
    ema_long_momentum = ema(ema_short_momentum, long_period)

    ema_short_abs_momentum = ema(abs_momentum, short_period)
    ema_long_abs_momentum = ema(ema_short_abs_momentum, long_period)

    # TSI = 100 * (double smoothed momentum / double smoothed absolute momentum)
    tsi_values = np.full_like(data, np.nan, dtype=float)

    for i in range(len(data)):
        if (not np.isnan(ema_long_momentum[i]) and not np.isnan(ema_long_abs_momentum[i])
            and ema_long_abs_momentum[i] != 0):
            tsi_values[i] = 100 * (ema_long_momentum[i] / ema_long_abs_momentum[i])

    return tsi_values


def midpoint(data: Union[list, np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
    """
    Midpoint over period

    Args:
        data: Input price data
        period: Period for calculation

    Returns:
        numpy array with midpoint values ((highest + lowest) / 2 over period)
    """
    data = _verify_data(data, period)

    midpoint_values = np.full_like(data, np.nan, dtype=float)

    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]
        midpoint_values[i] = (np.max(window) + np.min(window)) / 2

    return midpoint_values


def midpoint_price(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Midpoint Price over period

    Args:
        high: High prices
        low: Low prices
        period: Period for calculation

    Returns:
        numpy array with midpoint price values
    """
    high = _verify_data(high, period)
    low = _verify_data(low, period)

    if len(high) != len(low):
        raise ValueError("High and low arrays must have the same length")

    midpoint_values = np.full_like(high, np.nan, dtype=float)

    for i in range(period - 1, len(high)):
        high_window = high[i - period + 1:i + 1]
        low_window = low[i - period + 1:i + 1]
        midpoint_values[i] = (np.max(high_window) + np.min(low_window)) / 2

    return midpoint_values


def avg_price(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Average Price ((open + high + low + close) / 4)

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with average price values
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    return (open_prices + high + low + close) / 4


def med_price(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Median Price ((high + low) / 2)

    Args:
        high: High prices
        low: Low prices

    Returns:
        numpy array with median price values
    """
    high = _verify_data(high)
    low = _verify_data(low)

    if len(high) != len(low):
        raise ValueError("High and low arrays must have the same length")

    return (high + low) / 2


def typ_price(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Typical Price ((high + low + close) / 3)

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with typical price values
    """
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("All input arrays must have the same length")

    return (high + low + close) / 3


def wcl_price(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Weighted Close Price ((high + low + close + close) / 4)

    Args:
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with weighted close price values
    """
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("All input arrays must have the same length")

    return (high + low + close + close) / 4


def doji(
    open_prices: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Doji Candlestick Pattern

    Args:
        open_prices: Open prices
        close: Close prices

    Returns:
        numpy array with 1 for doji pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    close = _verify_data(close)

    if len(open_prices) != len(close):
        raise ValueError("Open and close arrays must have the same length")

    doji_pattern = np.zeros_like(close, dtype=int)

    # Calculate body size (absolute difference between open and close)
    body_size = np.abs(close - open_prices)

    # Calculate total candle size heuristic (body_size * 0.1 for very small bodies)
    # A doji occurs when the body is very small relative to the overall range
    for i in range(len(close)):
        if body_size[i] <= np.maximum(np.abs(close[i]), np.abs(open_prices[i])) * 0.05:  # Body <= 5% of price
            doji_pattern[i] = 1

    return doji_pattern


def hammer(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Hammer Candlestick Pattern

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for hammer pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    hammer_pattern = np.zeros_like(close, dtype=int)

    for i in range(len(close)):
        body = abs(close[i] - open_prices[i])
        candle_range = high[i] - low[i]

        if abs(body) < 1e-10:  # Very small body
            continue

        # Upper shadow should be very small (1/3 of body or less)
        # Lower shadow should be at least 2x the body size
        upper_shadow = high[i] - max(open_prices[i], close[i])
        lower_shadow = min(open_prices[i], close[i]) - low[i]

        if (upper_shadow <= body * 0.3 and
            lower_shadow >= body * 2.0 and
            lower_shadow / candle_range > 0.6):
            hammer_pattern[i] = 1

    return hammer_pattern


def engulfing_bullish(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Bullish Engulfing Candlestick Pattern

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for bullish engulfing pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    engulfing_pattern = np.zeros_like(close, dtype=int)

    for i in range(1, len(close)):
        # Previous candle (i-1)
        prev_open = open_prices[i-1]
        prev_close = close[i-1]

        # Current candle (i)
        curr_open = open_prices[i]
        curr_close = close[i]

        # Previous candle must be bearish (close < open)
        # Current candle must be bullish (close > open)
        # Current candle body must completely engulf previous candle body
        if (prev_close < prev_open and  # Previous bearish
            curr_close > curr_open and  # Current bullish
            curr_open <= prev_close and  # Current open <= previous close
            curr_close >= prev_open):   # Current close >= previous open
            engulfing_pattern[i] = 1

    return engulfing_pattern


def engulfing_bearish(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Bearish Engulfing Candlestick Pattern

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for bearish engulfing pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    engulfing_pattern = np.zeros_like(close, dtype=int)

    for i in range(1, len(close)):
        # Previous candle (i-1)
        prev_open = open_prices[i-1]
        prev_close = close[i-1]

        # Current candle (i)
        curr_open = open_prices[i]
        curr_close = close[i]

        # Previous candle must be bullish (close > open)
        # Current candle must be bearish (close < open)
        # Current candle body must completely engulf previous candle body
        if (prev_close > prev_open and  # Previous bullish
            curr_close < curr_open and  # Current bearish
            curr_open >= prev_close and  # Current open >= previous close
            curr_close <= prev_open):   # Current close <= previous open
            engulfing_pattern[i] = 1

    return engulfing_pattern


def morning_star(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Morning Star Candlestick Pattern (3-candle reversal pattern)

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for morning star pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    morning_star_pattern = np.zeros_like(close, dtype=int)

    for i in range(2, len(close)):
        # First candle (bearish)
        first_open = open_prices[i-2]
        first_close = close[i-2]

        # Second candle (small, can be bullish or bearish)
        second_open = open_prices[i-1]
        second_close = close[i-1]

        # Third candle (bullish)
        third_open = open_prices[i]
        third_close = close[i]

        # Pattern conditions:
        # 1. First candle is bearish with significant body
        # 2. Second candle has small body (star)
        # 3. Third candle is bullish and closes well above first candle's midpoint
        if (first_close < first_open and  # First bearish
            abs(second_close - second_open) < abs(first_close - first_open) * 0.5 and  # Small second candle
            third_close > third_open and  # Third bullish
            third_close > (first_open + first_close) / 2):  # Closes above first candle mid
            morning_star_pattern[i] = 1

    return morning_star_pattern


def evening_star(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Evening Star Candlestick Pattern (3-candle reversal pattern)

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for evening star pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    evening_star_pattern = np.zeros_like(close, dtype=int)

    for i in range(2, len(close)):
        # First candle (bullish)
        first_open = open_prices[i-2]
        first_close = close[i-2]

        # Second candle (small, can be bullish or bearish)
        second_open = open_prices[i-1]
        second_close = close[i-1]

        # Third candle (bearish)
        third_open = open_prices[i]
        third_close = close[i]

        # Pattern conditions:
        # 1. First candle is bullish with significant body
        # 2. Second candle has small body (star)
        # 3. Third candle is bearish and closes well below first candle's midpoint
        if (first_close > first_open and  # First bullish
            abs(second_close - second_open) < abs(first_close - first_open) * 0.5 and  # Small second candle
            third_close < third_open and  # Third bearish
            third_close < (first_open + first_close) / 2):  # Closes below first candle mid
            evening_star_pattern[i] = 1

    return evening_star_pattern


def shooting_star(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Shooting Star Candlestick Pattern (bearish reversal)

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for shooting star pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    shooting_star_pattern = np.zeros_like(close, dtype=int)

    for i in range(len(close)):
        body = abs(close[i] - open_prices[i])
        candle_range = high[i] - low[i]

        if body < 1e-10:  # Very small body
            continue

        # Lower shadow should be very small (1/3 of body or less)
        # Upper shadow should be at least 2x the body size
        upper_shadow = high[i] - max(open_prices[i], close[i])
        lower_shadow = min(open_prices[i], close[i]) - low[i]

        if (lower_shadow <= body * 0.3 and
            upper_shadow >= body * 2.0 and
            upper_shadow / candle_range > 0.6):
            shooting_star_pattern[i] = 1

    return shooting_star_pattern


def spinning_top(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Spinning Top Candlestick Pattern (indecision/inversion potential)

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for spinning top pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    spinning_top_pattern = np.zeros_like(close, dtype=int)

    for i in range(len(close)):
        body = abs(close[i] - open_prices[i])
        candle_range = high[i] - low[i]

        # Spinning top has small body with upper and lower shadows of similar length
        upper_shadow = high[i] - max(open_prices[i], close[i])
        lower_shadow = min(open_prices[i], close[i]) - low[i]

        # Conditions: small body, balanced shadows, shadows significantly larger than body
        if (body <= candle_range * 0.3 and  # Body <= 30% of total range
            abs(upper_shadow - lower_shadow) <= body and  # Shadows roughly equal
            upper_shadow >= body * 2 and lower_shadow >= body * 2):
            spinning_top_pattern[i] = 1

    return spinning_top_pattern


def marubozu(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Marubozu Candlestick Pattern (strong trend continuation/breakout)

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for marubozu (bullish), -1 for bearish, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    marubozu_pattern = np.zeros_like(close, dtype=int)

    for i in range(len(close)):
        body = abs(close[i] - open_prices[i])
        upper_shadow = high[i] - max(open_prices[i], close[i])
        lower_shadow = min(open_prices[i], close[i]) - low[i]
        total_range = high[i] - low[i]

        # Marubozu: very small shadows (less than 5% of body)
        if (upper_shadow <= body * 0.05 and lower_shadow <= body * 0.05 and
            body > total_range * 0.8):  # Body > 80% of total range
            if close[i] > open_prices[i]:
                marubozu_pattern[i] = 1   # Bullish Marubozu
            else:
                marubozu_pattern[i] = -1  # Bearish Marubozu

    return marubozu_pattern


def harami_bullish(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Bullish Harami Candlestick Pattern (potential bullish reversal)

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for bullish harami pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    harami_pattern = np.zeros_like(close, dtype=int)

    for i in range(1, len(close)):
        # Previous candle (bearish)
        prev_open = open_prices[i-1]
        prev_close = close[i-1]
        prev_range = high[i-1] - low[i-1]

        # Current candle (bullish, smaller)
        curr_open = open_prices[i]
        curr_close = close[i]
        curr_range = high[i] - low[i]

        # Conditions: previous bearish, current bullish, current smaller range,
        # current candle contained within previous candle range
        if (prev_close < prev_open and  # Previous bearish
            curr_close > curr_open and  # Current bullish
            curr_range < prev_range * 0.7 and  # Current range significantly smaller
            min(curr_open, curr_close) > prev_open and  # Current body starts within previous
            max(curr_open, curr_close) < prev_close):   # Current body ends within previous
            harami_pattern[i] = 1

    return harami_pattern


def harami_bearish(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Bearish Harami Candlestick Pattern (potential bearish reversal)

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for bearish harami pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    harami_pattern = np.zeros_like(close, dtype=int)

    for i in range(1, len(close)):
        # Previous candle (bullish)
        prev_open = open_prices[i-1]
        prev_close = close[i-1]
        prev_range = high[i-1] - low[i-1]

        # Current candle (bearish, smaller)
        curr_open = open_prices[i]
        curr_close = close[i]
        curr_range = high[i] - low[i]

        # Conditions: previous bullish, current bearish, current smaller range,
        # current candle contained within previous candle range
        if (prev_close > prev_open and  # Previous bullish
            curr_close < curr_open and  # Current bearish
            curr_range < prev_range * 0.7 and  # Current range significantly smaller
            min(curr_open, curr_close) > prev_close and  # Current body starts within previous
            max(curr_open, curr_close) < prev_open):     # Current body ends within previous
            harami_pattern[i] = 1

    return harami_pattern


def harami_cross_bullish(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Bullish Harami Cross Candlestick Pattern (stronger reversal signal)

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for bullish harami cross pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    harami_cross_pattern = np.zeros_like(close, dtype=int)

    for i in range(1, len(close)):
        # Previous candle (strong bearish)
        prev_open = open_prices[i-1]
        prev_close = close[i-1]
        prev_high_shadow = prev_open - high[i-1] if prev_open > prev_close else prev_close - high[i-1]
        prev_low_shadow = low[i-1] - prev_close if prev_open > prev_close else low[i-1] - prev_open

        # Current candle (doji cross)
        curr_open = open_prices[i]
        curr_close = close[i]
        curr_body = abs(curr_close - curr_open)
        curr_range = high[i] - low[i]

        # Check if current is a doji/cross (very small body)
        is_cross = curr_body <= curr_range * 0.05

        # Conditions: previous strongly bearish, current is cross contained within previous
        if (prev_close < prev_open and  # Previous bearish
            is_cross and  # Current is cross/doji
            min(high[i], low[i]) > prev_open and     # Current contained within previous high
            max(high[i], low[i]) < prev_close):      # Current contained within previous low
            harami_cross_pattern[i] = 1

    return harami_cross_pattern


def piercing_pattern(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Piercing Pattern Candlestick (2-candle bullish reversal)

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for piercing pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    piercing_pattern = np.zeros_like(close, dtype=int)

    for i in range(1, len(close)):
        # First candle (bearish)
        first_open = open_prices[i-1]
        first_close = close[i-1]

        # Second candle (bullish)
        second_open = open_prices[i]
        second_close = close[i]

        # Conditions for piercing pattern
        if (first_close < first_open and  # First candle bearish
            second_close > second_open and  # Second candle bullish
            second_open < first_close and
            second_close > (first_open + first_close) / 2):
            piercing_pattern[i] = 1

    return piercing_pattern


def cloud_cover_dark(
    open_prices: Union[list, np.ndarray, pd.Series],
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Dark Cloud Cover Candlestick Pattern (bearish reversal)

    Args:
        open_prices: Open prices
        high: High prices
        low: Low prices
        close: Close prices

    Returns:
        numpy array with 1 for dark cloud cover pattern, 0 otherwise
    """
    open_prices = _verify_data(open_prices)
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)

    if len(open_prices) != len(high) or len(high) != len(low) or len(low) != len(close):
        raise ValueError("All input arrays must have the same length")

    dark_cloud_pattern = np.zeros_like(close, dtype=int)

    for i in range(1, len(close)):
        # First candle (strong bullish)
        first_open = open_prices[i-1]
        first_close = close[i-1]
        first_midpoint = (first_open + first_close) / 2

        # Second candle (bearish)
        second_open = open_prices[i]
        second_close = close[i]

        # Conditions: gaps up, closes deep into first candle body
        if (first_close > first_open and  # First candle bullish
            second_close < second_open and  # Second candle bearish
            second_open > first_close and    # Gaps up above first close
            second_close < first_midpoint):   # Closes into upper-mid of first body
            dark_cloud_pattern[i] = 1

    return dark_cloud_pattern


def parabolic_sar(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    acceleration: float = 0.02,
    max_acceleration: float = 0.2
) -> np.ndarray:
    """
    Parabolic SAR (Stop and Reverse) - trend following indicator

    Args:
        high: High prices
        low: Low prices
        acceleration: Initial acceleration factor (default 0.02)
        max_acceleration: Maximum acceleration factor (default 0.2)

    Returns:
        numpy array with Parabolic SAR values (stop levels)
    """
    high = _verify_data(high)
    low = _verify_data(low)

    if len(high) != len(low):
        raise ValueError("High and low arrays must have the same length")

    sar_values = np.full_like(high, np.nan, dtype=float)

    # Initialize first SAR value
    if len(high) > 0:
        sar_values[0] = low[0]  # Start with first low

    trend = 1  # 1 for uptrend, -1 for downtrend
    af = acceleration  # Acceleration factor

    for i in range(1, len(high)):
        # Calculate SAR for current period
        if trend == 1:  # Uptrend
            new_sar = sar_values[i-1] + af * (high[i-1] - sar_values[i-1])
            # Check if trend should reverse (SAR crosses above low)
            if new_sar > low[i]:
                new_sar = min(high[i-1], low[i])  # Reset SAR
                trend = -1  # Switch to downtrend
                af = acceleration  # Reset acceleration
            else:
                af = min(af + acceleration, max_acceleration)  # Increase acceleration
        else:  # Downtrend
            new_sar = sar_values[i-1] + af * (low[i-1] - sar_values[i-1])
            # Check if trend should reverse (SAR crosses below high)
            if new_sar < high[i]:
                new_sar = max(low[i-1], high[i])  # Reset SAR
                trend = 1  # Switch to uptrend
                af = acceleration  # Reset acceleration
            else:
                af = min(af + acceleration, max_acceleration)  # Increase acceleration

        sar_values[i] = new_sar

    return sar_values


def dpo(data: Union[list, np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
    """
    Detrended Price Oscillator

    Args:
        data: Input price data
        period: Period for calculation (default 14)

    Returns:
        numpy array with DPO values
    """
    data = _verify_data(data, period)

    dpo_values = np.full_like(data, np.nan, dtype=float)

    # DPO shifts SMA backwards by (period/2 + 1) periods to center it
    shift_period = period // 2 + 1

    for i in range(period + shift_period - 1, len(data)):
        # Calculate SMA for current period
        sma_current = np.mean(data[i - period + 1:i + 1])

        # Use SMA from (period/2 + 1) periods ago as detrending baseline
        baseline_index = i - shift_period
        if baseline_index >= 0:
            baseline = np.mean(data[baseline_index - period + 1:baseline_index + 1])
            dpo_values[i] = data[i] - baseline

    return dpo_values


def ppo(data: Union[list, np.ndarray, pd.Series], fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Percentage Price Oscillator

    Args:
        data: Input price data
        fastperiod: Fast EMA period (default 12)
        slowperiod: Slow EMA period (default 26)
        signalperiod: Signal line period (default 9)

    Returns:
        Tuple of (PPO line, Signal line, Histogram)
    """
    data = _verify_data(data, max(fastperiod, slowperiod, signalperiod))

    # Calculate EMAs
    fast_ema = ema(data, fastperiod)
    slow_ema = ema(data, slowperiod)

    # PPO calculation
    # PPO = ((fast_ema - slow_ema) / slow_ema) * 100
    ppo_line = np.full_like(data, np.nan, dtype=float)

    for i in range(len(data)):
        if not np.isnan(fast_ema[i]) and not np.isnan(slow_ema[i]) and slow_ema[i] != 0:
            ppo_line[i] = ((fast_ema[i] - slow_ema[i]) / slow_ema[i]) * 100

    # Signal line is EMA of PPO line
    signal_line = ema(ppo_line, signalperiod)

    # Histogram is PPO - Signal
    histogram = ppo_line - signal_line

    return ppo_line, signal_line, histogram


def kst(
    data: Union[list, np.ndarray, pd.Series],
    roc_periods: List[int] = None,
    sma_periods: List[int] = None
) -> np.ndarray:
    """
    Know Sure Thing (KST) Oscillator

    Args:
        data: Input price data
        roc_periods: List of ROC periods (default [10, 15, 20, 30])
        sma_periods: List of SMA periods for smoothing (default [10, 10, 10, 15])

    Returns:
        numpy array with KST values
    """
    if roc_periods is None:
        roc_periods = [10, 15, 20, 30]
    if sma_periods is None:
        sma_periods = [10, 10, 10, 15]

    if len(roc_periods) != len(sma_periods):
        raise ValueError("ROC periods and SMA periods lists must have the same length")

    # Calculate ROC for each period
    roc_list = []
    for period in roc_periods:
        roc_values = roc(data, period)
        roc_list.append(roc_values)

    # Apply SMA smoothing to each ROC
    smoothed_roc_list = []
    max_period = max(roc_periods + sma_periods)
    data = _verify_data(data, max_period)

    for i, (roc_values, sma_period) in enumerate(zip(roc_list, sma_periods)):
        # Apply SMA smoothing
        smoothed_values = np.full_like(data, np.nan, dtype=float)
        for j in range(sma_period - 1, len(roc_values)):
            window = roc_values[j - sma_period + 1:j + 1]
            valid_values = window[~np.isnan(window)]
            if len(valid_values) > 0:
                smoothed_values[j] = np.mean(valid_values)
        smoothed_roc_list.append(smoothed_values)

    # Combine all smoothed ROCs into KST
    kst_values = np.full_like(data, np.nan, dtype=float)

    for i in range(len(data)):
        valid_sum = 0
        valid_count = 0

        for smoothed_roc in smoothed_roc_list:
            if not np.isnan(smoothed_roc[i]):
                valid_sum += smoothed_roc[i]
                valid_count += 1

        if valid_count > 0:
            kst_values[i] = valid_sum

    return kst_values


def ease_of_movement(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    volume: Union[list, np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Ease of Movement (EMV) Oscillator

    Args:
        high: High prices
        low: Low prices
        volume: Volume data
        period: Period for EMA smoothing (default 14)

    Returns:
        numpy array with EMV values
    """
    high = _verify_data(high)
    low = _verify_data(low)
    volume = _verify_data(volume)

    if len(high) != len(low) or len(high) != len(volume):
        raise ValueError("All input arrays must have the same length")

    emv_raw = np.full_like(high, np.nan, dtype=float)

    for i in range(1, len(high)):
        # Midpoint move
        midpoint_move = (high[i] + low[i]) / 2 - (high[i-1] + low[i-1]) / 2

        # Box ratio (range/volume)
        price_range = high[i] - low[i]
        if price_range != 0:
            box_ratio = (volume[i] / price_range) / 1000000  # Scale down
        else:
            box_ratio = 0

        # Raw EMV
        emv_raw[i] = midpoint_move * box_ratio

    # Apply EMA smoothing
    emv_values = ema(emv_raw, period)

    return emv_values


def chaikin_volatility(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    ema_period: int = 10,
    roc_period: int = 12
) -> np.ndarray:
    """
    Chaikin Volatility

    Args:
        high: High prices
        low: Low prices
        ema_period: Period for EMA calculation (default 10)
        roc_period: Period for ROC calculation (default 12)

    Returns:
        numpy array with Chaikin Volatility values
    """
    high = _verify_data(high)
    low = _verify_data(low)

    if len(high) != len(low):
        raise ValueError("High and low arrays must have the same length")

    # Calculate high-low range
    hl_range = high - low

    # Calculate EMA of range
    ema_range = ema(hl_range, ema_period)

    # Calculate ROC of EMA
    chaikin_vol = roc(ema_range, roc_period)

    return chaikin_vol


def normalized_atr(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series],
    period: int = 14
) -> np.ndarray:
    """
    Normalized Average True Range (ATR % of Close Price)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for ATR calculation

    Returns:
        numpy array with normalized ATR values as percentage
    """
    # Calculate ATR
    atr_values = atr(high, low, close, period)

    # Normalize by close price
    close = _verify_data(close)
    normalized_atr_values = np.full_like(atr_values, np.nan, dtype=float)

    for i in range(len(atr_values)):
        if not np.isnan(atr_values[i]) and abs(close[i]) > 1e-10:
            normalized_atr_values[i] = (atr_values[i] / close[i]) * 100

    return normalized_atr_values


def volatility_ratio(
    data: Union[list, np.ndarray, pd.Series],
    short_period: int = 5,
    long_period: int = 15
) -> np.ndarray:
    """
    Volatility Ratio

    Args:
        data: Input price data
        short_period: Short-term volatility period (default 5)
        long_period: Long-term volatility period (default 15)

    Returns:
        numpy array with volatility ratio values
    """
    data = _verify_data(data, max(short_period, long_period))

    # Calculate short-term volatility (standard deviation)
    short_vol = stddev(data, short_period)

    # Calculate long-term volatility
    long_vol = stddev(data, long_period)

    # Volatility ratio = short_vol / long_vol
    ratio_values = np.full_like(data, np.nan, dtype=float)

    for i in range(len(data)):
        if not np.isnan(short_vol[i]) and not np.isnan(long_vol[i]) and long_vol[i] != 0:
            ratio_values[i] = short_vol[i] / long_vol[i]

    return ratio_values


def standard_error_channels(
    data: Union[list, np.ndarray, pd.Series],
    period: int = 20,
    nbdev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard Error Channels (similar to Bollinger but using standard error)

    Args:
        data: Input price data
        period: Period for calculation
        nbdev: Number of standard errors

    Returns:
        Tuple of (upper band, middle band, lower band)
    """
    data = _verify_data(data, period)

    # Calculate linear regression for the period
    linreg_values = linearreg(data, period)

    # Calculate standard error of regression
    se_values = np.full_like(data, np.nan, dtype=float)

    for i in range(period - 1, len(data)):
        data_slice = data[i - period + 1:i + 1]

        if not np.isnan(linreg_values[i]):
            # Calculate squared errors
            errors = (data_slice - linreg_values[i]) ** 2
            mean_squared_error = np.mean(errors)
            standard_error = np.sqrt(mean_squared_error)
            se_values[i] = standard_error

    # Calculate channels
    upper_band = linreg_values + nbdev * se_values
    middle_band = linreg_values
    lower_band = linreg_values - nbdev * se_values

    return upper_band, middle_band, lower_band


def linearregubslope(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Linear Regression Slope (using absolute values)

    Args:
        data: Input price data
        period: Period for linear regression

    Returns:
        numpy array with absolute slope values (always positive)
    """
    slopes = linearreg_slope(data, period)
    return np.abs(slopes)


def linearregangle(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Linear Regression Angle in degrees

    Args:
        data: Input price data
        period: Period for linear regression

    Returns:
        numpy array with angle values in degrees (arctan of slope)
    """
    slopes = linearreg_slope(data, period)
    angles = np.degrees(np.arctan(slopes))
    return angles


def linregrsi(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Linear Regression RSI (RSI calculated on linear regression values)

    Args:
        data: Input price data
        period: Period for calculation

    Returns:
        numpy array with RSI of linear regression values
    """
    linreg_values = linearreg(data, period)
    return rsi(linreg_values, 14)  # Standard RSI period of 14


def pfe(data: Union[list, np.ndarray, pd.Series], period: int = 10) -> np.ndarray:
    """
    Polarized Fractal Efficiency (PFE) - measures trend efficiency

    Args:
        data: Input price data
        period: Period for calculation

    Returns:
        numpy array with PFE values (-100 to +100)
    """
    data = _verify_data(data, period + 1)

    pfe_values = np.full_like(data, np.nan, dtype=float)

    for i in range(period, len(data)):
        # Calculate the straight line distance (first to last point)
        start_price = data[i - period]
        end_price = data[i]
        straight_distance = abs(end_price - start_price)

        # Calculate the sum of absolute price changes (actual path)
        actual_distance = sum(abs(data[j+1] - data[j]) for j in range(i - period, i))

        # Calculate efficiency
        efficiency = 0
        if actual_distance != 0:
            efficiency = straight_distance / actual_distance

        # Assign direction based on trend
        if end_price > start_price:
            pfe_values[i] = efficiency * 100  # Positive trend
        elif end_price < start_price:
            pfe_values[i] = -efficiency * 100  # Negative trend
        else:
            pfe_values[i] = 0  # No trend

    return pfe_values


def arfaith(data: Union[list, np.ndarray, pd.Series], period: int = 50) -> np.ndarray:
    """
    Arkin-Feder-Feingold-Adler Information-Theoretic Analysis
    Measures the information content of price movement

    Args:
        data: Input price data
        period: Period for calculation

    Returns:
        numpy array with Arkin-Feder-Feingold-Adler values
    """
    data = _verify_data(data, period + 1)

    arfaith_values = np.full_like(data, np.nan, dtype=float)

    for i in range(period, len(data)):
        # Calculate price changes
        changes = np.diff(data[i - period:i + 1])

        # Count positive and negative changes
        pos_changes = np.sum(changes > 0)
        neg_changes = np.sum(changes < 0)

        total_changes = len(changes)

        if total_changes > 0:
            # Calculate probabilities
            p_pos = pos_changes / total_changes
            p_neg = neg_changes / total_changes
            p_zero = (total_changes - pos_changes - neg_changes) / total_changes

            # Calculate information content using entropy formula
            entropy = 0
            if p_pos > 0:
                entropy -= p_pos * np.log2(p_pos)
            if p_neg > 0:
                entropy -= p_neg * np.log2(p_neg)
            if p_zero > 0:
                entropy -= p_zero * np.log2(p_zero)

            # Convert to Arkin index
            arfaith_values[i] = (1 + entropy / np.log2(3)) * 100 - 100

    return arfaith_values


def trend_intensity(data: Union[list, np.ndarray, pd.Series], period: int = 14) -> np.ndarray:
    """
    Trend Intensity Index - measures the strength of price trend

    Args:
        data: Input price data
        period: Period for calculation

    Returns:
        numpy array with trend intensity values (0-100)
    """
    data = _verify_data(data, period)

    trend_intensity_values = np.full_like(data, np.nan, dtype=float)

    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]

        # Calculate linear regression
        x = np.arange(len(window))
        y = window
        slope = np.cov(x, y)[0, 1] / np.var(x)
        intercept = np.mean(y) - slope * np.mean(x)

        # Calculate R-squared (coefficient of determination)
        y_predicted = intercept + slope * x
        ss_res = np.sum((y - y_predicted) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 0
        if ss_tot != 0:
            r_squared = 1 - (ss_res / ss_tot)

        # Calculate price change over period
        price_change = abs(window[-1] - window[0])
        avg_price = np.mean(window)

        # Normalize price change
        change_intensity = price_change / avg_price if avg_price != 0 else 0

        # Combine R-squared with change intensity
        trend_intensity_values[i] = min(100, r_squared * change_intensity * 100)

    return trend_intensity_values


def kvo(
    high: Union[list, np.ndarray, pd.Series],
    low: Union[list, np.ndarray, pd.Series],
    close: Union[list, np.ndarray, pd.Series],
    volume: Union[list, np.ndarray, pd.Series],
    short_period: int = 34,
    long_period: int = 55
) -> np.ndarray:
    """
    Klinger Volume Oscillator

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        short_period: Short EMA period (default 34)
        long_period: Long EMA period (default 55)

    Returns:
        numpy array with KVO values
    """
    high = _verify_data(high)
    low = _verify_data(low)
    close = _verify_data(close)
    volume = _verify_data(volume)

    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("All input arrays must have the same length")

    # Calculate Volume Force (VF)
    # VF = Volume * [2*(daily range delta/close) - 1] * direction where direction = uptrend (+1) or downtrend (-1)
    vf = np.full_like(close, np.nan)

    for i in range(1, len(close)):
        trend_direction = 1 if close[i] > close[i-1] else -1

        # Simplified Klinger Volume Force calculation
        range_delta = abs(high[i] - low[i])
        volume_force = volume[i] * trend_direction * (range_delta / close[i-1] if close[i-1] != 0 else 0)
        vf[i] = volume_force

    # Apply double smoothing
    ema_short = ema(vf, short_period)
    ema_long = ema(vf, long_period)

    # KVO = EMA(34) - EMA(55)
    kvo_values = ema_short - ema_long

    return kvo_values


def sine(data: Union[list, np.ndarray, pd.Series], period: int = 5) -> np.ndarray:
    """
    Sine Wave Indicator using Hilbert Transform approach

    Args:
        data: Input price data
        period: Period for calculation

    Returns:
        numpy array with sine wave values
    """
    data = _verify_data(data, period * 2)

    sine_values = np.full_like(data, np.nan, dtype=float)

    for i in range(period * 2 - 1, len(data)):
        # Simplified sine wave calculation based on moving average cycles
        recent_avg = np.mean(data[i - period + 1:i + 1])

        # Calculate Hilbert-like transform approximation
        # This is a simplified version - true Hilbert would require more complex FFT
        phase = (i % period) / period * 2 * np.pi
        sine_values[i] = np.sin(phase) * recent_avg + recent_avg

    return sine_values


def center_of_gravity(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Center of Gravity Oscillator - weighted average of price data

    Args:
        data: Input price data
        period: Period for calculation

    Returns:
        numpy array with Center of Gravity values
    """
    data = _verify_data(data, period)

    cog_values = np.full_like(data, np.nan, dtype=float)

    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]

        # Calculate weighted sum with weights increasing linearly
        weighted_sum = 0
        total_weight = 0

        for j in range(len(window)):
            weight = j + 1  # Simple linear weighting
            weighted_sum += window[j] * weight
            total_weight += weight

        if total_weight > 0:
            cog_values[i] = weighted_sum / total_weight

    return cog_values


def beta(
    market_data: Union[list, np.ndarray, pd.Series],
    asset_data: Union[list, np.ndarray, pd.Series],
    period: int = 30
) -> np.ndarray:
    """
    Beta coefficient - measure of volatility relative to market

    Args:
        market_data: Market/benchmark price data
        asset_data: Asset price data for beta calculation
        period: Period for beta calculation (default 30)

    Returns:
        numpy array with beta values (volatility measure)
    """
    market_data = _verify_data(market_data, period)
    asset_data = _verify_data(asset_data, period)

    if len(market_data) != len(asset_data):
        raise ValueError("Market and asset data arrays must have the same length")

    beta_values = np.full_like(asset_data, np.nan, dtype=float)

    # Calculate returns for the period
    market_returns = np.diff(market_data) / market_data[:-1]
    asset_returns = np.diff(asset_data) / asset_data[:-1]

    # Add NaN at beginning to align arrays
    market_returns = np.concatenate([[np.nan], market_returns])
    asset_returns = np.concatenate([[np.nan], asset_returns])

    for i in range(period, len(asset_data)):
        # Rolling beta calculation
        market_slice = market_returns[i - period + 1:i + 1]
        asset_slice = asset_returns[i - period + 1:i + 1]

        # Remove NaN values for calculation
        valid_indices = ~np.isnan(market_slice) & ~np.isnan(asset_slice)
        if np.sum(valid_indices) >= 2:
            market_valid = market_slice[valid_indices]
            asset_valid = asset_slice[valid_indices]

            # Calculate covariance and variance
            covariance = np.cov(market_valid, asset_valid)[0, 1]
            variance = np.var(market_valid)

            if variance != 0:
                beta_values[i] = covariance / variance

    return beta_values


# Missing functions that were referenced in __init__.py but not implemented

def linearreg(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Linear Regression - returns the linear regression line for the period

    Args:
        data: Input price data
        period: Period for linear regression

    Returns:
        numpy array with linear regression values
    """
    data = _verify_data(data, period)

    linreg_values = np.full_like(data, np.nan, dtype=float)

    for i in range(period - 1, len(data)):
        # Get the slice of data for this period
        y = data[i - period + 1:i + 1]

        # Create x values (0 to period-1)
        x = np.arange(period)

        # Calculate linear regression coefficients
        slope = np.cov(x, y)[0, 1] / np.var(x)
        intercept = np.mean(y) - slope * np.mean(x)

        # Calculate regression value for the last point (x = period - 1)
        linreg_values[i] = intercept + slope * (period - 1)

    return linreg_values


def linearreg_slope(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Linear Regression Slope

    Args:
        data: Input price data
        period: Period for linear regression

    Returns:
        numpy array with slope values for each period
    """
    data = _verify_data(data, period)

    slope_values = np.full_like(data, np.nan, dtype=float)

    for i in range(period - 1, len(data)):
        # Get the slice of data for this period
        y = data[i - period + 1:i + 1]

        # Create x values (0 to period-1)
        x = np.arange(period)

        # Calculate slope using covariance approach
        slope = np.cov(x, y)[0, 1] / np.var(x)
        slope_values[i] = slope

    return slope_values


def linearreg_intercept(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray:
    """
    Linear Regression Intercept

    Args:
        data: Input price data
        period: Period for linear regression

    Returns:
        numpy array with intercept values for each period
    """
    data = _verify_data(data, period)

    intercept_values = np.full_like(data, np.nan, dtype=float)

    for i in range(period - 1, len(data)):
        # Get the slice of data for this period
        y = data[i - period + 1:i + 1]

        # Create x values (0 to period-1)
        x = np.arange(period)

        # Calculate slope and intercept
        slope = np.cov(x, y)[0, 1] / np.var(x)
        intercept = np.mean(y) - slope * np.mean(x)

        intercept_values[i] = intercept

    return intercept_values


def stddev(data: Union[list, np.ndarray, pd.Series], period: int, method: int = 0) -> np.ndarray:
    """
    Standard Deviation over period

    Args:
        data: Input price data
        period: Period for calculation
        method: Method for calculation (0 for population std, 1 for sample std)

    Returns:
        numpy array with standard deviation values
    """
    data = _verify_data(data, period)

    std_values = np.full_like(data, np.nan, dtype=float)
    ddof = 1 if method == 1 else 0  # degrees of freedom

    for i in range(period - 1, len(data)):
        window = data[i - period + 1:i + 1]
        std_values[i] = np.std(window, ddof=ddof)

    return std_values
