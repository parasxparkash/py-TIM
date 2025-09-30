# TA-Analysis Library - Complete API Reference

## Table of Contents
1. [Trend Indicators](#trend-indicators)
2. [Momentum Indicators](#momentum-indicators)
3. [Volatility Indicators](#volatility-indicators)
4. [Volume Indicators](#volume-indicators)
5. [Statistical Indicators](#statistical-indicators)
6. [Price Transform Indicators](#price-transform-indicators)
7. [Candlestick Patterns](#candlestick-patterns)
8. [Common Parameters](#common-parameters)
9. [Data Types](#data-types)
10. [Error Handling](#error-handling)

---

## Trend Indicators

### Simple Moving Average
```python
sma(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray
```
- **Description**: Calculates the arithmetic mean of a series over a specified period
- **Formula**: `SMA = (sum of values over period) / period`
- **Parameters**: `period` (int): Number of periods for calculation
- **Returns**: Array with SMA values, NaN for insufficient data

### Exponential Moving Average
```python
ema(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray
```
- **Description**: Applies exponential smoothing to price data
- **Formula**: `EMA(today) = (price * multiplier) + (EMA(yesterday) * (1 - multiplier))`
- **Parameters**: `period` (int): Number of periods for calculation
- **Returns**: Array with EMA values

### Weighted Moving Average
```python
wma(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray
```
- **Description**: Applies linear weighting to price data
- **Formula**: `WMA = Σ(weight * price) / Σ(weights)`
- **Parameters**: `period` (int): Number of periods for calculation
- **Returns**: Array with WMA values

### Kaufman's Adaptive Moving Average
```python
kama(data: Union[list, np.ndarray, pd.Series], period: int = 30) -> np.ndarray
```
- **Description**: Adaptive moving average that adjusts to market volatility
- **Parameters**: `period` (int): Period for efficiency ratio calculation (default 30)
- **Returns**: Array with KAMA values

### Hull Moving Average
```python
hma(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray
```
- **Description**: Moving average designed to reduce lag and improve smoothness
- **Parameters**: `period` (int): Period for calculation
- **Returns**: Array with HMA values

---

## Momentum Indicators

### Relative Strength Index
```python
rsi(data: Union[list, np.ndarray, pd.Series], period: int = 14) -> np.ndarray
```
- **Description**: Measures price change velocity and magnitude
- **Formula**: `RSI = 100 - (100 / (1 + RS))` where `RS = Average Gain / Average Loss`
- **Parameters**: `period` (int): Period for gain/loss calculation (default 14)
- **Range**: 0-100

### MACD (Moving Average Convergence Divergence)
```python
macd(data: Union[list, np.ndarray, pd.Series], fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```
- **Description**: Shows relationship between two moving averages
- **Returns**: Tuple of (MACD line, Signal line, Histogram)

### Stochastic Oscillator
```python
stoch(high: Union[list, np.ndarray, pd.Series], low: Union[list, np.ndarray, pd.Series], close: Union[list, np.ndarray, pd.Series], fastk_period: int = 14, slowk_period: int = 1, slowd_period: int = 3) -> Tuple[np.ndarray, np.ndarray]
```
- **Description**: Compares closing price to price range over time
- **Returns**: Tuple of (%K, %D) values, both scaled 0-100

### Commodity Channel Index
```python
cci(high: Union[list, np.ndarray, pd.Series], low: Union[list, np.ndarray, pd.Series], close: Union[list, np.ndarray, pd.Series], period: int = 14) -> np.ndarray
```
- **Description**: Measures deviation from mean price
- **Range**: Typically -200 to +200

### Average Directional Index
```python
adx(high: Union[list, np.ndarray, pd.Series], low: Union[list, np.ndarray, pd.Series], close: Union[list, np.ndarray, pd.Series], period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```
- **Description**: Quantifies trend strength regardless of direction
- **Returns**: Tuple of (ADX, +DI, -DI), all scaled 0-100

### Momentum
```python
mom(data: Union[list, np.ndarray, pd.Series], period: int = 10) -> np.ndarray
```
- **Description**: Price difference from n periods ago
- **Formula**: `MOM = price(today) - price(today - n)`
- **Parameters**: `period` (int): Periods for comparison (default 10)

### Rate of Change
```python
roc(data: Union[list, np.ndarray, pd.Series], period: int = 10) -> np.ndarray
```
- **Description**: Percentage price change over specified period
- **Formula**: `ROC = ((price(today) / price(today - n)) - 1) * 100`
- **Range**: Unbounded percentage values

### Stochastic RSI
```python
stochrsi(data: Union[list, np.ndarray, pd.Series], rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]
```
- **Description**: Applies Stochastic oscillator to RSI values
- **Returns**: Tuple of (Stochastic RSI %K, Stochastic RSI %D)

---

## Volatility Indicators

### Bollinger Bands
```python
bollinger_bands(data: Union[list, np.ndarray, pd.Series], period: int = 20, nbdevup: float = 2.0, nbdevdn: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```
- **Description**: Price channels based on standard deviation
- **Returns**: Tuple of (upper band, middle band, lower band)

### Bollinger %B
```python
bollinger_percent_b(data: Union[list, np.ndarray, pd.Series], period: int = 20, nbdev: float = 2.0) -> np.ndarray
```
- **Description**: Position within Bollinger Bands as percentage
- **Formula**: `(price - lower) / (upper - lower)`
- **Range**: 0-1 (0 = at lower band, 1 = at upper band)

### Bollinger Bandwidth
```python
bollinger_bandwidth(data: Union[list, np.ndarray, pd.Series], period: int = 20, nbdev: float = 2.0) -> np.ndarray
```
- **Description**: Relative expansion/contraction of Bollinger Bands
- **Formula**: `(upper - lower) / middle`

### Average True Range
```python
atr(high: Union[list, np.ndarray, pd.Series], low: Union[list, np.ndarray, pd.Series], close: Union[list, np.ndarray, pd.Series], period: int = 14) -> np.ndarray
```
- **Description**: Average of true ranges over specified period
- **Formula**: `TR = max(high - low, |high - close_prev|, |low - close_prev|)`

### Mass Index
```python
mass_index(high: Union[list, np.ndarray, pd.Series], low: Union[list, np.ndarray, pd.Series], ema_period: int = 9, sum_period: int = 25) -> np.ndarray
```
- **Description**: Detects volatility patterns and potential reversals
- **Parameters**: `ema_period` (int): EMA period for range (default 9), `sum_period` (int): Sum period (default 25)

---

## Volume Indicators

### On-Balance Volume
```python
obv(close: Union[list, np.ndarray, pd.Series], volume: Union[list, np.ndarray, pd.Series]) -> np.ndarray
```
- **Description**: Cumulative volume based on price direction
- **Formula**: If close_up: OBV = prev_OBV + volume, If close_down: OBV = prev_OBV - volume

### Money Flow Index
```python
mfi(high: Union[list, np.ndarray, pd.Series], low: Union[list, np.ndarray, pd.Series], close: Union[list, np.ndarray, pd.Series], volume: Union[list, np.ndarray, pd.Series], period: int = 14) -> np.ndarray
```
- **Description**: Volume-weighted RSI
- **Range**: 0-100

### Chaikin Money Flow
```python
cmf(high: Union[list, np.ndarray, pd.Series], low: Union[list, np.ndarray, pd.Series], close: Union[list, np.ndarray, pd.Series], volume: Union[list, np.ndarray, pd.Series], period: int = 21) -> np.ndarray
```
- **Description**: Volume-weighted average of accumulation/distribution
- **Parameters**: `period` (int): Period for summation (default 21)

### Volume Weighted Average Price
```python
vwap(high: Union[list, np.ndarray, pd.Series], low: Union[list, np.ndarray, pd.Series], close: Union[list, np.ndarray, pd.Series], volume: Union[list, np.ndarray, pd.Series]) -> np.ndarray
```
- **Description**: Volume-weighted average price over the dataset

### Force Index
```python
force(close: Union[list, np.ndarray, pd.Series], volume: Union[list, np.ndarray, pd.Series], period: int = 13) -> np.ndarray
```
- **Description**: Combines price change with volume
- **Formula**: `(close - close_prev) * volume`, then smoothed with EMA

---

## Candlestick Patterns

### Doji Pattern
```python
doji(open_prices: Union[list, np.ndarray, pd.Series], close: Union[list, np.ndarray, pd.Series]) -> np.ndarray
```
- **Description**: Candle with very small body indicating indecision
- **Returns**: 1 for doji pattern detected, 0 otherwise

### Hammer Pattern
```python
hammer(open_prices, high, low, close, period=None) -> np.ndarray
```
- **Description**: Bullish reversal pattern with long lower wick
- **Returns**: 1 for hammer pattern detected, 0 otherwise

### Engulfing Patterns
```python
engulfing_bullish(open_prices, high, low, close) -> np.ndarray
engulfing_bearish(open_prices, high, low, close) -> np.ndarray
```
- **Description**: Larger candle completely engulfing previous candle
- **Returns**: 1 for engulfing pattern detected, 0 otherwise

### Morning Star Pattern
```python
morning_star(open_prices, high, low, close) -> np.ndarray
```
- **Description**: 3-candle bullish reversal pattern
- **Returns**: 1 for morning star pattern detected, 0 otherwise

---

## Statistical Indicators

### Pearson's Correlation Coefficient
```python
correl(data1: Union[list, np.ndarray, pd.Series], data2: Union[list, np.ndarray, pd.Series], period: int = 30) -> np.ndarray
```
- **Description**: Measures linear relationship between two datasets
- **Range**: -1 to +1

### Time Series Forecast
```python
tsf(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray
```
- **Description**: Linear regression forecast for next period
- **Formula**: Uses slope and intercept from linear regression

### Center of Gravity
```python
center_of_gravity(data: Union[list, np.ndarray, pd.Series], period: int) -> np.ndarray
```
- **Description**: Weighted average of price data over period
- **Formula**: Sum of (price * weight) / sum of weights

---

## Common Parameters

### Data Input Parameters
- **data**: Price/volume series as list, numpy array, or pandas Series
- **high**: High price series (for multi-price indicators)
- **low**: Low price series (for multi-price indicators)
- **close**: Close price series
- **open**: Open price series (for candlestick patterns)
- **volume**: Volume series (for volume-based indicators)

### Calculation Parameters
- **period**: Lookback window/averaging period (typically 5-200)
- **fastperiod**: Shorter period for indicators like MACD (default 12)
- **slowperiod**: Longer period for indicators like MACD (default 26)

### Standard Deviations
- **nbdev**: Number of standard deviations (typically 1.0-3.0)

---

## Data Types

All functions accept and return:

- **Input**: `list`, `numpy.ndarray`, or `pandas.Series`
- **Output**: `numpy.ndarray` (same length as input, with NaN for insufficient data)
- **Multi-return functions**: Tuple of numpy arrays

---

## Error Handling

All functions include comprehensive error checking:

### Common Errors:
- **ValueError**: Invalid input types or parameter values (e.g., negative periods)
- **TypeError**: Wrong data type passed to function
- **AssertionError**: Array length mismatches for multi-price indicators

### Error Examples:
```python
try:
    result = sma([1, 2, 3], period=-1)
except ValueError as e:
    # "Period must be greater than 0"
    print(e)

try:
    result = adx([1, 2, 3], [4, 5, 6], [7, 8, 9], period=14)
except ValueError as e:
    # "Array length must be at least period + 1" for ADX
    print(e)
```

### Data Validation:
- Automatic type conversion to numpy arrays
- NaN handling for divisions by zero
- Input length validation
- Array shape consistency checks

---

## Performance Notes

- **NumPy Optimization**: All calculations use vectorized NumPy operations
- **Memory Efficient**: Processes data in-place where possible
- **Large Dataset Ready**: Can handle 100,000+ data points efficiently
- **Time Complexity**: O(n) for most indicators, O(n*m) for lookback-dependent calculations

---

## Algorithmic Details

For detailed mathematical formulas and implementation specifics, see the function docstrings and source code. All indicators are implemented according to established financial literature and industry standards.
