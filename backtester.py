"""
py-TIM Library - Simple Backtesting Framework
Demonstrates practical usage of technical indicators for trading strategy development
"""

import numpy as np
import pandas as pd
import inspect
from typing import Dict, List, Tuple, Optional, Callable
from indicators import *


class SimpleBacktester:
    """
    Simple backtesting framework using py-TIM technical indicators
    """

    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0):
        """
        Initialize backtester

        Args:
            data: DataFrame with OHLCV columns (Open, High, Low, Close, Volume)
            initial_capital: Starting capital for backtesting
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        self.data = data.copy()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.trades = []
        self.balance_history = [initial_capital]

    def generate_signals(self, indicator_func: Callable, **kwargs) -> np.ndarray:
        """
        Generate signals using any py-TIM indicator

        Args:
            indicator_func: py-TIM indicator function
            **kwargs: Parameters for the indicator

        Returns:
            Signal array (-1 = sell, 0 = hold, 1 = buy)
        """
        # Extract OHLCV data based on function signature
        params = inspect.signature(indicator_func).parameters

        # Check if function needs OHLCV data
        if 'high' in params and 'low' in params and 'close' in params:
            if 'open_prices' in params:
                result = indicator_func(
                    self.data['Open'].values,
                    self.data['High'].values,
                    self.data['Low'].values,
                    self.data['Close'].values,
                    **kwargs
                )
            elif 'volume' in params:
                result = indicator_func(
                    self.data['High'].values,
                    self.data['Low'].values,
                    self.data['Close'].values,
                    self.data['Volume'].values,
                    **kwargs
                )
            else:
                result = indicator_func(
                    self.data['High'].values,
                    self.data['Low'].values,
                    self.data['Close'].values,
                    **kwargs
                )
        elif 'close' in params and 'volume' in params:
            result = indicator_func(
                self.data['Close'].values,
                self.data['Volume'].values,
                **kwargs
            )
        elif 'close' in params:
            result = indicator_func(self.data['Close'].values, **kwargs)
        else:
            # Fallback for indicators that use any data column
            result = indicator_func(self.data['Close'].values, **kwargs)

        return result

    def create_strategy(self,
                       buy_signal_func: Callable,
                       sell_signal_func: Callable,
                       buy_params: Dict = None,
                       sell_params: Dict = None,
                       position_size_pct: float = 0.1) -> Dict:
        """
        Create and run a trading strategy

        Args:
            buy_signal_func: Function to generate buy signals
            sell_signal_func: Function to generate sell signals
            buy_params: Parameters for buy signal function
            sell_params: Parameters for sell signal function
            position_size_pct: Percentage of capital to use per trade

        Returns:
            Dictionary with backtesting results
        """
        buy_params = buy_params or {}
        sell_params = sell_params or {}

        # Generate signals - check if these are TA indicator functions or raw signal generators
        if callable(buy_signal_func):
            try:
                # Try calling as a TA indicator function
                buy_signals_raw = self.generate_signals(buy_signal_func, **buy_params)
            except TypeError:
                # If that fails, call as a raw signal generator (no args)
                buy_signals_raw = buy_signal_func()
                if not hasattr(buy_signals_raw, '__len__') or len(buy_signals_raw) != len(self.data):
                    # Convert scalar signal to array
                    buy_signals_raw = np.full(len(self.data), buy_signals_raw, dtype=float)

        if callable(sell_signal_func):
            try:
                sell_signals_raw = self.generate_signals(sell_signal_func, **sell_params)
            except TypeError:
                # If that fails, call as a raw signal generator (no args)
                sell_signals_raw = sell_signal_func()
                if not hasattr(sell_signals_raw, '__len__') or len(sell_signals_raw) != len(self.data):
                    # Convert scalar signal to array
                    sell_signals_raw = np.full(len(self.data), sell_signals_raw, dtype=float)

        # Convert to binary signals (position sizing)
        position_size = self.initial_capital * position_size_pct

        # Run the strategy
        for i in range(len(self.data)):
            current_price = self.data.iloc[i]['Close']

            # Buy signals (enter long position)
            if buy_signals_raw[i] > 0 and self.position <= 0:
                if self.capital >= position_size:
                    shares = int(position_size / current_price)
                    cost = shares * current_price

                    if self.position == -1:  # Close short first
                        self.capital += (self.entry_price - current_price) * abs(self.position)
                        self.trades.append({
                            'type': 'close_short',
                            'price': current_price,
                            'shares': abs(self.position),
                            'profit_loss': (self.entry_price - current_price) * abs(self.position)
                        })

                    self.capital -= cost
                    self.position = shares
                    self.entry_price = current_price

                    self.trades.append({
                        'type': 'buy',
                        'price': current_price,
                        'shares': shares,
                        'capital_before': self.capital + cost
                    })

            # Sell signals (enter short position)
            elif sell_signals_raw[i] > 0 and self.position >= 0:
                if self.capital >= position_size:
                    shares = int(position_size / current_price)
                    cash_from_short = shares * current_price

                    if self.position > 0:  # Close long first
                        self.capital += (current_price - self.entry_price) * self.position
                        self.trades.append({
                            'type': 'close_long',
                            'price': current_price,
                            'shares': self.position,
                            'profit_loss': (current_price - self.entry_price) * self.position
                        })

                    self.capital += cash_from_short
                    self.position = -shares
                    self.entry_price = current_price

                    self.trades.append({
                        'type': 'short',
                        'price': current_price,
                        'shares': shares,
                        'capital_before': self.capital - cash_from_short
                    })

            # Update balance history
            if self.position > 0:
                unrealized_pnl = (current_price - self.entry_price) * self.position
            elif self.position < 0:
                unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
            else:
                unrealized_pnl = 0

            total_value = self.capital + unrealized_pnl
            self.balance_history.append(total_value)

        # Calculate final results
        final_value = self.balance_history[-1]
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100

        # Calculate performance metrics
        returns = np.diff(np.array(self.balance_history)) / np.array(self.balance_history[:-1])
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        total_trades = len([t for t in self.trades if t['type'] in ['buy', 'short']])
        winning_trades = len([t for t in self.trades if t.get('profit_loss', 0) > 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'balance_history': self.balance_history,
            'trades': self.trades
        }

    def rsi_strategy(self, rsi_overbought: int = 70, rsi_oversold: int = 30) -> Dict:
        """
        RSI-based mean reversion strategy

        Args:
            rsi_overbought: RSI level to trigger sells
            rsi_oversold: RSI level to trigger buys

        Returns:
            Dictionary with strategy results
        """
        def buy_signal():
            rsi_values = rsi(self.data['Close'].values, period=14)
            return np.where(rsi_values < rsi_oversold, 1, 0)

        def sell_signal():
            rsi_values = rsi(self.data['Close'].values, period=14)
            return np.where(rsi_values > rsi_overbought, 1, 0)

        return self.create_strategy(buy_signal, sell_signal)

    def macd_strategy(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict:
        """
        MACD crossover strategy

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Dictionary with strategy results
        """
        def buy_signal():
            macd_line, signal_line, _ = macd(
                self.data['Close'].values,
                fast_period,
                slow_period,
                signal_period
            )
            return np.where(macd_line > signal_line, 1, 0)

        def sell_signal():
            macd_line, signal_line, _ = macd(
                self.data['Close'].values,
                fast_period,
                slow_period,
                signal_period
            )
            return np.where(macd_line < signal_line, 1, 0)

        return self.create_strategy(buy_signal, sell_signal)

    def bollinger_strategy(self, period: int = 20, dev_up: float = 2.0, dev_down: float = 2.0) -> Dict:
        """
        Bollinger Bands mean reversion strategy

        Args:
            period: Bollinger Band period
            dev_up: Upper band standard deviations
            dev_down: Lower band standard deviations

        Returns:
            Dictionary with strategy results
        """
        upper, middle, lower = bollinger_bands(
            self.data['Close'].values,
            period,
            dev_up,
            dev_down
        )

        def buy_signal():
            return np.where(self.data['Close'].values < lower, 1, 0)

        def sell_signal():
            return np.where(self.data['Close'].values > upper, 1, 0)

        return self.create_strategy(buy_signal, sell_signal)

    def pattern_strategy(self) -> Dict:
        """
        Candlestick pattern-based strategy

        Returns:
            Dictionary with strategy results
        """
        # Use hammer patterns for buys and shooting stars for sells
        def buy_signal():
            return hammer(
                self.data['Open'].values,
                self.data['High'].values,
                self.data['Low'].values,
                self.data['Close'].values
            )

        def sell_signal():
            return shooting_star(
                self.data['Open'].values,
                self.data['High'].values,
                self.data['Low'].values,
                self.data['Close'].values
            )

        return self.create_strategy(buy_signal, sell_signal)


# Demo usage example
def demo_backtester():
    """
    Demonstrate the backtesting framework with sample data
    """
    # Create sample OHLCV data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)

    # Generate realistic looking price data
    close_prices = 100 + np.cumsum(np.random.randn(200) * 2)

    # Create OHLCV data
    high = close_prices + np.abs(np.random.randn(200))
    low = close_prices - np.abs(np.random.randn(200))
    open_prices = close_prices + np.random.randn(200) * 0.5
    volume = np.random.randint(1000, 10000, 200)

    # Ensure high >= close >= low and high >= open >= low
    for i in range(len(high)):
        open_prices[i] = np.clip(open_prices[i], low[i], high[i])
        close_prices[i] = np.clip(close_prices[i], low[i], high[i])
        high[i] = max(high[i], open_prices[i], close_prices[i])
        low[i] = min(low[i], open_prices[i], close_prices[i])

    # Create DataFrame
    data = pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)

    # Initialize backtester
    bt = SimpleBacktester(data, initial_capital=10000)

    print("py-TIM Library - Backtesting Demo")
    print("=" * 50)

    # Test RSI strategy
    print("\n1. RSI Mean Reversion Strategy:")
    rsi_results = bt.rsi_strategy()
    print(f"Initial Capital: ${rsi_results['initial_capital']:.2f}")
    print(f"Final Value: ${rsi_results['final_value']:.2f}")
    print(f"Total Return: {rsi_results['total_return_pct']:.1f}%")
    # Reset for next strategy
    bt = SimpleBacktester(data, initial_capital=10000)

    # Test MACD strategy
    print("\n2. MACD Crossover Strategy:")
    macd_results = bt.macd_strategy()
    print(f"Initial Capital: ${macd_results['initial_capital']:.2f}")
    print(f"Final Value: ${macd_results['final_value']:.2f}")
    print(f"Total Return: {macd_results['total_return_pct']:.1f}%")
    # Reset for next strategy
    bt = SimpleBacktester(data, initial_capital=10000)

    # Test Bollinger Bands strategy
    print("\n3. Bollinger Bands Mean Reversion:")
    bb_results = bt.bollinger_strategy()
    print(f"Initial Capital: ${bb_results['initial_capital']:.2f}")
    print(f"Final Value: ${bb_results['final_value']:.2f}")
    print(f"Total Return: {bb_results['total_return_pct']:.1f}%")
    print("\nBacktesting Demo Complete!")
    print("The py-TIM library successfully powers algorithmic trading strategies!")


if __name__ == "__main__":
    demo_backtester()
