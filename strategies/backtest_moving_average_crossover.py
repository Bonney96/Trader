import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import LOG_LEVEL, LOG_FILE, API_KEY, API_SECRET

logger = logging.getLogger(__name__)

class BacktestMovingAverageCrossoverStrategy:
    def __init__(self, api, symbol, short_window=20, long_window=50, initial_capital=100000):
        self.api = api
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.trades = []
        self.portfolio_values = []
        self.dates = []

    def get_historical_data(self, start_date, end_date):
        """Get historical data for backtesting"""
        try:
            # Ensure we're not using future dates
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            if end_date > current_date:
                end_date = current_date
                logger.info(f"⚠️ Adjusted end date to current date: {end_date.date()}")
            
            if start_date > end_date:
                logger.error("❌ Start date cannot be after end date")
                return None
            
            # Format dates as YYYY-MM-DD strings
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            logger.info(f"📊 Fetching historical data from {start_str} to {end_str}")
            
            # Get historical data using the data API
            # Use credentials directly from config
            data_api = tradeapi.REST(
                API_KEY,
                API_SECRET,
                base_url='https://data.alpaca.markets'
            )
            
            bars = data_api.get_bars(
                self.symbol,
                tradeapi.TimeFrame.Day,
                start=start_str,
                end=end_str,
                adjustment='raw'  # Use raw data without adjustments
            ).df
            
            if bars.empty:
                logger.error("❌ No historical data found for the specified period")
                return None
                
            logger.info(f"✅ Successfully fetched {len(bars)} days of historical data")
            return bars
            
        except Exception as e:
            logger.error(f"❌ Error fetching historical data: {str(e)}")
            return None

    def calculate_signals(self, df):
        """Calculate trading signals based on moving averages"""
        if df is None or len(df) < self.long_window:
            return None

        # Calculate moving averages
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()

        # Generate signals
        df['signal'] = 0
        df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
        df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1

        return df

    def execute_trade(self, signal, current_price):
        """Simulate trade execution"""
        try:
            if signal == 1 and self.position <= 0:  # Buy signal
                # Calculate position size (use 20% of capital)
                position_size = int((self.current_capital * 0.2) / current_price)
                if position_size < 1:
                    logger.info("ℹ️ Not enough capital to buy shares.")
                    return

                # Update position and capital
                self.position = position_size
                self.current_capital -= position_size * current_price
                
                # Record trade
                self.trades.append({
                    'date': self.current_date,
                    'price': current_price,
                    'shares': position_size,
                    'investment': position_size * current_price,
                    'position': self.position,
                    'portfolio_value': self.position * current_price + self.current_capital,
                    'type': 'buy'
                })
                
                logger.info(f"✅ Simulated buy: {position_size} shares at ${current_price:.2f}")
                logger.info(f"💼 New position: {self.position} shares")
                logger.info(f"💰 Remaining capital: ${self.current_capital:.2f}")

            elif signal == -1 and self.position >= 0:  # Sell signal
                if self.position > 0:
                    # Update position and capital
                    self.current_capital += self.position * current_price
                    
                    # Record trade
                    self.trades.append({
                        'date': self.current_date,
                        'price': current_price,
                        'shares': -self.position,
                        'investment': -self.position * current_price,
                        'position': 0,
                        'portfolio_value': self.current_capital,
                        'type': 'sell'
                    })
                    
                    logger.info(f"✅ Simulated sell: {self.position} shares at ${current_price:.2f}")
                    logger.info(f"💼 New position: 0 shares")
                    logger.info(f"💰 Remaining capital: ${self.current_capital:.2f}")
                    
                    self.position = 0

        except Exception as e:
            logger.error(f"❌ Error executing trade: {str(e)}")

    def run_backtest(self, start_date, end_date):
        """Run backtest over specified date range"""
        try:
            # Get historical data
            historical_data = self.get_historical_data(start_date, end_date)
            if historical_data is None:
                return None

            # Initialize results
            self.portfolio_values = []
            self.dates = []
            self.trades = []
            self.position = 0
            self.current_capital = self.initial_capital

            # Calculate signals
            historical_data = self.calculate_signals(historical_data)
            if historical_data is None:
                return None

            # Process each day
            for date, row in historical_data.iterrows():
                self.current_date = date
                current_price = row['close']
                current_signal = row['signal']

                # Execute trade based on signal
                self.execute_trade(current_signal, current_price)

                # Record daily portfolio value
                portfolio_value = (self.position * current_price) + self.current_capital
                self.portfolio_values.append(portfolio_value)
                self.dates.append(date)

            # Create results DataFrame
            results = pd.DataFrame({
                'date': self.dates,
                'portfolio_value': self.portfolio_values
            })

            # Calculate performance metrics
            initial_value = self.initial_capital
            final_value = self.portfolio_values[-1]
            total_return = (final_value - initial_value) / initial_value
            annual_return = (1 + total_return) ** (252 / len(results)) - 1
            daily_returns = results['portfolio_value'].pct_change()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()

            logger.info("\n📊 Backtest Results:")
            logger.info(f"Initial Capital: ${initial_value:,.2f}")
            logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
            logger.info(f"Total Return: {total_return:.2%}")
            logger.info(f"Annual Return: {annual_return:.2%}")
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"Number of Trades: {len(self.trades)}")

            return {
                'results': results,
                'trades': pd.DataFrame(self.trades),
                'metrics': {
                    'initial_capital': initial_value,
                    'final_value': final_value,
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe_ratio,
                    'num_trades': len(self.trades)
                }
            }

        except Exception as e:
            logger.error(f"❌ Error in backtest: {str(e)}")
            return None 