import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import LOG_LEVEL, LOG_FILE, API_KEY, API_SECRET

logger = logging.getLogger(__name__)

class BacktestDCAStrategy:
    def __init__(self, api, symbol, weekly_investment=100, initial_capital=10000):
        self.api = api
        self.symbol = symbol
        self.weekly_investment = weekly_investment
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.current_week = 0
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

    def execute_trade(self, current_price):
        """Simulate trade execution"""
        try:
            # Calculate number of shares to buy
            shares_to_buy = int(self.weekly_investment / current_price)
            if shares_to_buy < 1:
                logger.info("ℹ️ Weekly investment amount too small to buy any shares.")
                return

            # Update position and capital
            self.position += shares_to_buy
            self.current_capital -= shares_to_buy * current_price
            
            # Record trade
            self.trades.append({
                'date': self.current_date,
                'price': current_price,
                'shares': shares_to_buy,
                'investment': shares_to_buy * current_price,
                'position': self.position,
                'portfolio_value': self.position * current_price,
                'type': 'buy'  # DCA only buys
            })
            
            logger.info(f"✅ Simulated buy: {shares_to_buy} shares at ${current_price:.2f}")
            logger.info(f"💼 New position: {self.position} shares")
            logger.info(f"💰 Remaining capital: ${self.current_capital:.2f}")

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
            self.current_week = 0
            self.position = 0
            self.current_capital = self.initial_capital
            last_trade_date = None

            # Process each day
            for date, row in historical_data.iterrows():
                self.current_date = date
                current_price = row['close']

                # Check if we should trade (weekly)
                should_trade = False
                if last_trade_date is None:
                    should_trade = True
                    last_trade_date = date
                elif (date - last_trade_date).days >= 7:
                    should_trade = True
                    last_trade_date = date
                    self.current_week += 1

                # Execute trade if needed
                if should_trade:
                    self.execute_trade(current_price)

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

            logger.info("\n📊 DCA Backtest Results:")
            logger.info(f"Initial Capital: ${initial_value:,.2f}")
            logger.info(f"Final Portfolio Value: ${final_value:,.2f}")
            logger.info(f"Total Return: {total_return:.2%}")
            logger.info(f"Annual Return: {annual_return:.2%}")
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            logger.info(f"Number of Trades: {len(self.trades)}")
            logger.info(f"Total Investment: ${len(self.trades) * self.weekly_investment:,.2f}")

            return {
                'results': results,
                'trades': pd.DataFrame(self.trades),
                'metrics': {
                    'initial_capital': initial_value,
                    'final_value': final_value,
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe_ratio,
                    'num_trades': len(self.trades),
                    'total_investment': len(self.trades) * self.weekly_investment
                }
            }

        except Exception as e:
            logger.error(f"❌ Error in backtest: {str(e)}")
            return None 