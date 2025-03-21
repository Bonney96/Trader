import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import LOG_LEVEL, LOG_FILE, API_KEY, API_SECRET

logger = logging.getLogger(__name__)

class BacktestAdvancedStrategy:
    def __init__(self, api, symbol, initial_capital=100000):
        self.api = api
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        
        # Strategy parameters
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2
        self.volume_ma_period = 20
        self.atr_period = 14
        self.stop_loss_atr_multiplier = 2
        self.take_profit_atr_multiplier = 3
        self.max_position_size = 0.2  # Maximum 20% of capital per trade
        self.min_position_size = 0.05  # Minimum 5% of capital per trade

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
            # Use credentials directly from config instead of API object
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

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
            exp2 = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
            bb_std = df['close'].rolling(window=self.bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * self.bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std * self.bb_std)

            # Volume MA
            df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']

            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(window=self.atr_period).mean()

            return df

        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {str(e)}")
            return None

    def calculate_position_size(self, current_price, atr):
        """Calculate position size based on volatility"""
        try:
            # Calculate base position size (20% of capital)
            base_size = self.current_capital * self.max_position_size
            
            # Adjust position size based on ATR
            volatility_factor = 1 - (atr / current_price)  # Lower volatility = larger position
            volatility_factor = max(0.5, min(1.0, volatility_factor))  # Limit between 0.5 and 1.0
            
            # Calculate final position size
            position_size = base_size * volatility_factor
            
            # Ensure minimum position size
            min_size = self.current_capital * self.min_position_size
            position_size = max(min_size, position_size)
            
            # Calculate number of shares
            shares = int(position_size / current_price)
            
            return shares

        except Exception as e:
            logger.error(f"❌ Error calculating position size: {str(e)}")
            return 0

    def generate_signals(self, df):
        """Generate trading signals based on technical indicators"""
        try:
            df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
            
            # Buy conditions - require 2 out of 4 conditions to be met
            rsi_buy = df['rsi'] < self.rsi_oversold  # Oversold
            macd_buy = df['macd'] > df['macd_signal']  # MACD crossover
            bb_buy = df['close'] < df['bb_lower']  # Price below lower Bollinger Band
            volume_buy = df['volume_ratio'] > 1.2  # Above average volume
            
            # Count how many buy conditions are met
            buy_count = rsi_buy.astype(int) + macd_buy.astype(int) + bb_buy.astype(int) + volume_buy.astype(int)
            buy_conditions = buy_count >= 2  # Require at least 2 conditions
            
            # Sell conditions - require 2 out of 4 conditions to be met
            rsi_sell = df['rsi'] > self.rsi_overbought  # Overbought
            macd_sell = df['macd'] < df['macd_signal']  # MACD crossover
            bb_sell = df['close'] > df['bb_upper']  # Price above upper Bollinger Band
            volume_sell = df['volume_ratio'] > 1.2  # Above average volume
            
            # Count how many sell conditions are met
            sell_count = rsi_sell.astype(int) + macd_sell.astype(int) + bb_sell.astype(int) + volume_sell.astype(int)
            sell_conditions = sell_count >= 2  # Require at least 2 conditions
            
            # Set signals
            df.loc[buy_conditions, 'signal'] = 1
            df.loc[sell_conditions, 'signal'] = -1
            
            return df

        except Exception as e:
            logger.error(f"❌ Error generating signals: {str(e)}")
            return None

    def execute_trade(self, signal, current_price, atr):
        """Execute trade based on signal"""
        try:
            if signal == 1 and self.position <= 0:  # Buy signal
                # Calculate position size
                shares = self.calculate_position_size(current_price, atr)
                if shares < 1:
                    logger.info("ℹ️ Not enough capital to buy shares.")
                    return

                # Calculate stop loss and take profit levels
                stop_loss = current_price - (atr * self.stop_loss_atr_multiplier)
                take_profit = current_price + (atr * self.take_profit_atr_multiplier)

                # Update position and capital
                self.position = shares
                self.current_capital -= shares * current_price
                
                # Record trade
                self.trades.append({
                    'date': self.current_date,
                    'price': current_price,
                    'shares': shares,
                    'investment': shares * current_price,
                    'position': self.position,
                    'portfolio_value': self.position * current_price + self.current_capital,
                    'type': 'buy',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
                
                logger.info(f"✅ Simulated buy: {shares} shares at ${current_price:.2f}")
                logger.info(f"💼 New position: {self.position} shares")
                logger.info(f"💰 Remaining capital: ${self.current_capital:.2f}")
                logger.info(f"🛑 Stop Loss: ${stop_loss:.2f}")
                logger.info(f"🎯 Take Profit: ${take_profit:.2f}")

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

            # Calculate indicators
            historical_data = self.calculate_indicators(historical_data)
            if historical_data is None:
                return None

            # Generate signals
            historical_data = self.generate_signals(historical_data)
            if historical_data is None:
                return None

            # Initialize results
            self.portfolio_values = []
            self.dates = []
            self.trades = []
            self.position = 0
            self.current_capital = self.initial_capital
            
            # Track stop loss and take profit levels
            stop_loss = 0
            take_profit = 0
            entry_price = 0
            trailing_stop = 0
            highest_price = 0

            # Process each day
            for date, row in historical_data.iterrows():
                self.current_date = date
                current_price = row['close']
                current_signal = row['signal']
                current_atr = row['atr']
                
                # Check for stop loss or take profit if in a position
                sell_triggered = False
                if self.position > 0:
                    # Update trailing stop if price moves higher
                    if current_price > highest_price:
                        highest_price = current_price
                        # Set trailing stop to 2 ATR below the highest price
                        trailing_stop = highest_price - (current_atr * 1.5)
                        logger.info(f"🔄 Updated trailing stop to ${trailing_stop:.2f}")
                    
                    # Check if stop loss, take profit, or trailing stop is hit
                    if current_price <= stop_loss:
                        logger.info(f"🛑 Stop loss triggered at ${current_price:.2f}")
                        current_signal = -1
                        sell_triggered = True
                    elif current_price >= take_profit:
                        logger.info(f"🎯 Take profit triggered at ${current_price:.2f}")
                        current_signal = -1
                        sell_triggered = True
                    elif current_price <= trailing_stop and trailing_stop > 0:
                        logger.info(f"📉 Trailing stop triggered at ${current_price:.2f}")
                        current_signal = -1
                        sell_triggered = True

                # Execute trade based on signal
                if current_signal == 1 and self.position <= 0:
                    # Buy signal
                    shares = self.calculate_position_size(current_price, current_atr)
                    if shares < 1:
                        logger.info("ℹ️ Not enough capital to buy shares.")
                    else:
                        # Calculate stop loss and take profit levels
                        stop_loss = current_price - (current_atr * self.stop_loss_atr_multiplier)
                        take_profit = current_price + (current_atr * self.take_profit_atr_multiplier)
                        highest_price = current_price
                        trailing_stop = stop_loss
                        entry_price = current_price
                        
                        # Update position and capital
                        self.position = shares
                        self.current_capital -= shares * current_price
                        
                        # Record trade
                        self.trades.append({
                            'date': self.current_date,
                            'price': current_price,
                            'shares': shares,
                            'investment': shares * current_price,
                            'position': self.position,
                            'portfolio_value': self.position * current_price + self.current_capital,
                            'type': 'buy',
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        })
                        
                        logger.info(f"✅ Simulated buy: {shares} shares at ${current_price:.2f}")
                        logger.info(f"💼 New position: {self.position} shares")
                        logger.info(f"💰 Remaining capital: ${self.current_capital:.2f}")
                        logger.info(f"🛑 Stop Loss: ${stop_loss:.2f}")
                        logger.info(f"🎯 Take Profit: ${take_profit:.2f}")
                elif (current_signal == -1 or sell_triggered) and self.position > 0:
                    # Sell signal
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
                    stop_loss = 0
                    take_profit = 0
                    trailing_stop = 0
                    highest_price = 0

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

            logger.info("\n📊 Advanced Strategy Backtest Results:")
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