import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from config import MAX_POSITION_SIZE, STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE

logger = logging.getLogger(__name__)

class MovingAverageCrossoverStrategy:
    def __init__(self, api, symbol, short_window=20, long_window=50):
        self.api = api
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.position = 0
        self.contract = None

    def is_market_open(self):
        """Check if the market is currently open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False

    def get_historical_data(self):
        """Get historical data for the symbol"""
        try:
            # Get 1-minute bars for the last 100 minutes
            bars = self.api.get_bars(
                self.symbol,
                tradeapi.TimeFrame.Minute,
                limit=100
            ).df
            
            if bars.empty:
                logger.error(f"No historical data received for {self.symbol}")
                return None

            return bars
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
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

    def wait_for_order_fill(self, order_id, timeout=30):
        """Wait for an order to be filled"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            order = self.api.get_order(order_id)
            if order.status == 'filled':
                return order
            elif order.status in ['canceled', 'expired', 'rejected']:
                raise Exception(f"Order {order_id} {order.status}")
            time.sleep(1)
        raise Exception(f"Order {order_id} not filled within {timeout} seconds")

    def execute_trade(self, signal, current_price):
        """Execute trade based on signal"""
        try:
            # Check if market is open
            if not self.is_market_open():
                logger.info("Market is closed. Skipping trade execution.")
                return

            if signal == 1 and self.position <= 0:  # Buy signal
                quantity = min(MAX_POSITION_SIZE, self.calculate_position_size(current_price))
                
                # Place market buy order
                order = self.api.submit_order(
                    symbol=self.symbol,
                    qty=quantity,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                
                # Wait for order to be filled
                filled_order = self.wait_for_order_fill(order.id)
                filled_price = float(filled_order.filled_avg_price)
                
                self.position = quantity
                logger.info(f"Buy order filled for {quantity} shares of {self.symbol} at ${filled_price:.2f}")

                # Set stop loss and take profit after order is filled
                self.set_stop_loss_take_profit(filled_price)

            elif signal == -1 and self.position >= 0:  # Sell signal
                if self.position > 0:
                    # Place market sell order
                    order = self.api.submit_order(
                        symbol=self.symbol,
                        qty=self.position,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    
                    # Wait for order to be filled
                    filled_order = self.wait_for_order_fill(order.id)
                    filled_price = float(filled_order.filled_avg_price)
                    
                    self.position = -self.position
                    logger.info(f"Sell order filled for {self.position} shares of {self.symbol} at ${filled_price:.2f}")

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")

    def calculate_position_size(self, current_price):
        """Calculate position size based on account value"""
        try:
            account = self.api.get_account()
            account_value = float(account.portfolio_value)
            return int(account_value * 0.02 / current_price)  # Use 2% of account value
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0

    def set_stop_loss_take_profit(self, entry_price):
        """Set stop loss and take profit orders"""
        try:
            # Stop Loss - round to 2 decimal places
            stop_price = round(entry_price * (1 - STOP_LOSS_PERCENTAGE), 2)
            stop_order = self.api.submit_order(
                symbol=self.symbol,
                qty=self.position,
                side='sell',
                type='stop',
                stop_price=stop_price,
                time_in_force='gtc'
            )

            # Take Profit - round to 2 decimal places
            take_profit_price = round(entry_price * (1 + TAKE_PROFIT_PERCENTAGE), 2)
            take_profit_order = self.api.submit_order(
                symbol=self.symbol,
                qty=self.position,
                side='sell',
                type='limit',
                limit_price=take_profit_price,
                time_in_force='gtc'
            )

            logger.info(f"Set stop loss at ${stop_price:.2f} and take profit at ${take_profit_price:.2f}")
        except Exception as e:
            logger.error(f"Error setting stop loss/take profit: {str(e)}")

    def run(self):
        """Main strategy loop"""
        while True:
            try:
                # Check if market is open
                if not self.is_market_open():
                    logger.info("Market is closed. Waiting for market open...")
                    time.sleep(60)  # Check every minute
                    continue

                # Get historical data
                df = self.get_historical_data()
                if df is None:
                    continue

                # Calculate signals
                df = self.calculate_signals(df)
                if df is None:
                    continue

                # Get latest signal
                current_signal = df['signal'].iloc[-1]
                current_price = df['close'].iloc[-1]

                # Execute trade based on signal
                self.execute_trade(current_signal, current_price)

                # Wait for next iteration
                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in strategy loop: {str(e)}")
                time.sleep(60)  # Wait before retrying 