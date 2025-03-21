import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from config import MAX_POSITION_SIZE

logger = logging.getLogger(__name__)

class ValueAveragingStrategy:
    def __init__(self, api, symbol, weekly_target=100, stop_loss_pct=0.1, take_profit_pct=0.2):
        self.api = api
        self.symbol = symbol
        self.weekly_target = weekly_target
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position = 0
        self.week_start_value = 0
        self.current_week = 0
        self.last_trade_date = None
        self.price_history = []
        self.max_price = 0
        self.entry_price = 0
        self.retry_count = 0
        self.max_retries = 3

    def is_market_open(self):
        """Check if the market is currently open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"❌ Error checking market status: {str(e)}")
            return False

    def get_current_price(self):
        """Get current price for the symbol with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                bars = self.api.get_latest_bar(self.symbol)
                price = float(bars.c)
                self.price_history.append(price)
                if len(self.price_history) > 20:  # Keep last 20 prices
                    self.price_history.pop(0)
                return price
            except Exception as e:
                self.retry_count += 1
                if attempt < self.max_retries - 1:
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed to get price. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"❌ Failed to get price after {self.max_retries} attempts: {str(e)}")
                    return None

    def get_portfolio_value(self):
        """Get current portfolio value for the symbol"""
        try:
            positions = self.api.list_positions()
            for position in positions:
                if position.symbol == self.symbol:
                    return float(position.market_value)
            return 0.0
        except Exception as e:
            logger.error(f"❌ Error getting portfolio value: {str(e)}")
            return 0.0

    def adjust_weekly_target(self):
        """Dynamically adjust weekly target based on price trend"""
        if len(self.price_history) < 2:
            return self.weekly_target

        # Calculate price trend
        price_trend = (self.price_history[-1] - self.price_history[0]) / self.price_history[0]
        
        if price_trend < -0.05:  # Price down 5% or more
            logger.info("📉 Price trending down. Increasing weekly target by 20%")
            return self.weekly_target * 1.2
        elif price_trend > 0.05:  # Price up 5% or more
            logger.info("📈 Price trending up. Decreasing weekly target by 20%")
            return self.weekly_target * 0.8
        
        return self.weekly_target

    def check_stop_loss_take_profit(self):
        """Check if stop loss or take profit conditions are met"""
        if not self.entry_price:
            return True  # Allow trading if no entry price set

        current_price = self.get_current_price()
        if not current_price:
            return True

        price_change = (current_price - self.entry_price) / self.entry_price

        if price_change <= -self.stop_loss_pct:
            logger.warning(f"⚠️ Stop loss triggered at {price_change:.2%}")
            return False
        elif price_change >= self.take_profit_pct:
            logger.info(f"✅ Take profit triggered at {price_change:.2%}")
            return False

        return True

    def calculate_investment_needed(self):
        """Calculate how much investment is needed to reach target"""
        try:
            current_value = self.get_portfolio_value()
            adjusted_target = self.adjust_weekly_target()
            target_value = adjusted_target * (self.current_week + 1)
            investment_needed = target_value - current_value
            
            logger.info(f"💼 Current portfolio value: ${current_value:.2f}")
            logger.info(f"🎯 Target value: ${target_value:.2f}")
            logger.info(f"💰 Investment needed: ${investment_needed:.2f}")
            
            return max(0, investment_needed)  # Don't invest if we're above target
        except Exception as e:
            logger.error(f"❌ Error calculating investment needed: {str(e)}")
            return 0

    def execute_trade(self, investment_amount):
        """Execute trade based on investment amount needed"""
        try:
            if not self.is_market_open():
                logger.info("🚫 Market is closed. Skipping trade execution.")
                return

            if investment_amount <= 0:
                logger.info("ℹ️ No investment needed at this time.")
                return

            if not self.check_stop_loss_take_profit():
                logger.info("⏸️ Trading paused due to stop loss or take profit conditions.")
                return

            current_price = self.get_current_price()
            if current_price is None:
                return

            # Calculate number of shares to buy
            shares_to_buy = int(investment_amount / current_price)
            if shares_to_buy < 1:
                logger.info("ℹ️ Investment amount too small to buy any shares.")
                return

            # Place market buy order
            order = self.api.submit_order(
                symbol=self.symbol,
                qty=shares_to_buy,
                side='buy',
                type='market',
                time_in_force='gtc'
            )

            # Wait for order to be filled
            filled_order = self.wait_for_order_fill(order.id)
            filled_price = float(filled_order.filled_avg_price)
            
            # Update entry price if this is the first trade
            if not self.entry_price:
                self.entry_price = filled_price
            
            logger.info(f"✅ Bought {shares_to_buy} shares of {self.symbol} at ${filled_price:.2f}")
            self.position += shares_to_buy

        except Exception as e:
            logger.error(f"❌ Error executing trade: {str(e)}")

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

    def should_trade_today(self):
        """Check if we should trade today"""
        try:
            current_date = datetime.now().date()
            
            # If this is our first trade, initialize the week
            if self.last_trade_date is None:
                self.last_trade_date = current_date
                self.current_week = 0
                return True
            
            # Check if we're in a new week
            if (current_date - self.last_trade_date).days >= 7:
                self.last_trade_date = current_date
                self.current_week += 1
                return True
            
            return False
        except Exception as e:
            logger.error(f"❌ Error checking trade timing: {str(e)}")
            return False

    def run(self):
        """Main strategy loop"""
        while True:
            try:
                # Check if we should trade today
                if not self.should_trade_today():
                    logger.info("⏳ Not time to trade yet. Waiting...")
                    time.sleep(60)  # Check every minute
                    continue

                # Calculate investment needed
                investment_needed = self.calculate_investment_needed()
                
                # Execute trade if needed
                if investment_needed > 0:
                    self.execute_trade(investment_needed)

                # Wait for next iteration
                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"❌ Error in strategy loop: {str(e)}")
                time.sleep(60)  # Wait before retrying 