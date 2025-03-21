#!/usr/bin/env python3
"""
Alpha Trader - Main executor for alpha-generating strategies
Run multiple alpha strategies in parallel and manage the overall portfolio.
"""

import os
import logging
import argparse
import time
import threading
import json
from datetime import datetime
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from connection import AlpacaConnection
from strategies.alpha_ml_strategy import AlphaMLStrategy
from strategies.sentiment_alpha_strategy import SentimentAlphaStrategy
from strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from config import (
    API_KEY, 
    API_SECRET, 
    BASE_URL, 
    MAX_POSITION_SIZE, 
    STOP_LOSS_PERCENTAGE, 
    TAKE_PROFIT_PERCENTAGE, 
    MAX_DAILY_LOSS, 
    MAX_POSITION_SIZE_PERCENT, 
    MAX_OPEN_POSITIONS,
    LOG_LEVEL,
    LOG_FILE
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlphaTrader:
    """
    Main trader application that runs multiple alpha strategies and manages the overall portfolio.
    """
    def __init__(self, symbols, strategies_config=None):
        """
        Initialize the AlphaTrader.
        
        Args:
            symbols (list): List of ticker symbols to trade
            strategies_config (dict): Configuration for each strategy
        """
        self.symbols = symbols
        self.strategies_config = strategies_config or {}
        self.connection = AlpacaConnection()
        self.api = None
        self.strategy_threads = {}
        self.active_strategies = {}
        self.stop_event = threading.Event()
        
        # Strategy performance tracking
        self.strategy_performance = {}
        
        # Overall portfolio stats
        self.portfolio_value_history = []
        self.portfolio_returns = []
        self.drawdowns = []
        
    def connect(self):
        """Establish connection to the trading API"""
        if self.connection.connect():
            self.api = self.connection.api
            logger.info("Successfully connected to trading API")
            return True
        else:
            logger.error("Failed to connect to trading API")
            return False
            
    def disconnect(self):
        """Disconnect from the trading API"""
        if self.connection:
            self.connection.disconnect()
            logger.info("Disconnected from trading API")
            
    def initialize_strategies(self):
        """Initialize trading strategies based on configuration"""
        for symbol in self.symbols:
            if not self.api:
                logger.error("API connection not established")
                return False
                
            # Create strategies for this symbol based on config
            symbol_strategies = []
            
            # ML Strategy
            if self.strategies_config.get('ml', {}).get('enabled', True):
                ml_config = self.strategies_config.get('ml', {})
                lookback = ml_config.get('lookback_period', 30)
                prediction_horizon = ml_config.get('prediction_horizon', 5)
                
                # Create model directory if it doesn't exist
                os.makedirs('models', exist_ok=True)
                model_path = os.path.join('models', f"{symbol}_alpha_ml_model.joblib")
                
                ml_strategy = AlphaMLStrategy(
                    self.api, 
                    symbol, 
                    lookback_period=lookback,
                    prediction_horizon=prediction_horizon,
                    model_path=model_path
                )
                symbol_strategies.append(('ml', ml_strategy))
                
            # Sentiment Strategy
            if self.strategies_config.get('sentiment', {}).get('enabled', True):
                sentiment_config = self.strategies_config.get('sentiment', {})
                lookback = sentiment_config.get('sentiment_lookback', 3)
                
                sentiment_strategy = SentimentAlphaStrategy(
                    self.api,
                    symbol,
                    sentiment_lookback=lookback
                )
                symbol_strategies.append(('sentiment', sentiment_strategy))
                
            # Moving Average Crossover Strategy
            if self.strategies_config.get('ma_crossover', {}).get('enabled', True):
                ma_config = self.strategies_config.get('ma_crossover', {})
                short_window = ma_config.get('short_window', 20)
                long_window = ma_config.get('long_window', 50)
                
                ma_strategy = MovingAverageCrossoverStrategy(
                    self.api,
                    symbol,
                    short_window=short_window,
                    long_window=long_window
                )
                symbol_strategies.append(('ma_crossover', ma_strategy))
                
            # Store strategies
            if symbol_strategies:
                self.active_strategies[symbol] = symbol_strategies
                logger.info(f"Initialized {len(symbol_strategies)} strategies for {symbol}")
            else:
                logger.warning(f"No strategies created for {symbol}")
                
        return len(self.active_strategies) > 0
        
    def start_strategy(self, symbol, strategy_type, strategy):
        """Start a strategy in a separate thread"""
        thread_name = f"{symbol}_{strategy_type}"
        strategy_thread = threading.Thread(
            target=self._run_strategy,
            args=(strategy, thread_name),
            name=thread_name,
            daemon=True
        )
        self.strategy_threads[thread_name] = strategy_thread
        strategy_thread.start()
        logger.info(f"Started {strategy_type} strategy for {symbol}")
        
    def _run_strategy(self, strategy, thread_name):
        """Run the strategy in a loop until stopped"""
        try:
            # Train ML model if needed
            if hasattr(strategy, 'train_model') and callable(getattr(strategy, 'train_model')):
                train_first = self.strategies_config.get('ml', {}).get('train_first', True)
                if train_first and not strategy.model:
                    logger.info(f"Training ML model for {thread_name}")
                    if hasattr(strategy, 'run'):
                        strategy.run(train_first=True)
                        return  # Let the run method handle the loop
            
            # For other strategies, use their run method
            if hasattr(strategy, 'run'):
                strategy.run()
            else:
                logger.error(f"Strategy for {thread_name} has no run method")
        except Exception as e:
            logger.error(f"Error in strategy thread {thread_name}: {str(e)}")
            
    def start_all_strategies(self):
        """Start all initialized strategies"""
        for symbol, strategies in self.active_strategies.items():
            for strategy_type, strategy in strategies:
                self.start_strategy(symbol, strategy_type, strategy)
                
        return len(self.strategy_threads) > 0
        
    def stop_all_strategies(self):
        """Stop all running strategies"""
        self.stop_event.set()
        
        # Give strategies time to clean up
        time.sleep(5)
        
        # Check if any threads are still alive
        active_threads = [name for name, thread in self.strategy_threads.items() if thread.is_alive()]
        if active_threads:
            logger.warning(f"Some strategy threads are still active: {active_threads}")
            
        logger.info("All strategies stopped")
        
    def monitor_portfolio(self):
        """Monitor portfolio performance and risk"""
        while not self.stop_event.is_set():
            try:
                account_info = self.connection.get_account_info()
                positions = self.connection.get_positions()
                
                if not account_info:
                    logger.error("Failed to get account information")
                    time.sleep(60)
                    continue
                    
                # Update portfolio history
                timestamp = datetime.now()
                portfolio_value = account_info['portfolio_value']
                self.portfolio_value_history.append((timestamp, portfolio_value))
                
                # Calculate daily return if we have history
                if len(self.portfolio_value_history) > 1:
                    prev_value = self.portfolio_value_history[-2][1]
                    daily_return = (portfolio_value - prev_value) / prev_value
                    self.portfolio_returns.append((timestamp, daily_return))
                    
                    # Check for max daily loss
                    if daily_return < -MAX_DAILY_LOSS:
                        logger.warning(f"Max daily loss threshold exceeded: {daily_return:.2%}")
                        # Implement risk management actions here
                        
                # Calculate drawdown
                if self.portfolio_value_history:
                    peak = max([value for _, value in self.portfolio_value_history])
                    current_drawdown = (peak - portfolio_value) / peak
                    self.drawdowns.append((timestamp, current_drawdown))
                    
                    if current_drawdown > 0.1:  # 10% drawdown warning
                        logger.warning(f"Portfolio in significant drawdown: {current_drawdown:.2%}")
                        
                # Check position limits
                if positions:
                    if len(positions) > MAX_OPEN_POSITIONS:
                        logger.warning(f"Too many open positions: {len(positions)} (max: {MAX_OPEN_POSITIONS})")
                        # Implement position reduction logic here
                        
                    # Check individual position sizes
                    for position in positions:
                        position_value = float(position['market_value'])
                        position_pct = position_value / portfolio_value
                        
                        if position_pct > MAX_POSITION_SIZE_PERCENT:
                            logger.warning(f"Position {position['symbol']} too large: {position_pct:.2%} of portfolio")
                            # Implement position reduction logic here
                            
                # Save performance data periodically
                self._save_performance_data()
                
                # Sleep for a minute before checking again
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error monitoring portfolio: {str(e)}")
                time.sleep(60)
                
    def _save_performance_data(self):
        """Save performance data to disk"""
        try:
            # Create performance directory if needed
            os.makedirs('performance', exist_ok=True)
            
            # Portfolio value history
            if self.portfolio_value_history:
                pd.DataFrame(
                    self.portfolio_value_history, 
                    columns=['timestamp', 'portfolio_value']
                ).to_csv('performance/portfolio_value.csv', index=False)
                
            # Portfolio returns
            if self.portfolio_returns:
                pd.DataFrame(
                    self.portfolio_returns,
                    columns=['timestamp', 'return']
                ).to_csv('performance/portfolio_returns.csv', index=False)
                
            # Drawdowns
            if self.drawdowns:
                pd.DataFrame(
                    self.drawdowns,
                    columns=['timestamp', 'drawdown']
                ).to_csv('performance/drawdowns.csv', index=False)
                
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")
            
    def run(self):
        """Main trading loop"""
        try:
            # Connect to API
            if not self.connect():
                logger.error("Failed to connect. Exiting.")
                return False
                
            # Initialize strategies
            if not self.initialize_strategies():
                logger.error("Failed to initialize strategies. Exiting.")
                return False
                
            # Start all strategies
            if not self.start_all_strategies():
                logger.error("Failed to start strategies. Exiting.")
                return False
                
            # Start portfolio monitoring in a separate thread
            monitor_thread = threading.Thread(
                target=self.monitor_portfolio,
                name="portfolio_monitor",
                daemon=True
            )
            monitor_thread.start()
            
            logger.info("AlphaTrader running. Press Ctrl+C to stop.")
            
            # Keep main thread alive
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received stop signal. Shutting down...")
            self.stop_all_strategies()
            self.disconnect()
            return True
            
        except Exception as e:
            logger.error(f"Error in AlphaTrader main loop: {str(e)}")
            self.stop_all_strategies()
            self.disconnect()
            return False
            
def load_config(config_file):
    """Load configuration from a JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file {config_file}: {str(e)}")
        return None
        
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Alpha Trader - Run alpha-generating strategies')
    parser.add_argument('--symbols', type=str, required=True, help='Comma-separated list of symbols to trade')
    parser.add_argument('--config', type=str, default='alpha_config.json', help='Path to configuration file')
    
    return parser.parse_args()
    
def main():
    """Main entry point"""
    args = parse_args()
    
    # Parse symbols
    symbols = [sym.strip() for sym in args.symbols.split(',')]
    
    # Load configuration
    config = load_config(args.config)
    
    if not config:
        # Use default configuration
        config = {
            'ml': {
                'enabled': True,
                'lookback_period': 30,
                'prediction_horizon': 5,
                'train_first': True
            },
            'sentiment': {
                'enabled': True,
                'sentiment_lookback': 3
            },
            'ma_crossover': {
                'enabled': True,
                'short_window': 20,
                'long_window': 50
            }
        }
        
    # Initialize and run the trader
    trader = AlphaTrader(symbols, config)
    trader.run()
    
if __name__ == "__main__":
    main() 