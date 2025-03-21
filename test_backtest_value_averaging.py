from connection import AlpacaConnection
from strategies.backtest_value_averaging import BacktestValueAveragingStrategy
import logging
from config import LOG_LEVEL, LOG_FILE
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import os

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

def plot_results(results, trades):
    """Plot backtest results"""
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot portfolio value
        ax1.plot(results['date'], results['portfolio_value'], label='Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot trades
        if not trades.empty:
            ax2.scatter(trades['date'], trades['price'], color='green', label='Buy Trades')
            ax2.plot(results['date'], results['portfolio_value'], color='blue', alpha=0.3, label='Portfolio Value')
            ax2.set_title('Trade Points')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price ($)')
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save plot
        plt.savefig('results/backtest_results.png')
        logger.info("📊 Saved backtest results plot to 'results/backtest_results.png'")
        
    except Exception as e:
        logger.error(f"❌ Error plotting results: {str(e)}")

def test_backtest():
    """Test the Value Averaging strategy backtest"""
    # Initialize connection
    alpaca = AlpacaConnection()
    
    try:
        # Connect to Alpaca
        logger.info("🔄 Testing Alpaca connection...")
        if not alpaca.connect():
            logger.error("❌ Failed to connect to Alpaca")
            return False
        
        # Set backtest parameters - using historical data
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=90)  # Test for 3 months
        
        # Adjust dates to ensure we're using historical data
        if start_date > end_date:
            logger.error("❌ Start date cannot be in the future")
            return False
            
        # Initialize strategy
        logger.info("📈 Initializing Value Averaging strategy backtest for PLTR...")
        strategy = BacktestValueAveragingStrategy(
            api=alpaca.api,
            symbol='PLTR',
            weekly_target=100,  # Target $100 investment per week
            initial_capital=10000  # Start with $10,000
        )
        
        # Run backtest
        logger.info(f"🚀 Running backtest from {start_date.date()} to {end_date.date()}...")
        results = strategy.run_backtest(start_date, end_date)
        
        if results is None:
            logger.error("❌ Backtest failed")
            return False
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Plot results
        plot_results(results['results'], results['trades'])
        
        # Save detailed results to CSV
        results['results'].to_csv('results/backtest_portfolio_values.csv')
        results['trades'].to_csv('results/backtest_trades.csv')
        logger.info("💾 Saved detailed results to CSV files")
        
        # Print additional metrics
        metrics = results['metrics']
        logger.info("\n📈 Additional Performance Metrics:")
        logger.info(f"Average Trade Size: ${metrics['final_value'] / metrics['num_trades']:.2f}")
        logger.info(f"Total Investment: ${metrics['final_value'] - metrics['initial_capital']:.2f}")
        logger.info(f"Return per Trade: {metrics['total_return'] / metrics['num_trades']:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"⚠️ Unexpected error: {str(e)}")
        return False
    finally:
        # Clean up
        if alpaca.is_connected():
            alpaca.disconnect()

if __name__ == "__main__":
    test_backtest() 