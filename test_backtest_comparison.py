from connection import AlpacaConnection
from strategies.backtest_value_averaging import BacktestValueAveragingStrategy
from strategies.backtest_dca import BacktestDCAStrategy
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

def plot_comparison(va_results, dca_results, symbol):
    """Plot comparison of VA and DCA strategies"""
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot portfolio values
        ax1.plot(va_results['results']['date'], va_results['results']['portfolio_value'], 
                label='Value Averaging', color='blue')
        ax1.plot(dca_results['results']['date'], dca_results['results']['portfolio_value'], 
                label='Dollar Cost Averaging', color='green')
        ax1.set_title(f'Portfolio Value Comparison - {symbol}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot trades
        if not va_results['trades'].empty:
            ax2.scatter(va_results['trades']['date'], va_results['trades']['price'], 
                       color='blue', label='VA Trades', alpha=0.6)
        if not dca_results['trades'].empty:
            ax2.scatter(dca_results['trades']['date'], dca_results['trades']['price'], 
                       color='green', label='DCA Trades', alpha=0.6)
        ax2.plot(va_results['results']['date'], va_results['results']['portfolio_value'], 
                color='blue', alpha=0.3, label='VA Portfolio Value')
        ax2.plot(dca_results['results']['date'], dca_results['results']['portfolio_value'], 
                color='green', alpha=0.3, label='DCA Portfolio Value')
        ax2.set_title(f'Trade Points Comparison - {symbol}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price ($)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save plot
        plt.savefig(f'results/strategy_comparison_{symbol}.png')
        logger.info(f"📊 Saved strategy comparison plot to 'results/strategy_comparison_{symbol}.png'")
        
    except Exception as e:
        logger.error(f"❌ Error plotting comparison: {str(e)}")

def run_strategy_comparison(symbol, alpaca, start_date, end_date):
    """Run strategy comparison for a single symbol"""
    try:
        # Initialize strategies
        logger.info(f"📈 Initializing strategies for {symbol}...")
        
        # Adjust weekly investment based on symbol
        weekly_investment = 1000 if symbol == 'SPY' else 500  # $1000 for SPY, $500 for TSLA
        
        va_strategy = BacktestValueAveragingStrategy(
            api=alpaca.api,
            symbol=symbol,
            weekly_target=weekly_investment,  # Increased weekly target
            initial_capital=10000  # Start with $10,000
        )
        
        dca_strategy = BacktestDCAStrategy(
            api=alpaca.api,
            symbol=symbol,
            weekly_investment=weekly_investment,  # Increased weekly investment
            initial_capital=10000  # Start with $10,000
        )
        
        # Run backtests
        logger.info(f"🚀 Running backtests for {symbol} from {start_date.date()} to {end_date.date()}...")
        va_results = va_strategy.run_backtest(start_date, end_date)
        dca_results = dca_strategy.run_backtest(start_date, end_date)
        
        if va_results is None or dca_results is None:
            logger.error(f"❌ One or both backtests failed for {symbol}")
            return False
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Plot comparison
        plot_comparison(va_results, dca_results, symbol)
        
        # Save detailed results to CSV
        va_results['results'].to_csv(f'results/va_portfolio_values_{symbol}.csv')
        va_results['trades'].to_csv(f'results/va_trades_{symbol}.csv')
        dca_results['results'].to_csv(f'results/dca_portfolio_values_{symbol}.csv')
        dca_results['trades'].to_csv(f'results/dca_trades_{symbol}.csv')
        logger.info(f"💾 Saved detailed results to CSV files for {symbol}")
        
        # Print comparison metrics
        va_metrics = va_results['metrics']
        dca_metrics = dca_results['metrics']
        
        logger.info(f"\n📊 Strategy Comparison for {symbol}:")
        logger.info("\nValue Averaging:")
        logger.info(f"Final Portfolio Value: ${va_metrics['final_value']:,.2f}")
        logger.info(f"Total Return: {va_metrics['total_return']:.2%}")
        logger.info(f"Annual Return: {va_metrics['annual_return']:.2%}")
        logger.info(f"Sharpe Ratio: {va_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Number of Trades: {va_metrics['num_trades']}")
        logger.info(f"Total Investment: ${va_metrics.get('total_investment', 0):,.2f}")
        
        logger.info("\nDollar Cost Averaging:")
        logger.info(f"Final Portfolio Value: ${dca_metrics['final_value']:,.2f}")
        logger.info(f"Total Return: {dca_metrics['total_return']:.2%}")
        logger.info(f"Annual Return: {dca_metrics['annual_return']:.2%}")
        logger.info(f"Sharpe Ratio: {dca_metrics['sharpe_ratio']:.2f}")
        logger.info(f"Number of Trades: {dca_metrics['num_trades']}")
        logger.info(f"Total Investment: ${dca_metrics.get('total_investment', 0):,.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"⚠️ Unexpected error for {symbol}: {str(e)}")
        return False

def test_strategies():
    """Test and compare VA and DCA strategies for multiple symbols"""
    # Initialize connection
    alpaca = AlpacaConnection()
    
    try:
        # Connect to Alpaca
        logger.info("🔄 Testing Alpaca connection...")
        if not alpaca.connect():
            logger.error("❌ Failed to connect to Alpaca")
            return False
        
        # Set backtest parameters - use historical dates
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=180)  # Test for 6 months
        
        # Adjust dates to ensure we're using historical data
        if start_date > end_date:
            logger.error("❌ Start date cannot be in the future")
            return False
        
        # List of symbols to test
        symbols = ['SPY', 'TSLA']
        
        # Run comparison for each symbol
        for symbol in symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running comparison for {symbol}")
            logger.info(f"{'='*50}\n")
            run_strategy_comparison(symbol, alpaca, start_date, end_date)
        
        return True
        
    except Exception as e:
        logger.error(f"⚠️ Unexpected error: {str(e)}")
        return False
    finally:
        # Clean up
        if alpaca.is_connected():
            alpaca.disconnect()

if __name__ == "__main__":
    test_strategies() 