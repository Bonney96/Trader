import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from strategies.backtest_dca import BacktestDCAStrategy
from strategies.backtest_value_averaging import BacktestValueAveragingStrategy
from strategies.backtest_moving_average_crossover import BacktestMovingAverageCrossoverStrategy
from strategies.backtest_advanced_strategy import BacktestAdvancedStrategy
from config import API_KEY, API_SECRET, BASE_URL, LOG_LEVEL, LOG_FILE
import os

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def plot_comparison(symbol, results_dict):
    """Plot comparison of all strategies"""
    plt.figure(figsize=(15, 10))
    
    # Plot portfolio values
    colors = ['blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'v']
    labels = ['DCA', 'Value Averaging', 'MAC', 'Advanced']
    
    for (strategy, results), color, marker, label in zip(results_dict.items(), colors, markers, labels):
        if results and 'results' in results:
            df = results['results']
            plt.plot(df['date'], df['portfolio_value'], color=color, label=label, alpha=0.7)
            
            # Plot trade points
            if 'trades' in results and not results['trades'].empty:
                trades_df = results['trades']
                buy_trades = trades_df[trades_df['type'] == 'buy']
                sell_trades = trades_df[trades_df['type'] == 'sell']
                
                if not buy_trades.empty:
                    plt.scatter(buy_trades['date'], buy_trades['portfolio_value'], 
                              color=color, marker='^', s=100, alpha=0.7, label=f'{label} Buy')
                
                if not sell_trades.empty:
                    plt.scatter(sell_trades['date'], sell_trades['portfolio_value'], 
                              color=color, marker='v', s=100, alpha=0.7, label=f'{label} Sell')
    
    plt.title(f'Strategy Comparison for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'results/strategy_comparison_{symbol}.png')
    
    # Create an additional plot showing just the advanced strategy with buy/sell points
    if 'Advanced' in results_dict and results_dict['Advanced']:
        plt.figure(figsize=(15, 10))
        
        results = results_dict['Advanced']
        df = results['results']
        plt.plot(df['date'], df['portfolio_value'], color='purple', label='Advanced Strategy', linewidth=2)
        
        # Plot trade points
        if 'trades' in results and not results['trades'].empty:
            trades_df = results['trades']
            buy_trades = trades_df[trades_df['type'] == 'buy']
            sell_trades = trades_df[trades_df['type'] == 'sell']
            
            if not buy_trades.empty:
                plt.scatter(buy_trades['date'], buy_trades['portfolio_value'], 
                          color='green', marker='^', s=120, label='Buy')
                
                # Add annotations for buy points
                for i, trade in buy_trades.iterrows():
                    plt.annotate(f"${trade['price']:.2f}", 
                               (trade['date'], trade['portfolio_value']),
                               textcoords="offset points",
                               xytext=(0,10), 
                               ha='center',
                               fontsize=8)
            
            if not sell_trades.empty:
                plt.scatter(sell_trades['date'], sell_trades['portfolio_value'], 
                          color='red', marker='v', s=120, label='Sell')
                
                # Add annotations for sell points
                for i, trade in sell_trades.iterrows():
                    plt.annotate(f"${trade['price']:.2f}", 
                               (trade['date'], trade['portfolio_value']),
                               textcoords="offset points",
                               xytext=(0,-15), 
                               ha='center',
                               fontsize=8)
        
        plt.title(f'Advanced Strategy Performance for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'results/advanced_strategy_{symbol}.png')
    
    plt.close('all')

def run_strategy_comparison(api, symbol, start_date, end_date):
    """Run comparison of all strategies for a given symbol"""
    logger.info(f"\n🔄 Running strategy comparison for {symbol}")
    
    # Initialize strategies
    strategies = {
        'DCA': BacktestDCAStrategy(api, symbol, weekly_investment=1000, initial_capital=100000),
        'VA': BacktestValueAveragingStrategy(api, symbol, weekly_target=1000, initial_capital=100000),
        'MAC': BacktestMovingAverageCrossoverStrategy(api, symbol, initial_capital=100000),
        'Advanced': BacktestAdvancedStrategy(api, symbol, initial_capital=100000)
    }
    
    results_dict = {}
    
    # Run each strategy
    for name, strategy in strategies.items():
        logger.info(f"\n📊 Running {name} strategy...")
        results = strategy.run_backtest(start_date, end_date)
        
        if results:
            results_dict[name] = results
            
            # Save results to CSV
            results['results'].to_csv(f'results/{name.lower()}_portfolio_values_{symbol}.csv', index=False)
            results['trades'].to_csv(f'results/{name.lower()}_trades_{symbol}.csv', index=False)
            
            # Log performance metrics
            metrics = results['metrics']
            logger.info(f"\n📈 {name} Strategy Performance:")
            logger.info(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
            logger.info(f"Total Return: {metrics['total_return']:.2%}")
            logger.info(f"Annual Return: {metrics['annual_return']:.2%}")
            logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"Number of Trades: {metrics['num_trades']}")
        else:
            logger.error(f"❌ Failed to run {name} strategy")
            results_dict[name] = None
    
    return results_dict

def test_strategies():
    """Test all strategies"""
    try:
        # Connect to Alpaca
        api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL)
        
        # Set backtest parameters - use fixed historical dates for 2 years
        end_date = datetime(2024, 3, 1)  # Use March 1, 2024 as end date
        start_date = datetime(2022, 3, 1)  # Use March 1, 2022 as start date (2 years)
        
        logger.info(f"📅 Running backtest from {start_date.date()} to {end_date.date()}")
        
        # List of symbols to test
        symbols = ['PLTR', 'TSLA', 'SPY']
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Run comparison for each symbol
        for symbol in symbols:
            results_dict = run_strategy_comparison(api, symbol, start_date, end_date)
            
            # Plot comparison
            plot_comparison(symbol, results_dict)
            
            logger.info(f"\n✅ Completed comparison for {symbol}")
            
    except Exception as e:
        logger.error(f"❌ Error in strategy comparison: {str(e)}")

if __name__ == "__main__":
    test_strategies() 