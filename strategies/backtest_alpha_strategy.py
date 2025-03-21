import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import logging
import alpaca_trade_api as tradeapi
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
import json
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY, API_SECRET, BASE_URL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AlphaStrategyBacktester:
    """
    Backtester for alpha-generating strategies.
    """
    def __init__(self, symbol, start_date, end_date, initial_capital=100000.0):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.api = self._connect_to_api()
        
    def _connect_to_api(self):
        """Connect to Alpaca API"""
        try:
            api = tradeapi.REST(
                API_KEY,
                API_SECRET,
                base_url=BASE_URL
            )
            return api
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {str(e)}")
            raise
            
    def get_historical_data(self):
        """
        Get historical price data for the specified period
        """
        try:
            # Convert dates to string format expected by Alpaca
            start_date_str = self.start_date.strftime('%Y-%m-%d')
            end_date_str = self.end_date.strftime('%Y-%m-%d')
            
            # Get daily data
            bars = self.api.get_bars(
                self.symbol,
                tradeapi.TimeFrame.Day,
                start=start_date_str,
                end=end_date_str,
                adjustment='raw'
            ).df
            
            if bars.empty:
                logger.error(f"No historical data received for {self.symbol}")
                return None
                
            # Reset index to make date a column
            bars.reset_index(inplace=True)
            
            return bars
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
            
    def engineer_features(self, df):
        """Generate features for machine learning strategy"""
        if df is None or len(df) < 20:  # Need at least 20 days of data for indicators
            return None
            
        # Make a copy to avoid warnings
        data = df.copy()
        
        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Volume features
        data['volume_ma5'] = data['volume'].rolling(window=5).mean()
        data['volume_ma10'] = data['volume'].rolling(window=10).mean()
        data['relative_volume'] = data['volume'] / data['volume_ma5']
        
        # Technical indicators
        # RSI
        data['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(data['close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['close'])
        data['bb_high'] = bollinger.bollinger_hband()
        data['bb_low'] = bollinger.bollinger_lband()
        data['bb_mid'] = bollinger.bollinger_mavg()
        data['bb_pct'] = (data['close'] - data['bb_low']) / (data['bb_high'] - data['bb_low'])
        
        # ADX - Trend strength
        adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'])
        data['adx'] = adx.adx()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
        
        # Moving Averages
        data['sma5'] = ta.trend.SMAIndicator(data['close'], window=5).sma_indicator()
        data['sma10'] = ta.trend.SMAIndicator(data['close'], window=10).sma_indicator()
        data['sma20'] = ta.trend.SMAIndicator(data['close'], window=20).sma_indicator()
        data['ema5'] = ta.trend.EMAIndicator(data['close'], window=5).ema_indicator()
        data['ema10'] = ta.trend.EMAIndicator(data['close'], window=10).ema_indicator()
        data['ema20'] = ta.trend.EMAIndicator(data['close'], window=20).ema_indicator()
        
        # MA Crossovers
        data['ma_cross_5_10'] = (data['sma5'] > data['sma10']).astype(int)
        data['ma_cross_10_20'] = (data['sma10'] > data['sma20']).astype(int)
        
        # Price relative to MAs
        data['price_to_sma20'] = data['close'] / data['sma20']
        
        # Create future returns (targets for ML)
        for horizon in [1, 3, 5, 10]:
            data[f'future_return_{horizon}d'] = data['close'].pct_change(horizon).shift(-horizon)
            # Create binary target (1 for positive return, 0 for negative)
            data[f'target_{horizon}d'] = (data[f'future_return_{horizon}d'] > 0).astype(int)
        
        # Clean up missing values
        data = data.dropna()
        
        return data
        
    def prepare_features(self, df):
        """Prepare features for ML model input"""
        if df is None:
            return None
            
        # Get features excluding target and irrelevant columns
        feature_cols = [
            'returns', 'log_returns', 'volume_ma5', 'volume_ma10', 'relative_volume',
            'rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_pct', 'adx', 
            'stoch_k', 'stoch_d', 'price_to_sma20', 'ma_cross_5_10', 'ma_cross_10_20'
        ]
        
        # Make sure all feature columns exist in the dataframe
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) < 5:  # Need at least 5 features
            logger.error("Not enough features available for prediction")
            return None
            
        # Extract features
        X = df[available_features].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        return X
        
    def train_ml_model(self, train_data, prediction_horizon=5):
        """Train ML model on training data"""
        try:
            # Prepare features using the same method as prediction
            X = self.prepare_features(train_data)
            if X is None or len(X) == 0:
                logger.error("Failed to prepare features for training")
                return None, None
                
            # Set up target column
            target_col = f'target_{prediction_horizon}d'
            
            if target_col not in train_data.columns:
                logger.error(f"Target column {target_col} not found in data")
                return None, None
                
            # Prepare target
            y = train_data[target_col]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
            model.fit(X_scaled, y)
            
            return model, scaler
            
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}")
            return None, None
            
    def backtest_ml_strategy(self, prediction_horizon=5, train_ratio=0.6):
        """Backtest machine learning alpha strategy"""
        try:
            # Get historical data
            data = self.get_historical_data()
            if data is None:
                return None
                
            # Engineer features
            data = self.engineer_features(data)
            if data is None:
                return None
                
            # Split data into training and testing
            split_idx = int(len(data) * train_ratio)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
            logger.info(f"Training data size: {len(train_data)}, Test data size: {len(test_data)}")
            
            # Train model
            model, scaler = self.train_ml_model(train_data, prediction_horizon)
            if model is None or scaler is None:
                return None
                
            # Generate predictions on test data
            X_test = self.prepare_features(test_data)
            if X_test is None:
                logger.error("Failed to prepare features for testing")
                return None
                
            X_test_scaled = scaler.transform(X_test)
            
            # Get probabilities
            test_data['prediction_proba'] = model.predict_proba(X_test_scaled)[:, 1]
            
            # Generate signals (1 for buy, -1 for sell, 0 for hold)
            test_data['signal'] = 0
            test_data.loc[test_data['prediction_proba'] > 0.65, 'signal'] = 1  # Strong buy signal
            test_data.loc[test_data['prediction_proba'] < 0.35, 'signal'] = -1  # Strong sell signal
            
            # Backtest trading strategy
            test_data['position'] = test_data['signal'].shift(1).fillna(0)
            test_data['returns'] = test_data['close'].pct_change()
            test_data['strategy_returns'] = test_data['position'] * test_data['returns']
            
            # Calculate cumulative returns
            test_data['cum_returns'] = (1 + test_data['returns']).cumprod() - 1
            test_data['cum_strategy_returns'] = (1 + test_data['strategy_returns']).cumprod() - 1
            
            # Calculate equity curve
            test_data['equity'] = self.initial_capital * (1 + test_data['cum_strategy_returns'])
            
            # Calculate performance metrics
            total_return = test_data['cum_strategy_returns'].iloc[-1]
            annual_return = (1 + total_return) ** (252 / len(test_data)) - 1
            
            # Calculate Sharpe ratio
            sharpe_ratio = (
                test_data['strategy_returns'].mean() / test_data['strategy_returns'].std()
                * np.sqrt(252)  # Annualize
            )
            
            # Max drawdown
            drawdown = 1 - test_data['equity'] / test_data['equity'].cummax()
            max_drawdown = drawdown.max()
            
            # Calculate win rate
            winning_days = test_data[test_data['strategy_returns'] > 0]
            win_rate = len(winning_days) / len(test_data[test_data['strategy_returns'] != 0])
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': list(X_test.columns),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results = {
                'test_data': test_data,
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'feature_importance': feature_importance
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ML strategy backtest: {str(e)}")
            return None
            
    def backtest_ma_crossover(self, short_window=20, long_window=50):
        """Backtest moving average crossover strategy"""
        try:
            # Get historical data
            data = self.get_historical_data()
            if data is None:
                return None
                
            # Calculate moving averages
            data['short_ma'] = data['close'].rolling(window=short_window).mean()
            data['long_ma'] = data['close'].rolling(window=long_window).mean()
            
            # Generate signals
            data['signal'] = 0
            data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
            data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
            
            # Drop rows with NaN values (due to rolling windows)
            data = data.dropna()
            
            # Backtest trading strategy
            data['position'] = data['signal'].shift(1).fillna(0)
            data['returns'] = data['close'].pct_change()
            data['strategy_returns'] = data['position'] * data['returns']
            
            # Calculate cumulative returns
            data['cum_returns'] = (1 + data['returns']).cumprod() - 1
            data['cum_strategy_returns'] = (1 + data['strategy_returns']).cumprod() - 1
            
            # Calculate equity curve
            data['equity'] = self.initial_capital * (1 + data['cum_strategy_returns'])
            
            # Calculate performance metrics
            total_return = data['cum_strategy_returns'].iloc[-1]
            annual_return = (1 + total_return) ** (252 / len(data)) - 1
            
            # Calculate Sharpe ratio
            sharpe_ratio = (
                data['strategy_returns'].mean() / data['strategy_returns'].std()
                * np.sqrt(252)  # Annualize
            )
            
            # Max drawdown
            drawdown = 1 - data['equity'] / data['equity'].cummax()
            max_drawdown = drawdown.max()
            
            # Calculate win rate
            winning_days = data[data['strategy_returns'] > 0]
            win_rate = len(winning_days) / len(data[data['strategy_returns'] != 0])
            
            results = {
                'test_data': data,
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in MA Crossover backtest: {str(e)}")
            return None
            
    def backtest_rsi_strategy(self, period=14, oversold=30, overbought=70):
        """Backtest RSI mean reversion strategy"""
        try:
            # Get historical data
            data = self.get_historical_data()
            if data is None:
                return None
                
            # Calculate RSI
            data['rsi'] = ta.momentum.RSIIndicator(data['close'], window=period).rsi()
            
            # Generate signals
            data['signal'] = 0
            # Buy when RSI is below oversold threshold
            data.loc[data['rsi'] < oversold, 'signal'] = 1
            # Sell when RSI is above overbought threshold
            data.loc[data['rsi'] > overbought, 'signal'] = -1
            
            # Drop rows with NaN values (due to RSI calculation)
            data = data.dropna()
            
            # Backtest trading strategy
            data['position'] = data['signal'].shift(1).fillna(0)
            data['returns'] = data['close'].pct_change()
            data['strategy_returns'] = data['position'] * data['returns']
            
            # Calculate cumulative returns
            data['cum_returns'] = (1 + data['returns']).cumprod() - 1
            data['cum_strategy_returns'] = (1 + data['strategy_returns']).cumprod() - 1
            
            # Calculate equity curve
            data['equity'] = self.initial_capital * (1 + data['cum_strategy_returns'])
            
            # Calculate performance metrics
            total_return = data['cum_strategy_returns'].iloc[-1]
            annual_return = (1 + total_return) ** (252 / len(data)) - 1
            
            # Calculate Sharpe ratio
            sharpe_ratio = (
                data['strategy_returns'].mean() / data['strategy_returns'].std()
                * np.sqrt(252)  # Annualize
            )
            
            # Max drawdown
            drawdown = 1 - data['equity'] / data['equity'].cummax()
            max_drawdown = drawdown.max()
            
            # Calculate win rate
            winning_days = data[data['strategy_returns'] > 0]
            win_rate = len(winning_days) / len(data[data['strategy_returns'] != 0])
            
            results = {
                'test_data': data,
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in RSI strategy backtest: {str(e)}")
            return None
            
    def visualize_results(self, results, strategy_name):
        """Visualize backtest results"""
        if results is None:
            logger.error("No results to visualize")
            return
            
        test_data = results['test_data']
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Plot equity curve
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(test_data['timestamp'], test_data['equity'], label='Strategy Equity')
        plt.plot(test_data['timestamp'], self.initial_capital * (1 + test_data['cum_returns']), 
                 label='Buy & Hold', alpha=0.7)
        plt.title(f'{strategy_name} - Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        drawdown = 1 - test_data['equity'] / test_data['equity'].cummax()
        plt.fill_between(test_data['timestamp'], drawdown, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'results/{self.symbol}_{strategy_name.replace(" ", "_").lower()}_results.png')
        
        # Display performance metrics
        metrics = pd.DataFrame({
            'Metric': ['Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
            'Value': [
                f"{results['total_return']:.2%}",
                f"{results['annual_return']:.2%}",
                f"{results['sharpe_ratio']:.2f}",
                f"{results['max_drawdown']:.2%}",
                f"{results['win_rate']:.2%}"
            ]
        })
        
        # Save metrics to CSV
        metrics.to_csv(f'results/{self.symbol}_{strategy_name.replace(" ", "_").lower()}_metrics.csv', index=False)
        
        # Print metrics
        logger.info(f"\n{strategy_name} Performance Metrics for {self.symbol}:")
        logger.info(f"\n{metrics.to_string(index=False)}")
        
        # Plot feature importance if available
        if 'feature_importance' in results and results['feature_importance'] is not None:
            plt.figure(figsize=(10, 6))
            feature_imp = results['feature_importance'].head(10)  # Top 10 features
            sns.barplot(x='importance', y='feature', data=feature_imp)
            plt.title(f'Top Features for {strategy_name}')
            plt.tight_layout()
            plt.savefig(f'results/{self.symbol}_{strategy_name.replace(" ", "_").lower()}_feature_importance.png')
            
    def save_results(self, results, strategy_name):
        """Save backtest results to JSON"""
        if results is None:
            logger.error("No results to save")
            return
            
        # Create a simplified version of results (without DataFrames)
        results_to_save = {
            'symbol': self.symbol,
            'strategy': strategy_name,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'initial_capital': self.initial_capital,
            'total_return': float(results['total_return']),
            'annual_return': float(results['annual_return']),
            'sharpe_ratio': float(results['sharpe_ratio']),
            'max_drawdown': float(results['max_drawdown']),
            'win_rate': float(results['win_rate'])
        }
        
        # Save to JSON
        os.makedirs('results', exist_ok=True)
        with open(f'results/{self.symbol}_{strategy_name.replace(" ", "_").lower()}_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=4)
            
        logger.info(f"Results saved to results/{self.symbol}_{strategy_name.replace(" ", "_").lower()}_results.json")
        
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Backtest Alpha Strategies')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to backtest')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--strategy', type=str, default='all', 
                        choices=['ml', 'ma_crossover', 'rsi', 'all'],
                        help='Strategy to backtest')
    
    return parser.parse_args()
    
def main():
    """Main function"""
    args = parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Create backtester
    backtester = AlphaStrategyBacktester(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital
    )
    
    if args.strategy == 'ml' or args.strategy == 'all':
        # Backtest ML strategy
        ml_results = backtester.backtest_ml_strategy()
        if ml_results:
            backtester.visualize_results(ml_results, 'ML Alpha Strategy')
            backtester.save_results(ml_results, 'ML Alpha Strategy')
    
    if args.strategy == 'ma_crossover' or args.strategy == 'all':
        # Backtest MA Crossover strategy
        ma_results = backtester.backtest_ma_crossover()
        if ma_results:
            backtester.visualize_results(ma_results, 'MA Crossover Strategy')
            backtester.save_results(ma_results, 'MA Crossover Strategy')
    
    if args.strategy == 'rsi' or args.strategy == 'all':
        # Backtest RSI strategy
        rsi_results = backtester.backtest_rsi_strategy()
        if rsi_results:
            backtester.visualize_results(rsi_results, 'RSI Strategy')
            backtester.save_results(rsi_results, 'RSI Strategy')
            
if __name__ == "__main__":
    main() 