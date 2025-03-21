import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from config import MAX_POSITION_SIZE, STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE

logger = logging.getLogger(__name__)

class AlphaMLStrategy:
    """
    Alpha generation strategy using machine learning to predict price direction
    based on technical indicators and market data.
    """
    def __init__(self, api, symbol, lookback_period=30, prediction_horizon=5, model_path=None):
        self.api = api
        self.symbol = symbol
        self.lookback_period = lookback_period  # Days of historical data to use
        self.prediction_horizon = prediction_horizon  # Days to predict forward
        self.position = 0
        self.model_path = model_path
        self.model = self.load_model() if model_path and os.path.exists(model_path) else None
        self.scaler = None
        
    def load_model(self):
        """Load a pre-trained model if it exists"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                model = joblib.load(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
                
                # Also load the scaler if it exists
                scaler_path = self.model_path.replace('.joblib', '_scaler.joblib')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    logger.info(f"Loaded scaler from {scaler_path}")
                
                return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
        return None
    
    def is_market_open(self):
        """Check if the market is currently open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False
            
    def get_historical_data(self):
        """Get historical data for feature engineering"""
        try:
            # Get daily bars for the lookback period plus prediction horizon
            bars = self.api.get_bars(
                self.symbol,
                tradeapi.TimeFrame.Day,
                limit=self.lookback_period + self.prediction_horizon
            ).df
            
            if bars.empty:
                logger.error(f"No historical data received for {self.symbol}")
                return None

            return bars
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
            
    def engineer_features(self, df):
        """Create technical features from price data"""
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
        data['price_to_sma50'] = data['close'] / data['sma50'] if 'sma50' in data else np.nan
        
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
        
        # Scale features if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
        
    def create_target(self, df):
        """Create target variable for training (future returns)"""
        # Future returns over prediction horizon
        df['future_return'] = df['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # Binary classification target (1 for positive return, 0 for negative)
        df['target'] = (df['future_return'] > 0).astype(int)
        
        return df
        
    def train_model(self, save_path=None):
        """Train ML model on historical data"""
        try:
            # Get extended historical data for training
            extended_lookback = 252  # Use 1 year of data for training
            
            bars = self.api.get_bars(
                self.symbol,
                tradeapi.TimeFrame.Day,
                limit=extended_lookback + self.prediction_horizon
            ).df
            
            if bars.empty or len(bars) < 60:  # Need at least 60 days of data
                logger.error(f"Insufficient historical data for training")
                return False
                
            # Engineer features
            data = self.engineer_features(bars)
            if data is None:
                return False
                
            # Create target
            data = self.create_target(data)
            
            # Remove rows with NaN in target
            data = data.dropna(subset=['target'])
            
            if len(data) < 30:  # Need at least 30 data points for training
                logger.error(f"Insufficient data points after preprocessing")
                return False
                
            # Prepare features and target
            feature_cols = [col for col in data.columns if col not in ['target', 'future_return']]
            X = data[feature_cols]
            y = data['target']
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=5,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
            # Save model if path provided
            if save_path:
                joblib.dump(self.model, save_path)
                logger.info(f"Model saved to {save_path}")
                
                # Also save the scaler
                scaler_path = save_path.replace('.joblib', '_scaler.joblib')
                joblib.dump(self.scaler, scaler_path)
                logger.info(f"Scaler saved to {scaler_path}")
                
                self.model_path = save_path
                
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
            
    def predict(self, df):
        """Make prediction using trained model"""
        if self.model is None:
            logger.error("No trained model available")
            return None
            
        # Prepare features
        X = self.prepare_features(df)
        if X is None or len(X) == 0:
            return None
            
        # Use the latest data point for prediction
        if isinstance(X, np.ndarray):
            X_latest = X[-1].reshape(1, -1)
        else:
            X_latest = X.iloc[-1].values.reshape(1, -1)
            
        # Make prediction
        prediction = self.model.predict(X_latest)[0]
        
        # Get probability
        proba = self.model.predict_proba(X_latest)[0]
        confidence = proba[1] if prediction == 1 else proba[0]
        
        return {
            'prediction': prediction,  # 1 for buy, 0 for sell
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
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
            
    def calculate_position_size(self, current_price, confidence):
        """Calculate position size based on account value and prediction confidence"""
        try:
            account = self.api.get_account()
            account_value = float(account.portfolio_value)
            
            # Scale position size by confidence level
            position_size = int(account_value * 0.02 * confidence / current_price)
            
            # Limit to maximum position size
            return min(MAX_POSITION_SIZE, position_size)
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
            
    def set_stop_loss_take_profit(self, entry_price):
        """Set stop loss and take profit orders"""
        try:
            if self.position <= 0:
                return
                
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
            
    def execute_trade(self, prediction, confidence, current_price):
        """Execute trade based on prediction and confidence"""
        try:
            # Check if market is open
            if not self.is_market_open():
                logger.info("Market is closed. Skipping trade execution.")
                return
                
            # Check confidence threshold
            if confidence < 0.6:  # Minimum 60% confidence to trade
                logger.info(f"Confidence too low ({confidence:.2f}) to trade")
                return
                
            if prediction == 1 and self.position <= 0:  # Buy signal
                # Calculate position size based on confidence
                quantity = self.calculate_position_size(current_price, confidence)
                
                if quantity <= 0:
                    logger.info("Calculated position size too small, skipping trade")
                    return
                    
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

            elif prediction == 0 and self.position > 0:  # Sell signal
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
                
                logger.info(f"Sell order filled for {self.position} shares of {self.symbol} at ${filled_price:.2f}")
                self.position = 0

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            
    def run(self, train_first=False):
        """Main strategy loop"""
        # Train model first if required
        if train_first or self.model is None:
            logger.info("Training model...")
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{self.symbol}_alpha_ml_model.joblib")
            success = self.train_model(save_path=model_path)
            if not success:
                logger.error("Failed to train model. Exiting strategy.")
                return
                
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
                    time.sleep(60)
                    continue

                # Engineer features
                df = self.engineer_features(df)
                if df is None:
                    time.sleep(60)
                    continue
                
                # Get current price
                current_price = df['close'].iloc[-1]
                
                # Make prediction
                prediction_result = self.predict(df)
                if prediction_result is None:
                    time.sleep(60)
                    continue
                    
                # Log prediction
                logger.info(f"Prediction: {'BUY' if prediction_result['prediction'] == 1 else 'SELL'} "
                           f"with {prediction_result['confidence']:.2f} confidence")
                
                # Execute trade based on prediction
                self.execute_trade(
                    prediction_result['prediction'],
                    prediction_result['confidence'],
                    current_price
                )

                # Wait for next iteration - check every hour for this strategy
                time.sleep(3600)  # Check every hour

            except Exception as e:
                logger.error(f"Error in strategy loop: {str(e)}")
                time.sleep(60)  # Wait before retrying 