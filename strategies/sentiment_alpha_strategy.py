import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import requests
import json
import os
from config import MAX_POSITION_SIZE, STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE

# Add environment variables for news/sentiment API keys
# These should be added to .env file
# NEWS_API_KEY=your_key_here
# FINNHUB_API_KEY=your_key_here

logger = logging.getLogger(__name__)

class SentimentAlphaStrategy:
    """
    Alpha generation strategy using sentiment analysis of news and social media
    to predict price movements.
    """
    def __init__(self, api, symbol, sentiment_lookback=3):
        self.api = api
        self.symbol = symbol
        self.position = 0
        self.sentiment_lookback = sentiment_lookback  # Days of sentiment to analyze
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        
    def is_market_open(self):
        """Check if the market is currently open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False
            
    def get_current_price(self):
        """Get the current price of the symbol"""
        try:
            bars = self.api.get_bars(
                self.symbol,
                tradeapi.TimeFrame.Minute,
                limit=1
            ).df
            
            if bars.empty:
                logger.error(f"No price data received for {self.symbol}")
                return None
                
            return bars['close'].iloc[-1]
        except Exception as e:
            logger.error(f"Error getting current price: {str(e)}")
            return None
            
    def get_news_sentiment(self):
        """
        Get news sentiment for the symbol using News API
        Returns a sentiment score between -1 (very negative) and 1 (very positive)
        """
        if not self.news_api_key:
            logger.error("NEWS_API_KEY not found in environment variables")
            return 0
            
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.sentiment_lookback)
            
            # Format dates for API
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # Build query - use symbol and company name if available
            query = f"{self.symbol} stock"
            
            # Make request to News API
            url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&language=en&sortBy=relevancy&apiKey={self.news_api_key}"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"News API error: {response.status_code} - {response.text}")
                return 0
                
            data = response.json()
            articles = data.get('articles', [])
            
            if not articles:
                logger.info(f"No news articles found for {self.symbol}")
                return 0
                
            # Limit to top 10 most relevant articles
            articles = articles[:10]
            
            # Simple sentiment analysis based on keywords
            # In a production system, you would use a proper NLP sentiment analyzer
            positive_keywords = ['up', 'rise', 'gain', 'positive', 'growth', 'profit', 'surge', 'rally', 'bullish']
            negative_keywords = ['down', 'fall', 'drop', 'negative', 'loss', 'decline', 'bearish', 'crash', 'plunge']
            
            sentiment_scores = []
            
            for article in articles:
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                content = article.get('content', '').lower()
                
                # Combine all text for analysis
                text = f"{title} {description} {content}"
                
                # Count positive and negative keywords
                positive_count = sum(1 for word in positive_keywords if word in text)
                negative_count = sum(1 for word in negative_keywords if word in text)
                
                # Calculate article sentiment score
                if positive_count + negative_count > 0:
                    article_score = (positive_count - negative_count) / (positive_count + negative_count)
                else:
                    article_score = 0
                    
                sentiment_scores.append(article_score)
                
            # Calculate overall sentiment score (-1 to 1)
            if sentiment_scores:
                overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                logger.info(f"News sentiment for {self.symbol}: {overall_sentiment:.2f}")
                return overall_sentiment
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error getting news sentiment: {str(e)}")
            return 0
            
    def get_finnhub_sentiment(self):
        """
        Get social sentiment data from Finnhub API
        Returns a sentiment score between -1 (very negative) and 1 (very positive)
        """
        if not self.finnhub_api_key:
            logger.error("FINNHUB_API_KEY not found in environment variables")
            return 0
            
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.sentiment_lookback)
            
            # Format dates for API
            from_timestamp = int(start_date.timestamp())
            to_timestamp = int(end_date.timestamp())
            
            # Make request to Finnhub API
            url = f"https://finnhub.io/api/v1/stock/social-sentiment?symbol={self.symbol}&from={from_timestamp}&to={to_timestamp}&token={self.finnhub_api_key}"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"Finnhub API error: {response.status_code} - {response.text}")
                return 0
                
            data = response.json()
            reddit = data.get('reddit', [])
            twitter = data.get('twitter', [])
            
            # Combine Reddit and Twitter sentiment
            all_sentiment = reddit + twitter
            
            if not all_sentiment:
                logger.info(f"No social sentiment data found for {self.symbol}")
                return 0
                
            # Calculate weighted sentiment score
            total_mentions = sum(item.get('mention', 0) for item in all_sentiment)
            
            if total_mentions == 0:
                return 0
                
            weighted_sentiment = sum(
                item.get('score', 0) * item.get('mention', 0) 
                for item in all_sentiment
            ) / total_mentions
            
            # Normalize to -1 to 1 scale (Finnhub scores are typically 0 to 1)
            normalized_sentiment = (weighted_sentiment * 2) - 1
            
            logger.info(f"Social sentiment for {self.symbol}: {normalized_sentiment:.2f}")
            return normalized_sentiment
            
        except Exception as e:
            logger.error(f"Error getting Finnhub sentiment: {str(e)}")
            return 0
            
    def get_combined_sentiment(self):
        """Combine different sentiment sources into a single score"""
        # Get sentiment from different sources
        news_sentiment = self.get_news_sentiment()
        social_sentiment = self.get_finnhub_sentiment()
        
        # Calculate combined sentiment (give more weight to news)
        combined_sentiment = (news_sentiment * 0.7) + (social_sentiment * 0.3)
        
        # Log the sentiment values
        logger.info(f"Combined sentiment for {self.symbol}: {combined_sentiment:.2f}")
        
        return combined_sentiment
        
    def generate_signal(self, sentiment_score):
        """
        Generate trading signal based on sentiment score
        Returns: 1 for buy, -1 for sell, 0 for hold
        """
        # Strong positive sentiment -> Buy
        if sentiment_score > 0.3:
            return 1
        # Strong negative sentiment -> Sell
        elif sentiment_score < -0.3:
            return -1
        # Neutral sentiment -> Hold
        else:
            return 0
            
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
            
    def calculate_position_size(self, current_price, sentiment_strength):
        """Calculate position size based on account value and sentiment strength"""
        try:
            account = self.api.get_account()
            account_value = float(account.portfolio_value)
            
            # Scale position size by absolute sentiment value (0.3 to 1.0)
            sentiment_strength = max(0.3, min(1.0, abs(sentiment_strength)))
            position_size = int(account_value * 0.02 * sentiment_strength / current_price)
            
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
            
    def execute_trade(self, signal, sentiment_score, current_price):
        """Execute trade based on signal and sentiment strength"""
        try:
            # Check if market is open
            if not self.is_market_open():
                logger.info("Market is closed. Skipping trade execution.")
                return
                
            if signal == 1 and self.position <= 0:  # Buy signal
                # Calculate position size based on sentiment strength
                quantity = self.calculate_position_size(current_price, sentiment_score)
                
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

            elif signal == -1 and self.position > 0:  # Sell signal
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
            
    def run(self):
        """Main strategy loop"""
        while True:
            try:
                # Check if market is open
                if not self.is_market_open():
                    logger.info("Market is closed. Waiting for market open...")
                    time.sleep(60)  # Check every minute
                    continue

                # Get current price
                current_price = self.get_current_price()
                if current_price is None:
                    time.sleep(60)
                    continue
                    
                # Get sentiment analysis
                sentiment_score = self.get_combined_sentiment()
                
                # Generate trading signal
                signal = self.generate_signal(sentiment_score)
                
                # Log signal
                signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
                logger.info(f"Generated {signal_text} signal with sentiment score: {sentiment_score:.2f}")
                
                # Execute trade based on signal
                self.execute_trade(signal, sentiment_score, current_price)

                # Wait for next check - run sentiment analysis every 4 hours
                time.sleep(14400)  # 4 hours in seconds

            except Exception as e:
                logger.error(f"Error in strategy loop: {str(e)}")
                time.sleep(60)  # Wait before retrying 