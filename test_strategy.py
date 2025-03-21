from connection import AlpacaConnection
from strategies.moving_average_crossover import MovingAverageCrossoverStrategy
import logging
from config import LOG_LEVEL, LOG_FILE
import time
from datetime import datetime

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

def get_market_status(api):
    """Get and display market status"""
    try:
        clock = api.get_clock()
        next_open = clock.next_open
        next_close = clock.next_close
        
        if clock.is_open:
            logger.info("Market is currently OPEN")
            logger.info(f"Market closes at: {next_close}")
        else:
            logger.info("Market is currently CLOSED")
            logger.info(f"Next market open: {next_open}")
            logger.info(f"Next market close: {next_close}")
        
        return clock.is_open
    except Exception as e:
        logger.error(f"Error getting market status: {str(e)}")
        return False

def test_strategy():
    """Test the moving average crossover strategy"""
    # Initialize connection
    alpaca = AlpacaConnection()
    
    try:
        # Connect to Alpaca
        logger.info("Testing Alpaca connection...")
        if not alpaca.connect():
            logger.error("Failed to connect to Alpaca")
            return False
        
        # Check market status
        is_market_open = get_market_status(alpaca.api)
        if not is_market_open:
            logger.info("Market is closed. Strategy will wait for market open.")
        
        # Initialize strategy
        logger.info("Initializing strategy...")
        strategy = MovingAverageCrossoverStrategy(
            api=alpaca.api,
            symbol='AAPL',  # Trading Apple stock as an example
            short_window=20,
            long_window=50
        )
        
        # Run strategy for 5 minutes
        logger.info("Running strategy for 5 minutes...")
        start_time = time.time()
        while time.time() - start_time < 300:  # Run for 5 minutes
            try:
                # Get historical data
                df = strategy.get_historical_data()
                if df is not None:
                    logger.info(f"Latest price for {strategy.symbol}: ${df['close'].iloc[-1]:.2f}")
                    
                    # Calculate signals
                    df = strategy.calculate_signals(df)
                    if df is not None:
                        current_signal = df['signal'].iloc[-1]
                        logger.info(f"Current signal: {current_signal}")
                        
                        # Execute trade based on signal
                        strategy.execute_trade(current_signal, df['close'].iloc[-1])
                
                # Wait for 1 minute before next iteration
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in strategy loop: {str(e)}")
                time.sleep(60)
        
        logger.info("Strategy test completed!")
        return True
        
    except KeyboardInterrupt:
        logger.info("Strategy test interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False
    finally:
        # Clean up
        if alpaca.is_connected():
            alpaca.disconnect()

if __name__ == "__main__":
    test_strategy() 