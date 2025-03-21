from connection import AlpacaConnection
from strategies.value_averaging import ValueAveragingStrategy
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
            logger.info(f"✅ Market is OPEN | Closes at: {next_close}")
        else:
            logger.info(f"❌ Market is CLOSED | Next open: {next_open}")
        
        return clock.is_open
    except Exception as e:
        logger.error(f"❌ Error getting market status: {str(e)}")
        return False

def wait_for_market_open(api, check_interval=60):
    """Wait for market to open"""
    while True:
        if get_market_status(api):
            logger.info("🚀 Market is now open!")
            return True
        logger.info(f"⏳ Waiting for market to open... Checking again in {check_interval} seconds")
        time.sleep(check_interval)

def test_strategy():
    """Test the Value Averaging strategy for PLTR"""
    # Initialize connection
    alpaca = AlpacaConnection()
    
    try:
        # Connect to Alpaca
        logger.info("🔄 Testing Alpaca connection...")
        if not alpaca.connect():
            logger.error("❌ Failed to connect to Alpaca")
            return False
        
        # Check market status and wait if closed
        is_market_open = get_market_status(alpaca.api)
        if not is_market_open:
            logger.warning("🚦 Market closed. Waiting for the next trading session...")
            wait_for_market_open(alpaca.api)
        
        # Initialize strategy
        logger.info("📈 Initializing Value Averaging strategy for PLTR...")
        strategy = ValueAveragingStrategy(
            api=alpaca.api,
            symbol='PLTR',
            weekly_target=100,  # Target $100 investment per week
            stop_loss_pct=0.1,  # 10% stop loss
            take_profit_pct=0.2  # 20% take profit
        )
        
        # Run strategy for 5 minutes
        logger.info("🚀 Running strategy for 5 minutes...")
        start_time = time.time()
        while time.time() - start_time < 300:  # Run for 5 minutes
            try:
                # Check market status
                if not get_market_status(alpaca.api):
                    logger.warning("🚦 Market closed. Waiting for next session...")
                    wait_for_market_open(alpaca.api)
                
                # Get current price
                current_price = strategy.get_current_price()
                if current_price is None:
                    logger.warning("⚠️ Could not fetch stock price. Retrying...")
                    time.sleep(30)
                    continue
                
                logger.info(f"💲 Current PLTR Price: ${current_price:.2f}")
                
                # Calculate investment needed
                investment_needed = strategy.calculate_investment_needed()
                logger.info(f"🔢 Investment Needed: ${investment_needed:.2f}")
                
                # Execute trade if needed
                if investment_needed > 0:
                    strategy.execute_trade(investment_needed)
                    logger.info(f"✅ Executed trade for ${investment_needed:.2f}")
                
                # Wait for 1 minute before next iteration
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"⚠️ Error in strategy loop: {str(e)}")
                time.sleep(60)
        
        logger.info("🏁 Strategy test completed!")
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 Strategy test interrupted by user")
        return False
    except Exception as e:
        logger.error(f"⚠️ Unexpected error: {str(e)}")
        return False
    finally:
        # Clean up
        if alpaca.is_connected():
            alpaca.disconnect()

if __name__ == "__main__":
    test_strategy() 