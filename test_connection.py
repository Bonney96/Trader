from connection import AlpacaConnection
import logging
from config import LOG_LEVEL, LOG_FILE

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

def test_alpaca_connection():
    """Test Alpaca connection and basic API functionality"""
    alpaca = AlpacaConnection()
    
    # Test connection
    logger.info("Testing Alpaca connection...")
    if not alpaca.connect():
        logger.error("Failed to connect to Alpaca")
        return False
    
    # Test account info
    logger.info("Testing account information retrieval...")
    account_info = alpaca.get_account_info()
    if account_info:
        logger.info("Account Information:")
        for key, value in account_info.items():
            logger.info(f"{key}: {value}")
    else:
        logger.error("Failed to get account information")
        return False
    
    # Test market status
    logger.info("Testing market status retrieval...")
    market_status = alpaca.get_market_status()
    if market_status:
        logger.info("Market Status:")
        for key, value in market_status.items():
            logger.info(f"{key}: {value}")
    else:
        logger.error("Failed to get market status")
        return False
    
    # Test positions
    logger.info("Testing positions retrieval...")
    positions = alpaca.get_positions()
    if positions is not None:
        logger.info(f"Current positions: {positions}")
    else:
        logger.error("Failed to get positions")
        return False
    
    logger.info("All tests completed successfully!")
    return True

if __name__ == "__main__":
    test_alpaca_connection() 