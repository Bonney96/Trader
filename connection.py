import alpaca_trade_api as tradeapi
import logging
from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_PAPER,
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

class AlpacaConnection:
    def __init__(self):
        self.api = None
        self.connected = False

    def connect(self):
        """Establish connection to Alpaca API"""
        try:
            if not self.connected:
                if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                    raise ValueError("Alpaca API keys not found in environment variables")

                # Set the base URL based on paper trading setting
                base_url = 'https://paper-api.alpaca.markets' if ALPACA_PAPER else 'https://api.alpaca.markets'

                # Initialize the API
                self.api = tradeapi.REST(
                    ALPACA_API_KEY,
                    ALPACA_SECRET_KEY,
                    base_url=base_url
                )
                
                # Test the connection
                self.api.get_account()
                self.connected = True
                logger.info(f"Successfully connected to Alpaca {'Paper' if ALPACA_PAPER else 'Live'} Trading")
                return True

        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {str(e)}")
            return False

    def disconnect(self):
        """Disconnect from Alpaca API"""
        try:
            if self.connected:
                self.api = None
                self.connected = False
                logger.info("Disconnected from Alpaca")
        except Exception as e:
            logger.error(f"Error disconnecting from Alpaca: {str(e)}")

    def get_account_info(self):
        """Get account information"""
        try:
            if not self.is_connected():
                logger.error("Not connected to Alpaca")
                return None
            
            account = self.api.get_account()
            return {
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'last_equity': float(account.last_equity),
                'status': account.status
            }
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None

    def get_positions(self):
        """Get current positions"""
        try:
            if not self.is_connected():
                logger.error("Not connected to Alpaca")
                return None
            
            positions = self.api.list_positions()
            return [{
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc)
            } for pos in positions]
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return None

    def is_connected(self):
        """Check if connected to Alpaca"""
        return self.connected and self.api is not None

    def is_market_open(self):
        """Check if the market is currently open"""
        try:
            if not self.is_connected():
                return False
            
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False

    def get_market_status(self):
        """Get detailed market status"""
        try:
            if not self.is_connected():
                return None
            
            clock = self.api.get_clock()
            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close
            }
        except Exception as e:
            logger.error(f"Error getting market status: {str(e)}")
            return None 