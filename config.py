import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Alpaca API Configuration
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Trading Parameters
MAX_POSITION_SIZE = 100  # Maximum number of shares per position
STOP_LOSS_PERCENTAGE = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENTAGE = 0.04  # 4% take profit

# Trading Schedule
TRADING_HOURS = {
    'start': '09:30',  # Market open (EST)
    'end': '16:00'     # Market close (EST)
}

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'trading.log')

# Risk Management
MAX_DAILY_LOSS = 0.02  # 2% maximum daily loss
MAX_POSITION_SIZE_PERCENT = 0.02  # Maximum position size as percentage of portfolio
MAX_OPEN_POSITIONS = 5  # Maximum number of open positions
