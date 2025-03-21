import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Alpaca API Configuration
API_KEY = 'PKQGQ3901AX5VIJ1ENTO'
API_SECRET = 'I4BtwZSK9vWaqmscdamN722tgLfDYNlzm8Y4U3Z6'
BASE_URL = 'https://paper-api.alpaca.markets'

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
LOG_LEVEL = 'INFO'
LOG_FILE = 'trading.log'

# Risk Management
MAX_DAILY_LOSS = 0.02  # 2% maximum daily loss
MAX_POSITION_SIZE_PERCENT = 0.02  # Maximum position size as percentage of portfolio
MAX_OPEN_POSITIONS = 5  # Maximum number of open positions
