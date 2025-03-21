# Trader - Algorithmic Trading Strategy Backtester

This repository contains a backtesting framework for comparing different trading strategies.

## Supported Strategies

1. **Dollar-Cost Averaging (DCA)** - Invests a fixed amount at regular intervals
2. **Value Averaging (VA)** - Adjusts investment amounts to reach target portfolio growth
3. **Moving Average Crossover (MAC)** - Uses technical analysis to identify trends
4. **Advanced Strategy** - Combines multiple technical indicators with risk management

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Trader.git
cd Trader
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up API credentials

Create a `.env` file in the root directory based on the provided `.env.example`:

```bash
cp .env.example .env
```

Edit the `.env` file and add your Alpaca API credentials:

```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

You can get your API keys by signing up at [Alpaca](https://app.alpaca.markets/signup).

### 4. Run the backtest

```bash
python3 test_strategy_comparison.py
```

## Results

The backtest results will be saved in the `results` directory, including:
- Portfolio value CSV files
- Trade history CSV files
- Strategy comparison plots
- Individual strategy performance plots

## Configuration

You can modify various parameters in the `config.py` file:
- API configuration
- Trading parameters
- Risk management settings
- Logging configuration 