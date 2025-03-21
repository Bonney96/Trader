# Alpha Trader

An advanced algorithmic trading system for generating alpha through machine learning, sentiment analysis, and technical indicators.

## Overview

Alpha Trader is a comprehensive algorithmic trading platform that implements multiple alpha-generating strategies, with a focus on:

1. **Machine Learning Alpha**: Predicts price movements using technical features and machine learning
2. **Sentiment Analysis Alpha**: Leverages news and social media sentiment for trading signals
3. **Technical Indicator Alpha**: Implements classic trading strategies like moving average crossovers

The system integrates with Alpaca Markets API for paper/live trading and provides robust backtesting capabilities.

## Features

- **Multiple Alpha Strategies**:
  - Machine Learning strategy using Random Forest classification
  - Sentiment Analysis using news and social media data
  - Moving Average Crossover and other technical strategies

- **Risk Management**:
  - Position sizing based on model confidence
  - Automatic stop-loss and take-profit orders
  - Portfolio diversification controls
  - Maximum drawdown protection

- **Performance Tracking**:
  - Real-time portfolio monitoring
  - Strategy performance metrics
  - Comprehensive backtesting

- **Modular Architecture**:
  - Easily add new strategies
  - Configure strategies via JSON
  - Run strategies in parallel

## Getting Started

### Prerequisites

- Python 3.8+
- Alpaca Markets API key
- (Optional) News API and Finnhub API keys for sentiment analysis

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/alpha-trader.git
   cd alpha-trader
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure your API keys:
   - Copy `.env.example` to `.env`
   - Add your Alpaca API keys and other credentials

### Configuration

The system is configured via `alpha_config.json`. Key configuration options:

```json
{
  "ml": {
    "enabled": true,
    "lookback_period": 30,
    "prediction_horizon": 5
  },
  "sentiment": {
    "enabled": true,
    "sentiment_lookback": 3
  },
  "ma_crossover": {
    "enabled": true,
    "short_window": 20,
    "long_window": 50
  }
}
```

See the complete `alpha_config.json` for all configuration options.

### Running the System

#### Live Trading

```
python alpha_trader.py --symbols AAPL,MSFT,GOOG --config alpha_config.json
```

#### Backtesting

```
python strategies/backtest_alpha_strategy.py --symbol AAPL --start_date 2022-01-01 --end_date 2023-01-01 --strategy all
```

## System Architecture

### Modules

- **alpha_trader.py**: Main executor that manages all strategies
- **connection.py**: Handles API connectivity to Alpaca
- **config.py**: System-wide configuration settings
- **strategies/**:
  - **alpha_ml_strategy.py**: Machine learning-based alpha generation
  - **sentiment_alpha_strategy.py**: News and social sentiment analysis
  - **moving_average_crossover.py**: Technical indicator strategy
  - **backtest_alpha_strategy.py**: Backtesting framework

### Data Flow

1. Market data is fetched from Alpaca API
2. Data preprocessing and feature engineering
3. Alpha signal generation from multiple strategies
4. Risk management and position sizing
5. Order execution
6. Performance monitoring and logging

## Extending the System

### Adding a New Strategy

1. Create a new strategy file in the `strategies/` directory
2. Implement the strategy class with at least:
   - `__init__` method with API and symbol parameters
   - `run()` method for continuous execution
3. Add strategy configuration to `alpha_config.json`
4. Import and register the strategy in `alpha_trader.py`

### Customizing Risk Management

Modify `risk_management` section in `alpha_config.json`:

```json
"risk_management": {
  "max_position_size_percent": 0.05,
  "max_open_positions": 5,
  "max_daily_loss": 0.02,
  "stop_loss_percent": 0.02,
  "take_profit_percent": 0.04
}
```

## Performance Analysis

Backtest results are saved in the `results/` directory:
- Equity curves
- Performance metrics (Sharpe ratio, drawdown, win rate)
- Feature importance for ML strategies

Live trading performance is tracked in the `performance/` directory:
- Portfolio value history
- Daily returns
- Drawdown history

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. 