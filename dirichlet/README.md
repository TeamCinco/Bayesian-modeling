# Bayesian Market Regime Modeling

A modular library for detecting and analyzing market regimes using Bayesian Hidden Markov Models.

## Overview

This project uses Bayesian statistics and Hidden Markov Models to identify different market regimes from financial time series data. It extracts features from price data, normalizes them, and uses a Bayesian HMM to classify market states into distinct regimes. Based on the current regime and forecasted regime transitions, it provides option trading strategy recommendations.

## Project Structure

```
Bayesian-modeling/
├── models/              # Core statistical models
│   └── bayesian_hmm.py  # Bayesian Hidden Markov Model implementation
├── features/            # Feature extraction
│   └── feature_extractor.py  # Extracts features from OHLCV data
├── data/                # Data handling
│   ├── data_loader.py   # Loads data from CSV files
│   └── feature_processor.py  # Normalizes and combines features
├── analysis/            # Analysis modules
│   ├── regime_analyzer.py  # Analyzes regime characteristics
│   └── regime_forecast.py  # Forecasts future regime probabilities
├── trading/             # Trading strategy modules
│   └── strategy_recommender.py  # Recommends option strategies
├── utils/               # Utility functions
│   └── file_io.py       # File I/O operations
└── main.py              # Main entry point
```

## Usage

```bash
# Process all CSV files in a directory
python main.py --data_dir /path/to/data --output_dir /path/to/output --n_regimes 3

# Process a single file
python main.py --single_file /path/to/file.csv --output_dir /path/to/output
```

## Features

- **Market Regime Detection**: Identifies distinct market regimes (e.g., low volatility bullish, high volatility bearish)
- **Regime Statistics**: Calculates key statistics for each regime (returns, volatility, Sharpe ratio)
- **Regime Forecasting**: Forecasts probabilities of transitioning between regimes
- **Option Strategy Recommendations**: Recommends option trading strategies based on current regime
- **Cross-Asset Analysis**: Analyzes regime correlations across multiple assets

## Implementation Details

- **Bayesian HMM**: Implements a Bayesian Hidden Markov Model with priors for more robust regime detection
- **Feature Extraction**: Extracts technical features from price data (volatility, momentum, ATR, etc.)
- **Feature Normalization**: Normalizes and combines features for better model performance
- **Regime Analysis**: Analyzes and describes characteristics of each regime
- **CSV Export**: Exports analysis results to CSV files for further analysis
