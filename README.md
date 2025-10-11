# ERCOT Real-Time Peak Detection System

An algorithmic trading system for detecting daily electricity price peaks in the ERCOT real-time market.

## Overview

This system analyzes 5-minute interval electricity price data to detect daily price peaks using multiple trading strategies with ensemble methods.

## Features

- **Data Pipeline**: Load and preprocess RTM (Real-Time Market) and DAM (Day-Ahead Market) data
- **Multiple Strategies**: Baseline and advanced detection strategies
- **Backtesting Framework**: Rigorous day-by-day validation with no lookahead bias
- **Performance Metrics**: Success rate, precision, false positives, signal timing
- **Ensemble Methods**: Smart voting system combining multiple strategies
- **Visualization**: Comprehensive charts and performance reports

## System Architecture

```
src/
├── data_loader.py           # Data loading and preprocessing
├── eda.py                   # Exploratory data analysis
├── strategies.py            # Baseline trading strategies
├── advanced_strategies.py   # Advanced detection strategies
├── backtester.py            # Backtesting framework
├── comparator.py            # Strategy comparison
├── parameter_tuning.py      # Parameter optimization
└── failure_analysis.py      # Error analysis
```

## Installation

```bash
# Clone repository
git clone https://github.com/Rzhan16/ercot_peak_detection.git
cd ercot_peak_detection

# Install dependencies
pip install -r requirements.txt
```

## Data Format

Place your data files in the `data/` directory:

- `rtm_prices.csv`: Real-time market prices (5-minute intervals)
  - Columns: `timestamp`, `price`
- `dam_prices.csv`: Day-ahead market forecasts (hourly)
  - Columns: `timestamp`, `price`

## Usage

### Run Complete Pipeline

```bash
python main.py --mode full
```

### Run Specific Analyses

```bash
# Exploratory Data Analysis
python main.py --mode eda

# Backtest Strategies
python main.py --mode backtest

# Parameter Optimization
python main.py --mode optimize

# Generate Reports
python main.py --mode report
```

## Strategies

### Baseline Strategies

1. **PriceDrop**: Detects drops from recent price highs
2. **VelocityReversal**: Identifies negative price acceleration
3. **DayAheadDeviation**: Compares RTM vs DAM forecasts
4. **NaiveTime**: Time-of-day based triggers

### Advanced Strategies

1. **TwoStageConfirmation**: Waits to verify peak has passed
2. **HighValuePeak**: Focuses on high-value price peaks
3. **SmartEnsemble**: Weighted voting across multiple strategies

## Performance Metrics

- **Success Rate**: Percentage of peaks caught within ±5 minutes
- **Precision**: Percentage of signals that are true peaks
- **False Positive Rate**: Invalid signals per day
- **Average Delay**: Time between signal and actual peak
- **F1-Score**: Harmonic mean of success and precision

## Results

Results are saved to the `results/` directory:

- Strategy comparison charts
- Daily performance visualizations
- Parameter optimization reports
- Failure analysis breakdowns

## Key Features

### No Lookahead Bias
- Day-by-day processing ensures no future data is used
- Each prediction uses only historically available data
- Proper train/test temporal splits

### Rigorous Validation
- 365 days of backtesting
- Multiple performance metrics
- Detailed error analysis

### Modular Design
- Clean separation of concerns
- Easy to add new strategies
- Extensible backtesting framework

## Configuration

Edit strategy parameters in the code or use command-line optimization mode:

```python
# Example: Configure PriceDrop strategy
strategy = PriceDropStrategy(
    lookback_minutes=10,
    drop_threshold=0.035,
    min_price_multiplier=1.25
)
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- tqdm
- openpyxl

## Project Structure

```
ercot_peak_detection/
├── src/                    # Source code
├── data/                   # Data files (user-provided)
├── results/                # Output visualizations and reports
├── main.py                 # Main orchestration script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## License

MIT License - See LICENSE file for details

## Contributing

This is a research/analysis project. Feel free to fork and adapt for your own use cases.

## Contact

For questions or issues, please open a GitHub issue.
