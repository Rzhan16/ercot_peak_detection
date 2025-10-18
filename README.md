# ERCOT Real-Time Peak Detection System

A sophisticated algorithmic trading system for detecting daily electricity price peaks in the ERCOT real-time market using advanced signal processing and machine learning techniques.

## Overview

This system analyzes 5-minute interval electricity price data to detect daily price peaks using multiple trading strategies, regime classification, and ensemble methods. The system implements differential geometry features, dynamic thresholds, and rigorous backtesting to achieve high precision in peak detection.

## Features

- **Data Pipeline**: Load and preprocess RTM (Real-Time Market) and DAM (Day-Ahead Market) data
- **Multi-Scale Feature Engineering**: Velocity, acceleration, and volatility features at multiple time scales
- **Regime Classification**: Automatic classification of price behavior patterns (extreme spike, gradual rise, oscillating, etc.)
- **Regime-Adaptive Strategies**: Different trigger logic based on identified market regimes
- **Dynamic Thresholds**: Seasonal and time-of-day adaptive parameters
- **Multiple Strategies**: Baseline and advanced detection strategies
- **Backtesting Framework**: Rigorous day-by-day validation with no lookahead bias
- **Performance Metrics**: Success rate, precision, false positives, signal timing
- **Ensemble Methods**: Smart voting system combining multiple strategies
- **Visualization**: Comprehensive charts and performance reports

## System Architecture

```
src/
├── data_loader.py              # Data loading and preprocessing
├── eda.py                      # Exploratory data analysis
├── strategies.py               # Baseline trading strategies
├── advanced_strategies.py      # Advanced detection strategies
├── feature_engineering.py      # Multi-scale differential features and regime classification
├── regime_adaptive_strategy.py # Regime-adaptive and dynamic threshold strategies
├── backtester.py               # Backtesting framework
├── comparator.py               # Strategy comparison
├── parameter_tuning.py         # Parameter optimization
└── failure_analysis.py         # Error analysis
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

### Run Enhanced Strategies

```bash
# Test enhanced strategies with regime classification
python main.py --mode enhanced

# Quick comprehensive test (60 days)
python comprehensive_test.py

# Parameter tuning and visualization
python tune_and_visualize.py
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

### Enhanced Strategies

1. **RegimeAdaptiveStrategy**: Applies different trigger logic based on price regime classification
2. **DynamicThresholdStrategy**: Adjusts thresholds based on seasonal patterns and time of day
3. **Enhanced Ensemble**: Combines multiple strategies with regime-aware voting

## Performance Metrics

- **Success Rate**: Percentage of peaks caught within ±5 minutes
- **Precision**: Percentage of signals that are true peaks
- **False Positive Rate**: Invalid signals per day
- **Average Delay**: Time between signal and actual peak
- **F1-Score**: Harmonic mean of success and precision

## Results

### Performance Improvements

The enhanced system achieves significant improvements over baseline strategies:

- **Precision**: 4.1% → 15.4% (+275% improvement)
- **Signal Noise**: 20.6/day → 2.6/day (-88% reduction)
- **False Positive Rate**: 95.9% → 84.6% (-12% reduction)
- **Success Rate**: Maintained at 40% with higher precision

### Output Files

Results are saved to the `results/` directory:

- Strategy comparison charts
- Daily performance visualizations
- Parameter optimization reports
- Failure analysis breakdowns
- Enhanced analysis comprehensive visualization

## Key Features

### No Lookahead Bias
- Day-by-day processing ensures no future data is used
- Each prediction uses only historically available data
- Proper train/test temporal splits

### Rigorous Validation
- 365 days of backtesting
- Multiple performance metrics
- Detailed error analysis
- No lookahead bias verification

### Advanced Feature Engineering
- Multi-scale differential features (velocity, acceleration, volatility)
- Regime classification (extreme spike, gradual rise, oscillating, normal)
- Dynamic threshold adaptation
- Time-based rolling windows

### Modular Design
- Clean separation of concerns
- Easy to add new strategies
- Extensible backtesting framework
- Comprehensive test suite

## Configuration

Edit strategy parameters in the code or use command-line optimization mode:

```python
# Example: Configure RegimeAdaptive strategy
strategy = RegimeAdaptiveStrategy(
    spike_dam_ratio=0.88,
    spike_velocity_min=60.0,
    spike_accel_min=-10.0,
    gradual_dam_ratio=0.93,
    gradual_velocity_min=50.0,
    normal_dam_ratio=0.91,
    normal_price_change_min=30.0
)

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
├── src/                           # Source code
│   ├── feature_engineering.py     # Multi-scale features and regime classification
│   ├── regime_adaptive_strategy.py # Enhanced strategies
│   └── ...                        # Other modules
├── data/                          # Data files (user-provided)
├── results/                       # Output visualizations and reports
│   └── plots/                     # Generated charts and analysis
├── main.py                        # Main orchestration script
├── comprehensive_test.py          # Quick strategy comparison
├── tune_and_visualize.py          # Parameter tuning and visualization
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## License

MIT License - See LICENSE file for details

## Contributing

This is a research/analysis project. Feel free to fork and adapt for your own use cases.

## Contact

For questions or issues, please open a GitHub issue.
