"""Simple full year test to get clean results."""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import sys

# Suppress all logging
import logging
logging.getLogger().setLevel(logging.ERROR)

from src.data_loader import ERCOTDataLoader
from src.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.backtester import Backtester

print("="*80)
print("FULL YEAR TEST (365 days)")
print("="*80)

# Load data
loader = ERCOTDataLoader()
rtm_df = loader.load_rtm_data('data/rtm_prices.csv')
peaks_df = loader.get_daily_peaks(rtm_df)

print(f"✓ Loaded {len(rtm_df):,} RTM records")
print(f"✓ Testing on {len(peaks_df)} days (full year 2024)")
print()

# Test RegimeAdaptive strategy
strategy = RegimeAdaptiveStrategy(
    spike_dam_ratio=0.88,
    spike_velocity_min=60.0,
    spike_accel_min=-10.0
)

backtester = Backtester(strategy, peaks_df)
result = backtester.run_backtest(rtm_df, verbose=False)

print("="*80)
print("FULL YEAR RESULTS")
print("="*80)
print(f"Success Rate:  {result['success_rate']:.1%}")
print(f"Precision:     {result.get('precision', 0):.1%}")
print(f"Signals/Day:   {result['total_signals'] / 365:.1f}")
print(f"Total Signals: {result['total_signals']:,}")
print(f"Avg Delay:     {result.get('avg_delay_minutes', 0):.1f} min")
print()

# Compare to 60-day results
print("="*80)
print("COMPARISON: 60-day vs 365-day")
print("="*80)
print("60-day results (from earlier):")
print("  Success: 40.0%")
print("  Precision: 15.4%")
print("  Signals/Day: 2.6")
print()
print("365-day results (current):")
print(f"  Success: {result['success_rate']:.1%}")
print(f"  Precision: {result.get('precision', 0):.1%}")
print(f"  Signals/Day: {result['total_signals'] / 365:.1f}")
print()

# Calculate differences
success_diff = result['success_rate'] - 0.40
precision_diff = result.get('precision', 0) - 0.154
signals_diff = (result['total_signals'] / 365) - 2.6

print("Differences (365-day - 60-day):")
print(f"  Success: {success_diff:+.1%}")
print(f"  Precision: {precision_diff:+.1%}")
print(f"  Signals/Day: {signals_diff:+.1f}")
print()

if abs(success_diff) < 0.05 and abs(precision_diff) < 0.02:
    print("✅ Results are consistent! 60-day test was representative.")
else:
    print("⚠️  Results differ significantly. 60-day test may not be representative.")
