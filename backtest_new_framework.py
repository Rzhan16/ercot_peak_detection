"""
Re-run backtest with 1-min data and new success criteria (±5 min OR ±10% price).
Compare old results vs new results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sys

# Import strategy and feature engineering
from src.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.feature_engineering import DifferentialFeatures, RegimeClassifier

def is_successful_new(trigger_time, trigger_price, peak_time, peak_price):
    """New success criteria: ±5 min OR ±10% price"""
    time_diff_minutes = abs((trigger_time - peak_time).total_seconds() / 60)
    time_success = time_diff_minutes <= 5
    
    price_diff_pct = abs(trigger_price - peak_price) / peak_price
    price_success = price_diff_pct <= 0.10
    
    return (time_success or price_success)

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate performance metrics from backtest results"""
    total_days = len(results)
    successful_days = sum(1 for r in results if r['success'])
    total_signals = sum(r['num_signals'] for r in results)
    successful_signals = sum(1 for r in results if r['success'])
    
    success_rate = (successful_days / total_days * 100) if total_days > 0 else 0
    precision = (successful_signals / total_signals * 100) if total_signals > 0 else 0
    signals_per_day = total_signals / total_days if total_days > 0 else 0
    false_positive_rate = 100 - precision
    
    # Calculate average timing error for successful catches
    timing_errors = [abs(r['timing_error']) for r in results if r['success'] and r['timing_error'] is not None]
    avg_timing_error = np.mean(timing_errors) if timing_errors else None
    
    return {
        'success_rate': success_rate,
        'precision': precision,
        'signals_per_day': signals_per_day,
        'false_positive_rate': false_positive_rate,
        'avg_timing_error': avg_timing_error,
        'total_days': total_days,
        'successful_days': successful_days,
        'total_signals': total_signals
    }

def run_backtest(df: pd.DataFrame, dates: List, strategy, scenario_name: str) -> List[Dict]:
    """Run backtest for given dates"""
    results = []
    
    print(f"\nRunning backtest: {scenario_name}")
    print(f"Total days: {len(dates)}")
    
    for date in dates:
        # Get day's data
        day_data = df[df['timestamp'].dt.date == date].copy()
        
        if len(day_data) == 0:
            continue
        
        # Find actual peak
        peak_idx = day_data['price'].idxmax()
        peak_time = day_data.loc[peak_idx, 'timestamp']
        peak_price = day_data.loc[peak_idx, 'price']
        
        # Generate signals
        day_with_signals = strategy.generate_signals(day_data)
        signals = day_with_signals[day_with_signals['signal'] == 1]
        
        num_signals = len(signals)
        success = False
        timing_error = None
        
        # Check if any signal is successful
        if num_signals > 0:
            for idx, signal in signals.iterrows():
                signal_time = signal['timestamp']
                signal_price = signal['price']
                
                if is_successful_new(signal_time, signal_price, peak_time, peak_price):
                    success = True
                    timing_error = (signal_time - peak_time).total_seconds() / 60
                    break
        
        results.append({
            'date': date,
            'peak_time': peak_time,
            'peak_price': peak_price,
            'num_signals': num_signals,
            'success': success,
            'timing_error': timing_error,
            'scenario': scenario_name
        })
    
    return results

# Load data
print("="*70)
print("BACKTEST WITH NEW FRAMEWORK")
print("="*70)
print("\nLoading data...")

# Load 1-minute RTM data
rtm_1min = pd.read_csv('data/rtm_prices_1min.csv')
rtm_1min['timestamp'] = pd.to_datetime(rtm_1min['timestamp'])

# Load 5-minute RTM data for comparison
rtm_5min = pd.read_csv('data/rtm_prices.csv')
rtm_5min = rtm_5min[['interval_start_local', 'lmp']].copy()
rtm_5min.columns = ['timestamp', 'price']
rtm_5min['timestamp'] = pd.to_datetime(rtm_5min['timestamp'])
rtm_5min = rtm_5min.drop_duplicates(subset='timestamp').sort_values('timestamp')

# Load daily summary for high/low value classification (60-day corrected)
daily_summary = pd.read_csv('data/daily_summary_60day.csv')
daily_summary['date'] = pd.to_datetime(daily_summary['date']).dt.date

high_value_dates = daily_summary[daily_summary['is_high_value']]['date'].tolist()
low_value_dates = daily_summary[~daily_summary['is_high_value']]['date'].tolist()

# Use 60 test days (Jan-Feb 2024 based on FINAL_TUNED_RESULTS.md)
test_start = pd.Timestamp('2024-01-01')
test_end = pd.Timestamp('2024-03-01')

all_dates = pd.date_range(test_start, test_end, freq='D').date.tolist()[:60]

# Filter high/low value dates to test period
high_value_test = [d for d in high_value_dates if d in all_dates]
low_value_test = [d for d in low_value_dates if d in all_dates]

print(f"Test period: {test_start.date()} to {test_end.date()}")
print(f"Total test days: {len(all_dates)}")
print(f"High-value days: {len(high_value_test)}")
print(f"Low-value days: {len(low_value_test)}")

# Initialize strategy (use tuned parameters from FINAL_TUNED_RESULTS.md)
strategy = RegimeAdaptiveStrategy(
    spike_dam_ratio=0.88,
    spike_velocity_min=60.0,
    spike_accel_min=-10.0,
    gradual_dam_ratio=0.93,
    gradual_velocity_min=50.0,
    normal_dam_ratio=0.91,
    normal_price_change_min=30.0,
    peak_hour_start=14,
    peak_hour_end=21
)

# Run backtests
print("\n" + "="*70)
print("RUNNING BACKTESTS...")
print("="*70)

# 1. Old baseline (5-min, all days) - from FINAL_TUNED_RESULTS.md
old_results = {
    'success_rate': 40.0,
    'precision': 15.4,
    'signals_per_day': 2.6,
    'false_positive_rate': 84.6,
    'avg_timing_error': None  # Not provided in original
}

# 2. New (1-min, all days)
results_1min_all = run_backtest(rtm_1min, all_dates, strategy, "1-min All Days")
metrics_1min_all = calculate_metrics(results_1min_all)

# 3. New (1-min, high-value days)
results_1min_high = run_backtest(rtm_1min, high_value_test, strategy, "1-min High-Value")
metrics_1min_high = calculate_metrics(results_1min_high)

# 4. New (1-min, low-value days)
results_1min_low = run_backtest(rtm_1min, low_value_test, strategy, "1-min Low-Value")
metrics_1min_low = calculate_metrics(results_1min_low)

# Create comparison table
print("\n" + "="*70)
print("BACKTEST RESULTS - NEW FRAMEWORK")
print("="*70)
print("\n| Metric | Old (5-min, all days) | New (1-min, all days) | New (1-min, high-value) | New (1-min, low-value) |")
print("|--------|----------------------|---------------------|------------------------|----------------------|")
print(f"| Success Rate | {old_results['success_rate']:.1f}% | {metrics_1min_all['success_rate']:.1f}% | {metrics_1min_high['success_rate']:.1f}% | {metrics_1min_low['success_rate']:.1f}% |")
print(f"| Precision | {old_results['precision']:.1f}% | {metrics_1min_all['precision']:.1f}% | {metrics_1min_high['precision']:.1f}% | {metrics_1min_low['precision']:.1f}% |")
print(f"| Signals/Day | {old_results['signals_per_day']:.1f} | {metrics_1min_all['signals_per_day']:.1f} | {metrics_1min_high['signals_per_day']:.1f} | {metrics_1min_low['signals_per_day']:.1f} |")
print(f"| False Positive Rate | {old_results['false_positive_rate']:.1f}% | {metrics_1min_all['false_positive_rate']:.1f}% | {metrics_1min_high['false_positive_rate']:.1f}% | {metrics_1min_low['false_positive_rate']:.1f}% |")

timing_old = "N/A"
timing_all = f"{metrics_1min_all['avg_timing_error']:.1f} min" if metrics_1min_all['avg_timing_error'] else "N/A"
timing_high = f"{metrics_1min_high['avg_timing_error']:.1f} min" if metrics_1min_high['avg_timing_error'] else "N/A"
timing_low = f"{metrics_1min_low['avg_timing_error']:.1f} min" if metrics_1min_low['avg_timing_error'] else "N/A"

print(f"| Avg Timing Error | {timing_old} | {timing_all} | {timing_high} | {timing_low} |")

# Key findings
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

change_all = metrics_1min_all['success_rate'] - old_results['success_rate']
print(f"\n1-min data impact: {change_all:+.1f}% change in success rate")
print(f"High-value day performance: {metrics_1min_high['success_rate']:.1f}% success rate")
print(f"Low-value day performance: {metrics_1min_low['success_rate']:.1f}% success rate")

print("\nHypothesis validation:")
if metrics_1min_all['success_rate'] > old_results['success_rate']:
    print("  ✓ 1-min data IMPROVED performance")
else:
    print("  ✗ 1-min data DID NOT improve performance")

if metrics_1min_high['success_rate'] >= 70:
    print("  ✓ High-value days achieved 70-80% success rate target")
else:
    print(f"  ✗ High-value days below target (got {metrics_1min_high['success_rate']:.1f}%, need 70%)")

if metrics_1min_high['success_rate'] > metrics_1min_low['success_rate']:
    print("  ✓ High-value days outperform low-value days")
else:
    print("  ✗ High-value days do not outperform low-value days")

print("\nTop 5 improvements from new framework:")
improvements = []

if metrics_1min_all['success_rate'] > old_results['success_rate']:
    improvements.append(f"Success rate increased by {change_all:.1f}%")
    
if metrics_1min_high['success_rate'] > 60:
    improvements.append(f"High-value days show strong {metrics_1min_high['success_rate']:.1f}% success rate")

if metrics_1min_all['precision'] > old_results['precision']:
    improvements.append(f"Precision improved from {old_results['precision']:.1f}% to {metrics_1min_all['precision']:.1f}%")

if metrics_1min_all['signals_per_day'] < old_results['signals_per_day']:
    improvements.append(f"Reduced signal noise ({metrics_1min_all['signals_per_day']:.1f} vs {old_results['signals_per_day']:.1f} signals/day)")
    
improvements.append(f"New success criteria captures value-based triggers (±10% price tolerance)")

for i, imp in enumerate(improvements[:5], 1):
    print(f"{i}. {imp}")

# Create visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS...")
print("="*70)

# 1. Bar chart: Success rates
fig, ax = plt.subplots(figsize=(12, 6))
scenarios = ['Old\n(5-min, all)', 'New\n(1-min, all)', 'New\n(1-min, high-value)', 'New\n(1-min, low-value)']
success_rates = [
    old_results['success_rate'],
    metrics_1min_all['success_rate'],
    metrics_1min_high['success_rate'],
    metrics_1min_low['success_rate']
]
colors = ['gray', 'steelblue', 'green', 'orange']

bars = ax.bar(scenarios, success_rates, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Success Rate Comparison Across Scenarios', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('results/plots/new_framework_success_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Bar chart saved: results/plots/new_framework_success_comparison.png")

# 2. Scatter plot: All 60 days
fig, ax = plt.subplots(figsize=(14, 6))

# Combine all 1-min results
all_1min_results = results_1min_all

dates_plot = [r['date'] for r in all_1min_results]
prices_plot = [r['peak_price'] for r in all_1min_results]
success_plot = [r['success'] for r in all_1min_results]
scenario_plot = []

for r in all_1min_results:
    if r['date'] in high_value_test:
        scenario_plot.append('High-Value')
    else:
        scenario_plot.append('Low-Value')

# Create DataFrame for easier plotting
plot_df = pd.DataFrame({
    'date': dates_plot,
    'peak_price': prices_plot,
    'success': success_plot,
    'scenario': scenario_plot
})

# Plot
for scenario, marker, color in [('High-Value', 's', 'green'), ('Low-Value', 'o', 'orange')]:
    subset = plot_df[plot_df['scenario'] == scenario]
    
    # Success
    success_subset = subset[subset['success']]
    ax.scatter(success_subset['date'], success_subset['peak_price'], 
               marker=marker, s=100, color=color, alpha=0.7, 
               label=f'{scenario} (Success)', edgecolor='black', linewidth=1)
    
    # Failure
    fail_subset = subset[~subset['success']]
    ax.scatter(fail_subset['date'], fail_subset['peak_price'], 
               marker=marker, s=100, color=color, alpha=0.3, 
               label=f'{scenario} (Fail)', edgecolor='black', linewidth=1)

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Peak Price ($/MWh)', fontsize=12, fontweight='bold')
ax.set_title('Peak Detection Results: 60-Day Test Period', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/plots/new_framework_scatter_60days.png', dpi=300, bbox_inches='tight')
print("✓ Scatter plot saved: results/plots/new_framework_scatter_60days.png")

# Save detailed results
results_df = pd.DataFrame(all_1min_results)
results_df.to_csv('results/new_framework_backtest_results.csv', index=False)
print("\n✓ Detailed results saved: results/new_framework_backtest_results.csv")

print("\n" + "="*70)
print("BACKTEST COMPLETE")
print("="*70)

