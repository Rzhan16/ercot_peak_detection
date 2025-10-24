"""
Calculate three-tier financial metrics: Theoretical Best, Actual Captures, System Total
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Constants
TRADING_VOLUME = 100  # MWh
EXECUTION_COST = 100  # $ per false positive

def is_successful_new(trigger_time, trigger_price, peak_time, peak_price):
    """New success criteria: ±5 min OR ±10% price"""
    time_diff_minutes = abs((trigger_time - peak_time).total_seconds() / 60)
    time_success = time_diff_minutes <= 5
    
    price_diff_pct = abs(trigger_price - peak_price) / peak_price
    price_success = price_diff_pct <= 0.10
    
    return (time_success or price_success)

def calculate_financial_metrics(results_df, rtm_1min, scenario_name):
    """Calculate A, B, C metrics for given scenario"""
    
    metric_a_total = 0.0  # Theoretical best
    metric_b_total = 0.0  # Actual captures
    metric_c_total = 0.0  # System total
    
    false_positive_count = 0
    missed_peak_loss = 0.0
    
    for idx, row in results_df.iterrows():
        date = row['date']
        peak_time = row['peak_time']
        peak_price = row['peak_price']
        
        # Get day's data
        day_data = rtm_1min[rtm_1min['timestamp'].dt.date == date].copy()
        
        if len(day_data) == 0:
            continue
        
        # Calculate buy price (20th percentile of daily prices)
        buy_price = day_data['price'].quantile(0.20)
        
        # METRIC A: Theoretical Best
        daily_profit_a = (peak_price - buy_price) * TRADING_VOLUME
        metric_a_total += daily_profit_a
        
        # Get signals for this day
        threshold_80pct = day_data['price'].quantile(0.80)
        signals = day_data[day_data['price'] >= threshold_80pct]
        
        # Check if day was successful and get trigger price
        day_successful = False
        trigger_price = None
        
        for sig_idx in signals.index:
            signal_time = day_data.loc[sig_idx, 'timestamp']
            signal_price = day_data.loc[sig_idx, 'price']
            
            if is_successful_new(signal_time, signal_price, peak_time, peak_price):
                day_successful = True
                trigger_price = signal_price
                break
        
        # METRIC B: Actual Captures
        if day_successful and trigger_price is not None:
            daily_profit_b = (trigger_price - buy_price) * TRADING_VOLUME
            metric_b_total += daily_profit_b
        
        # METRIC C: System Total
        if len(signals) > 0:
            # Algorithm triggered
            if day_successful and trigger_price is not None:
                # Successful trigger
                daily_profit_c = (trigger_price - buy_price) * TRADING_VOLUME
                metric_c_total += daily_profit_c
            else:
                # False positive - use first signal price
                first_signal_price = day_data.loc[signals.index[0], 'price']
                daily_profit_c = (first_signal_price - buy_price) * TRADING_VOLUME
                daily_profit_c -= EXECUTION_COST  # Subtract false positive cost
                metric_c_total += daily_profit_c
                false_positive_count += 1
        else:
            # No trigger - use conservative estimate (mean of 15:00-20:00)
            afternoon_data = day_data[
                (day_data['timestamp'].dt.hour >= 15) & 
                (day_data['timestamp'].dt.hour <= 20)
            ]
            
            if len(afternoon_data) > 0:
                sell_price = afternoon_data['price'].mean()
            else:
                sell_price = day_data['price'].mean()
            
            daily_profit_c = (sell_price - buy_price) * TRADING_VOLUME
            metric_c_total += daily_profit_c
    
    # Calculate derived metrics
    capture_rate = (metric_b_total / metric_a_total * 100) if metric_a_total > 0 else 0
    system_efficiency = (metric_c_total / metric_a_total * 100) if metric_a_total > 0 else 0
    
    missed_peak_loss = metric_a_total - metric_b_total
    false_positive_costs = false_positive_count * EXECUTION_COST
    
    return {
        'metric_a': metric_a_total,
        'metric_b': metric_b_total,
        'metric_c': metric_c_total,
        'capture_rate': capture_rate,
        'system_efficiency': system_efficiency,
        'missed_peak_loss': missed_peak_loss,
        'false_positive_costs': false_positive_costs,
        'false_positive_count': false_positive_count
    }

# Load data
print("Loading data...")
rtm_1min = pd.read_csv('data/rtm_prices_1min.csv')
rtm_1min['timestamp'] = pd.to_datetime(rtm_1min['timestamp'])

results_df = pd.read_csv('results/new_framework_backtest_results.csv')
results_df['date'] = pd.to_datetime(results_df['date']).dt.date
results_df['peak_time'] = pd.to_datetime(results_df['peak_time'])

# Load high-value days (60-day corrected)
daily_summary = pd.read_csv('data/daily_summary_60day.csv')
daily_summary['date'] = pd.to_datetime(daily_summary['date']).dt.date
high_value_dates = daily_summary[daily_summary['is_high_value']]['date'].tolist()

# Filter for high-value days in test period
results_all = results_df.copy()
results_high_value = results_df[results_df['date'].isin(high_value_dates)].copy()

print(f"Total days: {len(results_all)}")
print(f"High-value days: {len(results_high_value)}")

# Calculate metrics
print("\nCalculating financial metrics...")
print("This may take a moment...\n")

metrics_all = calculate_financial_metrics(results_all, rtm_1min, "All Days")
metrics_high = calculate_financial_metrics(results_high_value, rtm_1min, "High-Value Days")

# Print results
print("="*70)
print("FINANCIAL ANALYSIS - ALL DAYS")
print("="*70)
print(f"Metric A (Theoretical Best):    ${metrics_all['metric_a']:,.2f}")
print(f"Metric B (Actual Captures):     ${metrics_all['metric_b']:,.2f} ({metrics_all['capture_rate']:.1f}% capture)")
print(f"Metric C (System Total):         ${metrics_all['metric_c']:,.2f} ({metrics_all['system_efficiency']:.1f}% efficiency)")
print(f"\nMoney lost to:")
print(f"  Missed peaks:                  ${metrics_all['missed_peak_loss']:,.2f}")
print(f"  False positive costs:          ${metrics_all['false_positive_costs']:,.2f}")

print("\n" + "="*70)
print("FINANCIAL ANALYSIS - HIGH-VALUE DAYS ONLY")
print("="*70)
print(f"Metric A (Theoretical Best):    ${metrics_high['metric_a']:,.2f}")
print(f"Metric B (Actual Captures):     ${metrics_high['metric_b']:,.2f} ({metrics_high['capture_rate']:.1f}% capture)")
print(f"Metric C (System Total):         ${metrics_high['metric_c']:,.2f} ({metrics_high['system_efficiency']:.1f}% efficiency)")
print(f"\nMoney lost to:")
print(f"  Missed peaks:                  ${metrics_high['missed_peak_loss']:,.2f}")
print(f"  False positive costs:          ${metrics_high['false_positive_costs']:,.2f}")

# Create visualization
print("\n" + "="*70)
print("CREATING VISUALIZATION...")
print("="*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: All days
categories = ['Metric A\n(Theoretical)', 'Metric B\n(Captures)', 'Metric C\n(System Total)']
values_all = [metrics_all['metric_a'], metrics_all['metric_b'], metrics_all['metric_c']]
colors_all = ['green', 'gold', 'tomato']

bars1 = ax1.bar(categories, values_all, color=colors_all, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Total Profit ($)', fontsize=12, fontweight='bold')
ax1.set_title('All Days (n=60)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(values_all) * 1.2)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'${height/1000:.1f}K',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Right: High-value days
values_high = [metrics_high['metric_a'], metrics_high['metric_b'], metrics_high['metric_c']]
colors_high = ['green', 'gold', 'tomato']

bars2 = ax2.bar(categories, values_high, color=colors_high, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Total Profit ($)', fontsize=12, fontweight='bold')
ax2.set_title(f'High-Value Days Only (n={len(results_high_value)})', fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(values_high) * 1.2)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'${height/1000:.1f}K',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('results/plots/financial_analysis_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: results/plots/financial_analysis_comparison.png")

# Save results to file
with open('results/financial_analysis_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("FINANCIAL ANALYSIS - ALL DAYS\n")
    f.write("="*70 + "\n")
    f.write(f"Metric A (Theoretical Best):    ${metrics_all['metric_a']:,.2f}\n")
    f.write(f"Metric B (Actual Captures):     ${metrics_all['metric_b']:,.2f} ({metrics_all['capture_rate']:.1f}% capture)\n")
    f.write(f"Metric C (System Total):         ${metrics_all['metric_c']:,.2f} ({metrics_all['system_efficiency']:.1f}% efficiency)\n")
    f.write(f"\nMoney lost to:\n")
    f.write(f"  Missed peaks:                  ${metrics_all['missed_peak_loss']:,.2f}\n")
    f.write(f"  False positive costs:          ${metrics_all['false_positive_costs']:,.2f}\n")
    f.write("\n" + "="*70 + "\n")
    f.write("FINANCIAL ANALYSIS - HIGH-VALUE DAYS ONLY\n")
    f.write("="*70 + "\n")
    f.write(f"Metric A (Theoretical Best):    ${metrics_high['metric_a']:,.2f}\n")
    f.write(f"Metric B (Actual Captures):     ${metrics_high['metric_b']:,.2f} ({metrics_high['capture_rate']:.1f}% capture)\n")
    f.write(f"Metric C (System Total):         ${metrics_high['metric_c']:,.2f} ({metrics_high['system_efficiency']:.1f}% efficiency)\n")
    f.write(f"\nMoney lost to:\n")
    f.write(f"  Missed peaks:                  ${metrics_high['missed_peak_loss']:,.2f}\n")
    f.write(f"  False positive costs:          ${metrics_high['false_positive_costs']:,.2f}\n")

print("✓ Summary saved: results/financial_analysis_summary.txt")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

