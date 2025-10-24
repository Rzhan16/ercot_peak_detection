"""
Filter signals by minimum profit spread threshold.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

def calculate_signal_spread(df: pd.DataFrame, signal_idx: int) -> float:
    """
    Calculate profit spread for a signal.
    
    Args:
        df: DataFrame with timestamp and price
        signal_idx: Index of the signal
        
    Returns:
        Signal spread (trigger_price - baseline_price)
    """
    signal_time = df.loc[signal_idx, 'timestamp']
    signal_price = df.loc[signal_idx, 'price']
    
    # Get 30 minutes before trigger
    baseline_start = signal_time - timedelta(minutes=30)
    baseline_data = df[(df['timestamp'] >= baseline_start) & (df['timestamp'] < signal_time)]
    
    if len(baseline_data) == 0:
        return 0.0
    
    baseline_price = baseline_data['price'].min()
    signal_spread = signal_price - baseline_price
    
    return signal_spread

def is_successful_new(trigger_time, trigger_price, peak_time, peak_price):
    """New success criteria: ±5 min OR ±10% price"""
    time_diff_minutes = abs((trigger_time - peak_time).total_seconds() / 60)
    time_success = time_diff_minutes <= 5
    
    price_diff_pct = abs(trigger_price - peak_price) / peak_price
    price_success = price_diff_pct <= 0.10
    
    return (time_success or price_success)

def apply_threshold_filtering(results_df: pd.DataFrame, rtm_1min: pd.DataFrame, threshold: float):
    """
    Apply profit threshold filtering to backtest results.
    
    Args:
        results_df: Backtest results with columns [date, peak_time, peak_price, num_signals, success]
        rtm_1min: 1-minute RTM data
        threshold: Minimum profit spread threshold
        
    Returns:
        Dictionary with filtered metrics
    """
    total_signals_before = 0
    total_signals_after = 0
    successful_signals = 0
    days_with_success = 0
    total_profit = 0.0
    
    for idx, row in results_df.iterrows():
        date = row['date']
        peak_time = row['peak_time']
        peak_price = row['peak_price']
        
        # Get day's data
        day_data = rtm_1min[rtm_1min['timestamp'].dt.date == date].copy()
        
        if len(day_data) == 0:
            continue
        
        # Get signals for this day (using same strategy logic)
        threshold_80pct = day_data['price'].quantile(0.80)
        signals = day_data[day_data['price'] >= threshold_80pct].copy()
        
        total_signals_before += len(signals)
        
        # Filter by profit threshold
        day_has_success = False
        for sig_idx in signals.index:
            signal_spread = calculate_signal_spread(day_data, sig_idx)
            
            if signal_spread >= threshold:
                total_signals_after += 1
                total_profit += signal_spread
                
                # Check if this signal is successful
                signal_time = day_data.loc[sig_idx, 'timestamp']
                signal_price = day_data.loc[sig_idx, 'price']
                
                if is_successful_new(signal_time, signal_price, peak_time, peak_price):
                    successful_signals += 1
                    day_has_success = True
        
        if day_has_success:
            days_with_success += 1
    
    # Calculate metrics
    signals_removed = total_signals_before - total_signals_after
    success_rate = (days_with_success / len(results_df) * 100) if len(results_df) > 0 else 0
    precision = (successful_signals / total_signals_after * 100) if total_signals_after > 0 else 0
    avg_profit = total_profit / total_signals_after if total_signals_after > 0 else 0
    
    return {
        'signals_before': total_signals_before,
        'signals_after': total_signals_after,
        'signals_removed': signals_removed,
        'success_rate': success_rate,
        'precision': precision,
        'avg_profit': avg_profit,
        'successful_signals': successful_signals
    }

# Load data
print("="*70)
print("PROFIT-THRESHOLD FILTERING ANALYSIS")
print("="*70)
print("\nLoading data...")

rtm_1min = pd.read_csv('data/rtm_prices_1min.csv')
rtm_1min['timestamp'] = pd.to_datetime(rtm_1min['timestamp'])

results_df = pd.read_csv('results/new_framework_backtest_results.csv')
results_df['date'] = pd.to_datetime(results_df['date']).dt.date
results_df['peak_time'] = pd.to_datetime(results_df['peak_time'])

# Test thresholds
thresholds = [0, 50, 80, 100, 150]

print("\nTesting profit thresholds...")
print("This may take a few minutes...\n")

results = []

for threshold in thresholds:
    print(f"Testing ${threshold} threshold...")
    metrics = apply_threshold_filtering(results_df, rtm_1min, threshold)
    
    results.append({
        'Threshold': f'${threshold}' if threshold > 0 else 'No filter',
        'Signals Kept': metrics['signals_after'],
        'Signals Removed': metrics['signals_removed'],
        'Success Rate': f"{metrics['success_rate']:.1f}%",
        'Precision': f"{metrics['precision']:.1f}%",
        'Avg Profit/Signal': f"${metrics['avg_profit']:.2f}",
        # Store numeric values for plotting
        '_signals_kept': metrics['signals_after'],
        '_precision': metrics['precision'],
        '_threshold_num': threshold
    })

# Create comparison table
print("\n" + "="*70)
print("COMPARISON TABLE")
print("="*70)
print("\n| Threshold | Signals Kept | Signals Removed | Success Rate | Precision | Avg Profit/Signal |")
print("|-----------|--------------|-----------------|--------------|-----------|-------------------|")

for r in results:
    print(f"| {r['Threshold']:9s} | {r['Signals Kept']:12d} | {r['Signals Removed']:15d} | {r['Success Rate']:12s} | {r['Precision']:9s} | {r['Avg Profit/Signal']:17s} |")

# Analyze results to find optimal threshold
# Look for best precision improvement while keeping reasonable signal count
baseline_precision = float(results[0]['Precision'].rstrip('%'))
baseline_signals = results[0]['Signals Kept']
baseline_success_rate = float(results[0]['Success Rate'].rstrip('%'))

best_threshold = None
best_score = -1

for r in results[1:]:  # Skip "No filter"
    precision = float(r['Precision'].rstrip('%'))
    signals_kept = r['Signals Kept']
    success_rate = float(r['Success Rate'].rstrip('%'))
    threshold_num = r['_threshold_num']
    
    # Score: balance precision gain with maintaining decent success rate
    # We want precision 3-4x better while keeping success rate >30%
    if success_rate >= 30.0 and signals_kept >= 200:  # Minimum viable thresholds
        precision_gain = precision - baseline_precision
        score = precision_gain * (success_rate / 100)  # Weighted by success rate
        
        if score > best_score:
            best_score = score
            best_threshold = r

# Print recommendation
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if best_threshold:
    optimal_value = best_threshold['_threshold_num']
    print(f"\nOptimal threshold: ${optimal_value}")
    print(f"\nReasoning: At the ${optimal_value} threshold, we achieve a good balance between")
    print(f"signal quality and quantity. This filters out {best_threshold['Signals Removed']} low-value")
    print(f"signals ({best_threshold['Signals Removed']/baseline_signals*100:.1f}% of total), improving precision from {results[0]['Precision']}")
    print(f"to {best_threshold['Precision']} while maintaining {best_threshold['Signals Kept']} actionable signals.")
    print(f"The average profit per remaining signal is {best_threshold['Avg Profit/Signal']}, indicating")
    print(f"meaningful trading opportunities.")
    
    print(f"\nImpact at optimal threshold (${optimal_value}):")
    print(f"  - Removed: {best_threshold['Signals Removed']} signals ({best_threshold['Signals Removed']/baseline_signals*100:.1f}% of total)")
    print(f"  - Improved precision: From {results[0]['Precision']} to {best_threshold['Precision']}")
    print(f"  - Maintained success rate: {best_threshold['Success Rate']}")
    print(f"  - Average profit per remaining signal: {best_threshold['Avg Profit/Signal']}")
else:
    print("\nNo clear optimal threshold found. Consider using no filter or adjusting thresholds.")

# Create visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS...")
print("="*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Signal count vs threshold
thresholds_plot = [r['_threshold_num'] for r in results]
signals_plot = [r['_signals_kept'] for r in results]
colors = ['gray' if r['Threshold'] == 'No filter' else 'steelblue' for r in results]

if best_threshold:
    # Highlight optimal threshold
    colors = []
    for r in results:
        if r['Threshold'] == best_threshold['Threshold']:
            colors.append('green')
        elif r['Threshold'] == 'No filter':
            colors.append('gray')
        else:
            colors.append('steelblue')

bars = ax1.bar(range(len(thresholds_plot)), signals_plot, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(thresholds_plot)))
ax1.set_xticklabels([r['Threshold'] for r in results], rotation=0)
ax1.set_xlabel('Profit Threshold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Signal Count', fontsize=12, fontweight='bold')
ax1.set_title('Signal Count vs Threshold', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

# Right: Precision vs threshold
precision_plot = [r['_precision'] for r in results]
ax2.plot(range(len(thresholds_plot)), precision_plot, 'o-', linewidth=2.5, markersize=10, 
         color='steelblue', markerfacecolor='white', markeredgewidth=2, markeredgecolor='steelblue')

# Highlight optimal
if best_threshold:
    opt_idx = next(i for i, r in enumerate(results) if r['Threshold'] == best_threshold['Threshold'])
    ax2.plot(opt_idx, precision_plot[opt_idx], 'o', markersize=15, 
             color='green', markeredgewidth=3, markeredgecolor='darkgreen', zorder=10)

ax2.set_xticks(range(len(thresholds_plot)))
ax2.set_xticklabels([r['Threshold'] for r in results], rotation=0)
ax2.set_xlabel('Profit Threshold', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax2.set_title('Precision vs Threshold', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, max(precision_plot) * 1.2)

# Add value labels
for i, val in enumerate(precision_plot):
    ax2.text(i, val + max(precision_plot)*0.03, f'{val:.1f}%',
            ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('results/plots/profit_threshold_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: results/plots/profit_threshold_analysis.png")

# Save results to CSV
results_df_save = pd.DataFrame(results)
results_df_save = results_df_save.drop(columns=['_signals_kept', '_precision', '_threshold_num'])
results_df_save.to_csv('results/profit_threshold_comparison.csv', index=False)
print("✓ Results saved: results/profit_threshold_comparison.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

