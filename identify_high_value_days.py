"""
Identify high-value days based on daily spread > $500.
Daily spread = Peak price - 20th percentile price
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load 1-minute data
print("Loading 1-minute RTM data...")
df = pd.read_csv('data/rtm_prices_1min.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Group by calendar day and calculate metrics
print("Calculating daily spreads...")
daily_stats = []

for date, group in df.groupby(df['timestamp'].dt.date):
    daily_peak = group['price'].max()
    daily_20th_pct = group['price'].quantile(0.20)
    daily_spread = daily_peak - daily_20th_pct
    is_high_value = daily_spread > 500
    
    daily_stats.append({
        'date': date,
        'daily_peak': daily_peak,
        'daily_20th_pct': daily_20th_pct,
        'daily_spread': daily_spread,
        'is_high_value': is_high_value
    })

daily_df = pd.DataFrame(daily_stats)

# Calculate statistics
total_days = len(daily_df)
high_value_days = daily_df['is_high_value'].sum()
high_value_pct = (high_value_days / total_days) * 100
low_value_days = total_days - high_value_days
low_value_pct = (low_value_days / total_days) * 100

mean_spread = daily_df['daily_spread'].mean()
median_spread = daily_df['daily_spread'].median()
max_spread = daily_df['daily_spread'].max()

# Create histogram
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(daily_df['daily_spread'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=500, color='red', linewidth=2, linestyle='--', label='$500 threshold')

# Add annotations
y_max = ax.get_ylim()[1]
ax.text(250, y_max * 0.85, f'Low-value:\n{low_value_days} days', 
        ha='center', fontsize=12, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
ax.text(750, y_max * 0.85, f'High-value:\n{high_value_days} days', 
        ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

ax.set_xlabel('Daily Spread ($)', fontsize=12)
ax.set_ylabel('Count of Days', fontsize=12)
ax.set_title('Daily Spread Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/daily_spread_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Histogram saved: results/plots/daily_spread_distribution.png")

# Save files
daily_df.to_csv('data/daily_summary.csv', index=False)
print("✓ File saved: data/daily_summary.csv")

high_value_df = daily_df[daily_df['is_high_value']].copy()
high_value_df.to_csv('data/high_value_days.csv', index=False)
print("✓ File saved: data/high_value_days.csv")

# Get top 5 high-value days
top_5 = daily_df.nlargest(5, 'daily_spread')[['date', 'daily_spread']]

# Print output
print("\n" + "="*60)
print("HIGH-VALUE DAY ANALYSIS")
print("="*60)
print(f"Total days analyzed: {total_days}")
print(f"High-value days (spread > $500): {high_value_days} ({high_value_pct:.1f}%)")
print(f"Low-value days (spread ≤ $500): {low_value_days} ({low_value_pct:.1f}%)")
print(f"\nSpread statistics:")
print(f"  Mean: ${mean_spread:.2f}")
print(f"  Median: ${median_spread:.2f}")
print(f"  Max: ${max_spread:.2f}")
print(f"\nHigh-value day examples:")
for idx, row in top_5.iterrows():
    print(f"  {row['date']}: ${row['daily_spread']:.2f}")
print(f"\nFiles saved:")
print(f"  - data/daily_summary.csv")
print(f"  - data/high_value_days.csv")
print("="*60)

