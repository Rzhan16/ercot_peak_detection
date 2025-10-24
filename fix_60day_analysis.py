"""
Fix: Re-run high-value day analysis on ONLY the 60-day test period (Jan-Feb 2024)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("FIXING DATA MISMATCH - 60-DAY PERIOD ONLY")
print("="*70)

# Load 1-minute data
print("\nLoading 1-minute RTM data...")
df = pd.read_csv('data/rtm_prices_1min.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Filter to 60-day test period (Jan 1 - Feb 29, 2024)
test_start = pd.Timestamp('2024-01-01')
test_end = pd.Timestamp('2024-03-01')  # Exclusive

df_60day = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)].copy()

print(f"Original data: {len(df):,} rows")
print(f"60-day period: {len(df_60day):,} rows")
print(f"Date range: {df_60day['timestamp'].min().date()} to {df_60day['timestamp'].max().date()}")

# Calculate daily spreads for 60-day period
print("\nCalculating daily spreads for 60 days...")
daily_stats = []

for date, group in df_60day.groupby(df_60day['timestamp'].dt.date):
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
ax.hist(daily_df['daily_spread'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
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
ax.set_title('Daily Spread Distribution (60-Day Test Period)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/daily_spread_distribution_60day.png', dpi=300, bbox_inches='tight')
print("✓ Histogram saved: results/plots/daily_spread_distribution_60day.png")

# Save files
daily_df.to_csv('data/daily_summary_60day.csv', index=False)
print("✓ File saved: data/daily_summary_60day.csv")

high_value_df = daily_df[daily_df['is_high_value']].copy()
high_value_df.to_csv('data/high_value_days_60day.csv', index=False)
print("✓ File saved: data/high_value_days_60day.csv")

# Get top 5 high-value days
top_5 = daily_df.nlargest(5, 'daily_spread')[['date', 'daily_spread']]

# Print output
print("\n" + "="*70)
print("HIGH-VALUE DAY ANALYSIS (60-DAY TEST PERIOD)")
print("="*70)
print(f"Test period: {test_start.date()} to {(test_end - pd.Timedelta(days=1)).date()}")
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
print(f"  - data/daily_summary_60day.csv")
print(f"  - data/high_value_days_60day.csv")
print("="*70)

# Show high-value dates
if high_value_days > 0:
    print(f"\nHigh-value dates in 60-day period:")
    for idx, row in high_value_df.iterrows():
        print(f"  {row['date']}: Spread ${row['daily_spread']:.2f}, Peak ${row['daily_peak']:.2f}")
else:
    print("\n⚠️  WARNING: No high-value days found in 60-day period!")
    print("    This is expected for Jan-Feb (winter months)")

print("\n" + "="*70)
print("CORRECTION COMPLETE")
print("="*70)
print("\nNext: Re-run Tasks 2.1-2.3 using these corrected files.")

