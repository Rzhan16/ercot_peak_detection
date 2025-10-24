"""
Interpolate 5-minute RTM data to 1-minute resolution using linear interpolation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

# Load 5-minute data
print("Loading 5-minute RTM data...")
df = pd.read_csv('data/rtm_prices.csv')
df['interval_start_local'] = pd.to_datetime(df['interval_start_local'])
df = df.sort_values('interval_start_local')

# Keep only necessary columns
df_5min = df[['interval_start_local', 'lmp']].copy()
df_5min.columns = ['timestamp', 'price']

# Remove duplicates (keep first occurrence)
duplicates_before = df_5min.duplicated(subset='timestamp').sum()
df_5min = df_5min.drop_duplicates(subset='timestamp', keep='first')
duplicates_after = df_5min.duplicated(subset='timestamp').sum()
print(f"Duplicates removed: {duplicates_before} (verified: {duplicates_after} remaining)")

df_5min.set_index('timestamp', inplace=True)

original_rows = len(df_5min)

# Resample to 1-minute with linear interpolation
print("Interpolating to 1-minute resolution...")
df_1min = df_5min.resample('1min').interpolate(method='linear')

interpolated_rows = len(df_1min)
increase_pct = ((interpolated_rows - original_rows) / original_rows) * 100

# Verify no NaN values
nan_count = df_1min['price'].isna().sum()
if nan_count > 0:
    print(f"WARNING: Found {nan_count} NaN values!")
else:
    print("✓ No NaN values")

# Verify 5-minute points are preserved
df_5min_indices = df_5min.index
preserved_values = df_1min.loc[df_5min_indices]
max_diff = (preserved_values['price'] - df_5min['price']).abs().max()
if max_diff < 1e-10:
    print("✓ Original 5-min values exactly preserved")
else:
    print(f"WARNING: Max difference in preserved values: {max_diff}")

# Pick a random day for validation plot
all_dates = df_1min.index.date
unique_dates = pd.Series(all_dates).unique()
random_date = random.choice(unique_dates)

# Extract data for plot window (14:00-18:00)
plot_start = pd.Timestamp(f"{random_date} 14:00:00")
plot_end = pd.Timestamp(f"{random_date} 18:00:00")

df_1min_plot = df_1min.loc[plot_start:plot_end]
df_5min_plot = df_5min.loc[plot_start:plot_end]

# Create validation plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_1min_plot.index, df_1min_plot['price'], 'b-', linewidth=1.5, label='1-min interpolated', alpha=0.7)
ax.plot(df_5min_plot.index, df_5min_plot['price'], 'ro', markersize=8, label='5-min original', zorder=10)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Price ($/MWh)', fontsize=12)
ax.set_title(f'Interpolation Verification: {random_date} (14:00-18:00)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/interpolation_verification.png', dpi=300, bbox_inches='tight')
print(f"✓ Validation plot saved: results/plots/interpolation_verification.png")

# Save interpolated data
df_1min.reset_index(inplace=True)
df_1min.to_csv('data/rtm_prices_1min.csv', index=False)
print("✓ File saved: data/rtm_prices_1min.csv")

# Print validation output
print("\n" + "="*60)
print("INTERPOLATION COMPLETE")
print("="*60)
print(f"Original rows: {original_rows:,}")
print(f"Interpolated rows: {interpolated_rows:,} (increase: {increase_pct:.1f}%)")

# Sample verification for a 10-minute window
sample_start = pd.Timestamp(f"{random_date} 14:00:00")
sample_end = pd.Timestamp(f"{random_date} 14:10:00")
df_sample = df_1min.set_index('timestamp').loc[sample_start:sample_end]

# Get 5-min points in this window
five_min_times = [t for t in df_sample.index if t.minute % 5 == 0]
five_min_values = [df_sample.loc[t, 'price'] for t in five_min_times[:3]]

# Get some intermediate 1-min points
one_min_times = [t for t in df_sample.index if t.minute % 5 != 0]
one_min_values = [df_sample.loc[t, 'price'] for t in one_min_times[:3]]

print(f"\nSample verification ({random_date} 14:00-14:10):")
print(f"  5-min points preserved: {[f'${v:.2f}' for v in five_min_values]}")
print(f"  New 1-min points added: {[f'${v:.2f}' for v in one_min_values]}")
print(f"\nFile saved: data/rtm_prices_1min.csv")
print("="*60)

