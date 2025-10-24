"""
Update success criteria from "±5 min only" to "±5 min OR ±10% price"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def is_successful_old(trigger_time, peak_time):
    """Old criteria: ±5 minutes from peak"""
    time_diff_minutes = abs((trigger_time - peak_time).total_seconds() / 60)
    return time_diff_minutes <= 5

def is_successful_new(trigger_time, trigger_price, peak_time, peak_price):
    """New criteria: ±5 min OR ±10% price"""
    # Time-based success
    time_diff_minutes = abs((trigger_time - peak_time).total_seconds() / 60)
    time_success = time_diff_minutes <= 5
    
    # Price-based success
    price_diff_pct = abs(trigger_price - peak_price) / peak_price
    price_success = price_diff_pct <= 0.10
    
    # Success if EITHER condition met
    return (time_success or price_success)

# Load data
print("Loading data for comparison...")
rtm_df = pd.read_csv('data/rtm_prices_1min.csv')
rtm_df['timestamp'] = pd.to_datetime(rtm_df['timestamp'])

high_value_days = pd.read_csv('data/high_value_days.csv')
high_value_days['date'] = pd.to_datetime(high_value_days['date']).dt.date

# For demonstration, let's use a simple threshold strategy
# Trigger when price > 80th percentile of daily prices
print("Running comparison backtest...")

old_successes = 0
new_successes = 0
changed_cases = []

for date in high_value_days['date']:
    # Get day's data
    day_data = rtm_df[rtm_df['timestamp'].dt.date == date].copy()
    
    if len(day_data) == 0:
        continue
    
    # Find peak
    peak_idx = day_data['price'].idxmax()
    peak_time = day_data.loc[peak_idx, 'timestamp']
    peak_price = day_data.loc[peak_idx, 'price']
    
    # Simple strategy: trigger when price exceeds 80th percentile
    threshold = day_data['price'].quantile(0.80)
    triggers = day_data[day_data['price'] >= threshold].copy()
    
    if len(triggers) == 0:
        continue
    
    # Evaluate each trigger
    for idx, trigger in triggers.iterrows():
        trigger_time = trigger['timestamp']
        trigger_price = trigger['price']
        
        old_success = is_successful_old(trigger_time, peak_time)
        new_success = is_successful_new(trigger_time, trigger_price, peak_time, peak_price)
        
        if old_success:
            old_successes += 1
        if new_success:
            new_successes += 1
        
        # Track cases that changed from FAIL → SUCCESS
        if not old_success and new_success:
            time_diff_minutes = abs((trigger_time - peak_time).total_seconds() / 60)
            price_diff_pct = abs(trigger_price - peak_price) / peak_price * 100
            
            changed_cases.append({
                'date': date,
                'trigger_time': trigger_time.strftime('%H:%M:%S'),
                'trigger_price': trigger_price,
                'peak_time': peak_time.strftime('%H:%M:%S'),
                'peak_price': peak_price,
                'time_diff_min': time_diff_minutes,
                'price_diff_pct': price_diff_pct,
                'reason': f'Price within {price_diff_pct:.1f}% of peak'
            })

# Calculate improvement
additional_successes = new_successes - old_successes
if old_successes > 0:
    improvement_pct = (additional_successes / old_successes) * 100
else:
    improvement_pct = 0

# Sort changed cases by date and take top 5 examples
changed_df = pd.DataFrame(changed_cases)
if len(changed_df) > 0:
    changed_df = changed_df.sort_values('date').head(5)

# Print results
print("\n" + "="*70)
print("SUCCESS CRITERIA UPDATE")
print("="*70)
print("Old: ±5 minutes from peak (strict timing)")
print("New: ±5 minutes OR ±10% price (timing OR value)")
print()
print("Backtest results comparison:")
print(f"  Old successes: {old_successes}")
print(f"  New successes: {new_successes}")
print(f"  Additional successes: {additional_successes} (increased by {improvement_pct:.1f}%)")
print()

if len(changed_df) > 0:
    print("Example cases that changed from FAIL → SUCCESS:")
    for i, row in enumerate(changed_df.itertuples(), 1):
        print(f"  {i}. {row.date} - Trigger at {row.trigger_time} (${row.trigger_price:.2f}), " +
              f"Peak at {row.peak_time} (${row.peak_price:.2f})")
        print(f"     Time diff: {row.time_diff_min:.1f} min, Price diff: {row.price_diff_pct:.1f}%")
        print(f"     → Now SUCCESS: {row.reason}")
        print()
else:
    print("No cases changed from FAIL → SUCCESS")
    print()

print("This makes sense because: We now capture triggers that occur slightly")
print("outside the ±5 minute window but still capture the peak price value,")
print("which is what matters for trading profitability.")
print("="*70)

# Save comparison results
comparison_results = {
    'old_criteria': '±5 minutes only',
    'new_criteria': '±5 minutes OR ±10% price',
    'old_successes': old_successes,
    'new_successes': new_successes,
    'additional_successes': additional_successes,
    'improvement_pct': improvement_pct
}

with open('results/success_criteria_comparison.txt', 'w') as f:
    f.write("SUCCESS CRITERIA UPDATE\n")
    f.write("="*70 + "\n")
    f.write(f"Old: {comparison_results['old_criteria']}\n")
    f.write(f"New: {comparison_results['new_criteria']}\n\n")
    f.write("Backtest results comparison:\n")
    f.write(f"  Old successes: {comparison_results['old_successes']}\n")
    f.write(f"  New successes: {comparison_results['new_successes']}\n")
    f.write(f"  Additional successes: {comparison_results['additional_successes']} ")
    f.write(f"(increased by {comparison_results['improvement_pct']:.1f}%)\n")

if len(changed_df) > 0:
    changed_df.to_csv('results/changed_success_cases.csv', index=False)
    print("\n✓ Changed cases saved: results/changed_success_cases.csv")

print("✓ Comparison saved: results/success_criteria_comparison.txt")

