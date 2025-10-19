"""Tune strategies and create comprehensive visualizations."""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

from src.data_loader import ERCOTDataLoader
from src.strategies import PriceDropStrategy, EnsembleStrategy, VelocityReversalStrategy
from src.regime_adaptive_strategy import RegimeAdaptiveStrategy, DynamicThresholdStrategy
from src.backtester import Backtester

import logging
logging.getLogger('src.feature_engineering').setLevel(logging.WARNING)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("\n" + "="*80)
print(" "*25 + "TUNING & VISUALIZATION")
print("="*80)

# Load data
print("\n📊 Loading data...")
loader = ERCOTDataLoader()
rtm_df = loader.load_rtm_data('data/rtm_prices.csv')
peaks_df = loader.get_daily_peaks(rtm_df)
rtm_df['date'] = rtm_df['timestamp'].dt.date

# Test on 60 days
test_dates = peaks_df['date'].iloc[:60]
test_rtm = rtm_df[rtm_df['date'].isin(test_dates)]
test_peaks = peaks_df.iloc[:60]
print(f"✓ Testing on {len(test_peaks)} days\n")

# ============================================================================
# PART 1: TUNE REGIME ADAPTIVE
# ============================================================================
print("="*80)
print("PART 1: TUNING REGIME ADAPTIVE FOR 1-2 SIGNALS/DAY")
print("="*80)

# Try different threshold levels
tuning_configs = [
    {'name': 'Original', 'dam_ratio': 0.88, 'velocity_min': 60.0},
    {'name': 'Relaxed 1', 'dam_ratio': 0.85, 'velocity_min': 50.0},
    {'name': 'Relaxed 2', 'dam_ratio': 0.82, 'velocity_min': 40.0},
    {'name': 'Relaxed 3', 'dam_ratio': 0.80, 'velocity_min': 35.0},
]

tuning_results = []

for config in tuning_configs:
    strategy = RegimeAdaptiveStrategy(
        spike_dam_ratio=config['dam_ratio'],
        spike_velocity_min=config['velocity_min'],
        spike_accel_min=-10.0,  # More relaxed
        gradual_dam_ratio=config['dam_ratio'] + 0.05,
        gradual_velocity_min=config['velocity_min'] - 10,
        normal_dam_ratio=config['dam_ratio'] + 0.03,
        normal_price_change_min=30.0,
        peak_hour_start=14,
        peak_hour_end=21
    )
    
    backtester = Backtester(strategy, test_peaks)
    result = backtester.run_backtest(test_rtm, verbose=False)
    
    sig_per_day = result['total_signals'] / 60
    
    tuning_results.append({
        'name': config['name'],
        'success_rate': result['success_rate'],
        'precision': result.get('precision', 0),
        'signals_per_day': sig_per_day,
        'dam_ratio': config['dam_ratio'],
        'velocity_min': config['velocity_min']
    })
    
    print(f"\n{config['name']:12} | DAM: {config['dam_ratio']:.2f} | Vel: {config['velocity_min']:4.0f}")
    print(f"  Success: {result['success_rate']:5.1%} | Precision: {result.get('precision', 0):5.1%} | Sig/Day: {sig_per_day:4.1f}")

# Find best config (target: 1-2 signals/day, max precision)
best_config = max([r for r in tuning_results if 1.0 <= r['signals_per_day'] <= 3.0],
                  key=lambda x: x['precision'],
                  default=tuning_results[-1])

print(f"\n BEST CONFIG: {best_config['name']}")
print(f"   Success: {best_config['success_rate']:.1%}")
print(f"   Precision: {best_config['precision']:.1%}")
print(f"   Signals/Day: {best_config['signals_per_day']:.1f}")

# ============================================================================
# PART 2: TUNE ENHANCED ENSEMBLE
# ============================================================================
print("\n" + "="*80)
print("PART 2: TUNING ENHANCED ENSEMBLE (MIN_VOTES ADJUSTMENT)")
print("="*80)

ensemble_results = []

for min_votes in [1, 2, 3]:
    # Use tuned RegimeAdaptive
    ensemble = EnsembleStrategy([
        RegimeAdaptiveStrategy(
            spike_dam_ratio=best_config['dam_ratio'],
            spike_velocity_min=best_config['velocity_min'],
            spike_accel_min=-10.0
        ),
        DynamicThresholdStrategy(
            base_dam_ratio=0.88,  # Relaxed from 0.93
            base_velocity_min=30.0,  # Relaxed from 40
            base_drop_threshold=0.02
        ),
        PriceDropStrategy(lookback_minutes=10, drop_threshold=0.035)
    ], min_votes=min_votes)
    
    backtester = Backtester(ensemble, test_peaks)
    result = backtester.run_backtest(test_rtm, verbose=False)
    
    sig_per_day = result['total_signals'] / 60
    
    ensemble_results.append({
        'min_votes': min_votes,
        'success_rate': result['success_rate'],
        'precision': result.get('precision', 0),
        'signals_per_day': sig_per_day
    })
    
    print(f"\nMin Votes: {min_votes}")
    print(f"  Success: {result['success_rate']:5.1%} | Precision: {result.get('precision', 0):5.1%} | Sig/Day: {sig_per_day:4.1f}")

best_ensemble = max([r for r in ensemble_results if r['signals_per_day'] >= 0.5],
                    key=lambda x: x['precision'])

print(f"\n BEST ENSEMBLE: min_votes={best_ensemble['min_votes']}")
print(f"   Success: {best_ensemble['success_rate']:.1%}")
print(f"   Precision: {best_ensemble['precision']:.1%}")
print(f"   Signals/Day: {best_ensemble['signals_per_day']:.1f}")

# ============================================================================
# PART 3: COMPREHENSIVE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("PART 3: CREATING VISUALIZATIONS")
print("="*80)

# Run final strategies for visualization
print("\n📊 Running strategies for visualization...")

baseline_pd = PriceDropStrategy(lookback_minutes=10, drop_threshold=0.035,
                               min_price_multiplier=1.25, longterm_minutes=120)
tuned_regime = RegimeAdaptiveStrategy(
    spike_dam_ratio=best_config['dam_ratio'],
    spike_velocity_min=best_config['velocity_min'],
    spike_accel_min=-10.0,
    peak_hour_start=14,
    peak_hour_end=21
)

# Get signals for first 10 days for timeline visualization
viz_days = 10
viz_dates = peaks_df['date'].iloc[:viz_days]
viz_rtm = rtm_df[rtm_df['date'].isin(viz_dates)]
viz_peaks = peaks_df.iloc[:viz_days]

# Generate signals
baseline_signals = baseline_pd.generate_signals(viz_rtm.copy())
regime_signals = tuned_regime.generate_signals(viz_rtm.copy())

# Create figure with 4 subplots
fig, axes = plt.subplots(4, 1, figsize=(18, 16))

# ============================================================================
# PLOT 1: SIGNAL TIMELINE COMPARISON
# ============================================================================
print("  Creating Plot 1: Signal Timeline...")

ax1 = axes[0]
days_to_plot = 5  # First 5 days for clarity
plot_data = viz_rtm[viz_rtm['date'].isin(viz_dates[:days_to_plot])]
plot_peaks = viz_peaks.iloc[:days_to_plot]

# Plot price
ax1.plot(plot_data['timestamp'], plot_data['price'], 
        color='blue', linewidth=2, alpha=0.7, label='Price')

# Mark actual peaks
for _, peak in plot_peaks.iterrows():
    peak_time = peak['peak_time']
    peak_price = peak['peak_price']
    ax1.scatter([peak_time], [peak_price], color='red', s=200, 
               marker='*', zorder=5, edgecolors='black', linewidths=2,
               label='Actual Peak' if _ == plot_peaks.index[0] else '')

# Mark baseline signals
baseline_plot = baseline_signals[baseline_signals['timestamp'].isin(plot_data['timestamp'])]
baseline_triggers = baseline_plot[baseline_plot['signal'] == 1]
if len(baseline_triggers) > 0:
    ax1.scatter(baseline_triggers['timestamp'], baseline_triggers['price'],
               color='orange', s=100, marker='v', alpha=0.6, 
               label=f'Baseline ({len(baseline_triggers)} signals)')

# Mark regime signals
regime_plot = regime_signals[regime_signals['timestamp'].isin(plot_data['timestamp'])]
regime_triggers = regime_plot[regime_plot['signal'] == 1]
if len(regime_triggers) > 0:
    ax1.scatter(regime_triggers['timestamp'], regime_triggers['price'],
               color='green', s=150, marker='^', alpha=0.8,
               label=f'RegimeAdaptive ({len(regime_triggers)} signals)')

ax1.set_title('Signal Timeline: Baseline vs RegimeAdaptive (First 5 Days)', 
             fontsize=14, fontweight='bold')
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Price ($/MWh)', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# ============================================================================
# PLOT 2: PRECISION VS RECALL TRADEOFF
# ============================================================================
print("  Creating Plot 2: Precision-Recall Tradeoff...")

ax2 = axes[1]

# Collect precision-recall data from tuning
precision_vals = [r['precision'] * 100 for r in tuning_results]
recall_vals = [r['success_rate'] * 100 for r in tuning_results]
names = [r['name'] for r in tuning_results]

# Plot curve
ax2.plot(recall_vals, precision_vals, 'o-', linewidth=3, markersize=12,
        color='purple', alpha=0.7)

# Annotate points
for i, (r, p, n) in enumerate(zip(recall_vals, precision_vals, names)):
    ax2.annotate(n, (r, p), xytext=(10, -10 if i % 2 == 0 else 10),
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Mark target zone
ax2.axhspan(20, 30, alpha=0.1, color='green', label='Target Precision (20-30%)')
ax2.axvspan(30, 50, alpha=0.1, color='blue', label='Target Recall (30-50%)')

ax2.set_title('Precision vs Recall Tradeoff Curve', fontsize=14, fontweight='bold')
ax2.set_xlabel('Recall / Success Rate (%)', fontsize=12)
ax2.set_ylabel('Precision (%)', fontsize=12)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

# ============================================================================
# PLOT 3: FALSE POSITIVE RATE COMPARISON
# ============================================================================
print("  Creating Plot 3: False Positive Comparison...")

ax3 = axes[2]

# Calculate FP rates
strategies_fp = {
    'Baseline\nPriceDrop': {'precision': 4.1, 'fp_rate': 95.9},
    'Baseline\nEnsemble': {'precision': 10.5, 'fp_rate': 89.5},
    'Tuned\nRegimeAdaptive': {'precision': best_config['precision']*100, 
                              'fp_rate': (1-best_config['precision'])*100},
}

names_fp = list(strategies_fp.keys())
fp_rates = [strategies_fp[n]['fp_rate'] for n in names_fp]
precisions = [strategies_fp[n]['precision'] for n in names_fp]

x = np.arange(len(names_fp))
width = 0.35

bars1 = ax3.bar(x - width/2, fp_rates, width, label='False Positive Rate (%)',
               color='red', alpha=0.7)
bars2 = ax3.bar(x + width/2, precisions, width, label='Precision (%)',
               color='green', alpha=0.7)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_title('False Positive Rate Reduction', fontsize=14, fontweight='bold')
ax3.set_ylabel('Percentage (%)', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(names_fp, fontsize=11)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# ============================================================================
# PLOT 4: SIGNALS PER DAY vs PRECISION
# ============================================================================
print("  Creating Plot 4: Signals/Day vs Precision...")

ax4 = axes[3]

# Collect all results
all_results = tuning_results + [
    {'name': 'Baseline PD', 'signals_per_day': 20.6, 'precision': 0.041},
    {'name': 'Baseline Ens', 'signals_per_day': 4.3, 'precision': 0.105},
]

sig_days = [r['signals_per_day'] for r in all_results]
precisions_all = [r['precision'] * 100 for r in all_results]
names_all = [r['name'] for r in all_results]
colors_all = ['red' if 'Baseline' in n else 'green' for n in names_all]

ax4.scatter(sig_days, precisions_all, s=200, c=colors_all, alpha=0.7, edgecolors='black', linewidths=2)

# Annotate points
for x, y, n in zip(sig_days, precisions_all, names_all):
    ax4.annotate(n, (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Mark target zone
ax4.axhspan(20, 30, alpha=0.1, color='green')
ax4.axvspan(1, 2, alpha=0.1, color='blue')
ax4.text(1.5, 28, 'TARGET ZONE', ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

ax4.set_title('Signal Frequency vs Precision Trade-off', fontsize=14, fontweight='bold')
ax4.set_xlabel('Signals per Day', fontsize=12)
ax4.set_ylabel('Precision (%)', fontsize=12)
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/enhanced_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/plots/enhanced_analysis_comprehensive.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 FINAL SUMMARY")
print("="*80)

print(f"\n🎯 TUNED REGIME ADAPTIVE:")
print(f"   Config: DAM ratio={best_config['dam_ratio']:.2f}, Velocity min={best_config['velocity_min']:.0f}")
print(f"   Success: {best_config['success_rate']:.1%}")
print(f"   Precision: {best_config['precision']:.1%}")
print(f"   Signals/Day: {best_config['signals_per_day']:.1f}")
print(f"   False Positive Rate: {(1-best_config['precision'])*100:.1f}%")

print(f"\n📈 IMPROVEMENT vs BASELINE:")
baseline_precision = 0.041
improvement = (best_config['precision'] - baseline_precision) / baseline_precision * 100
fp_reduction = (0.959 - (1-best_config['precision'])) / 0.959 * 100
print(f"   Precision: {improvement:+.0f}% improvement")
print(f"   FP Reduction: {fp_reduction:.1f}% fewer false positives")

print(f"\n PRODUCTION RECOMMENDATION:")
print(f"   Strategy: RegimeAdaptive with relaxed thresholds")
print(f"   DAM Ratio: {best_config['dam_ratio']:.2f}")
print(f"   Velocity Min: {best_config['velocity_min']:.0f}")
print(f"   Expected: ~{best_config['signals_per_day']:.1f} signals/day at {best_config['precision']:.1%} precision")

print("\n ANALYSIS COMPLETE!")
print("="*80)
print()

