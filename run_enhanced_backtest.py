"""Comprehensive backtest of enhanced strategies with regime classification and dynamic thresholds."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import sys

# Add src to path
sys.path.insert(0, 'src')

from data_loader import ERCOTDataLoader
from strategies import PriceDropStrategy, VelocityReversalStrategy, EnsembleStrategy
from regime_adaptive_strategy import RegimeAdaptiveStrategy, DynamicThresholdStrategy
from backtester import Backtester

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def run_comprehensive_comparison():
    """Run comprehensive comparison of all strategies including enhancements."""
    print("=" * 80)
    print("COMPREHENSIVE ENHANCED STRATEGY BACKTEST")
    print("=" * 80)
    print()
    
    # Load data
    print("📊 Loading ERCOT data...")
    loader = ERCOTDataLoader()
    rtm_df = loader.load_rtm_data('data/rtm_prices.csv')
    peaks_df = loader.get_daily_peaks(rtm_df)
    
    print(f"   ✓ Loaded {len(rtm_df):,} RTM records")
    print(f"   ✓ Extracted {len(peaks_df)} daily peaks")
    print()
    
    # Define strategies to test
    print("🎯 Initializing strategies...")
    
    strategies = {
        # Baseline (for comparison)
        'PriceDrop (Optimized)': PriceDropStrategy(
            lookback_minutes=10,
            drop_threshold=0.035,
            min_price_multiplier=1.25,
            longterm_minutes=120
        ),
        
        'VelocityReversal (Optimized)': VelocityReversalStrategy(
            velocity_window_minutes=15,
            acceleration_threshold=-1.0,
            price_percentile=80,
            lookback_minutes=60
        ),
        
        'Ensemble (Baseline)': EnsembleStrategy([
            PriceDropStrategy(
                lookback_minutes=10,
                drop_threshold=0.035,
                min_price_multiplier=1.25,
                longterm_minutes=120
            ),
            VelocityReversalStrategy(
                velocity_window_minutes=15,
                acceleration_threshold=-1.0,
                price_percentile=80,
                lookback_minutes=60
            )
        ], min_votes=2),
        
        # NEW: Enhanced strategies
        'RegimeAdaptive': RegimeAdaptiveStrategy(
            spike_dam_ratio=0.88,
            spike_velocity_min=60.0,
            spike_accel_min=-5.0,
            gradual_dam_ratio=0.95,
            gradual_velocity_min=30.0,
            gradual_price_multiplier=1.08,
            normal_dam_ratio=0.92,
            normal_price_change_min=40.0,
            peak_hour_start=14,
            peak_hour_end=21
        ),
        
        'DynamicThreshold': DynamicThresholdStrategy(
            base_dam_ratio=0.93,
            base_velocity_min=40.0,
            base_drop_threshold=0.03,
            summer_adjustment=0.03,
            winter_adjustment=-0.02,
            early_hour_adjustment=-0.02,
            mid_hour_adjustment=0.0,
            late_hour_adjustment=0.02,
            learning_window_days=90,
            peak_hour_start=15,
            peak_hour_end=20
        ),
        
        # NEW: Enhanced Ensemble
        'Enhanced Ensemble': EnsembleStrategy([
            RegimeAdaptiveStrategy(),
            DynamicThresholdStrategy(),
            PriceDropStrategy(
                lookback_minutes=10,
                drop_threshold=0.035,
                min_price_multiplier=1.25,
                longterm_minutes=120
            )
        ], min_votes=2)
    }
    
    print(f"   ✓ Initialized {len(strategies)} strategies")
    print()
    
    # Run backtests
    print("🚀 Running backtests...")
    print()
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"Testing: {name}")
        print("-" * 60)
        
        try:
            backtester = Backtester(strategy, peaks_df)
            result = backtester.run_backtest(rtm_df, verbose=False)
            results[name] = result
            
            # Print summary
            print(f"   Success Rate:  {result['success_rate']:.1%}")
            print(f"   Precision:     {result['precision']:.1%}")
            print(f"   Signals/Day:   {result['signals_per_day']:.1f}")
            print(f"   Avg Delay:     {result['avg_delay_minutes']:.2f} min")
            print(f"   False Pos/Day: {result['false_positives_per_day']:.1f}")
            print("   ✓ Complete")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            results[name] = None
        
        print()
    
    # Create comparison table
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print()
    
    # Build comparison DataFrame
    comparison_data = []
    for name, result in results.items():
        if result is not None:
            comparison_data.append({
                'Strategy': name,
                'Success Rate': f"{result['success_rate']:.1%}",
                'Precision': f"{result['precision']:.1%}",
                'Signals/Day': f"{result['signals_per_day']:.1f}",
                'Avg Delay (min)': f"{result['avg_delay_minutes']:.2f}",
                'FP/Day': f"{result['false_positives_per_day']:.1f}"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()
    
    # Save results
    print("💾 Saving results...")
    
    # Save as JSON
    results_serializable = {}
    for name, result in results.items():
        if result is not None:
            results_serializable[name] = {
                'success_rate': result['success_rate'],
                'precision': result['precision'],
                'signals_per_day': result['signals_per_day'],
                'avg_delay_minutes': result['avg_delay_minutes'],
                'false_positives_per_day': result['false_positives_per_day'],
                'total_signals': result['total_signals'],
                'successful_days': result['successful_days']
            }
    
    with open('results/enhanced_strategies_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print("   ✓ Saved: results/enhanced_strategies_results.json")
    
    # Save comparison table
    comparison_df.to_csv('results/enhanced_strategies_comparison.csv', index=False)
    print("   ✓ Saved: results/enhanced_strategies_comparison.csv")
    print()
    
    # Create visualizations
    print("📊 Creating visualizations...")
    create_comparison_visualizations(results)
    print("   ✓ Saved: results/plots/enhanced_strategies_comparison.png")
    print()
    
    # Highlight improvements
    print("=" * 80)
    print("🎉 KEY IMPROVEMENTS")
    print("=" * 80)
    print()
    
    if results.get('Ensemble (Baseline)') and results.get('Enhanced Ensemble'):
        baseline_success = results['Ensemble (Baseline)']['success_rate']
        baseline_precision = results['Ensemble (Baseline)']['precision']
        
        enhanced_success = results['Enhanced Ensemble']['success_rate']
        enhanced_precision = results['Enhanced Ensemble']['precision']
        
        success_improvement = ((enhanced_success - baseline_success) / baseline_success * 100)
        precision_improvement = ((enhanced_precision - baseline_precision) / baseline_precision * 100)
        
        print(f"Baseline Ensemble:  {baseline_success:.1%} success, {baseline_precision:.1%} precision")
        print(f"Enhanced Ensemble:  {enhanced_success:.1%} success, {enhanced_precision:.1%} precision")
        print()
        print(f"Success Rate:  {success_improvement:+.1f}% improvement")
        print(f"Precision:     {precision_improvement:+.1f}% improvement")
        print()
    
    # Find best strategy
    best_strategy = max(results.items(), 
                       key=lambda x: x[1]['precision'] if x[1] else 0,
                       default=(None, None))
    
    if best_strategy[0] and best_strategy[1]:
        print(f"🏆 BEST PRECISION: {best_strategy[0]}")
        print(f"   {best_strategy[1]['precision']:.1%} precision")
        print(f"   {best_strategy[1]['success_rate']:.1%} success rate")
        print()
    
    print("=" * 80)
    print("✅ COMPREHENSIVE BACKTEST COMPLETE!")
    print("=" * 80)
    
    return results


def create_comparison_visualizations(results):
    """Create comprehensive comparison visualizations."""
    
    # Filter out None results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("   No valid results to visualize")
        return
    
    # Prepare data
    strategies = list(valid_results.keys())
    success_rates = [valid_results[s]['success_rate'] * 100 for s in strategies]
    precisions = [valid_results[s]['precision'] * 100 for s in strategies]
    signals_per_day = [valid_results[s]['signals_per_day'] for s in strategies]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define color scheme
    colors = []
    for s in strategies:
        if 'Enhanced' in s or 'Regime' in s or 'Dynamic' in s:
            colors.append('#2ecc71')  # Green for new strategies
        else:
            colors.append('#3498db')  # Blue for baseline
    
    # Plot 1: Success Rate
    axes[0, 0].barh(strategies, success_rates, color=colors, alpha=0.8)
    axes[0, 0].set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].axvline(x=60, color='red', linestyle='--', alpha=0.5, label='Target (60%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(success_rates):
        axes[0, 0].text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # Plot 2: Precision
    axes[0, 1].barh(strategies, precisions, color=colors, alpha=0.8)
    axes[0, 1].set_xlabel('Precision (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Precision Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Target (10%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(precisions):
        axes[0, 1].text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    # Plot 3: Signals per Day
    axes[1, 0].barh(strategies, signals_per_day, color=colors, alpha=0.8)
    axes[1, 0].set_xlabel('Signals per Day', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Signal Frequency', fontsize=14, fontweight='bold')
    axes[1, 0].axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Threshold (20/day)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(signals_per_day):
        axes[1, 0].text(v + 0.5, i, f'{v:.1f}', va='center', fontweight='bold')
    
    # Plot 4: Success vs Precision Scatter
    axes[1, 1].scatter(success_rates, precisions, s=200, c=colors, alpha=0.8, edgecolors='black', linewidths=2)
    
    # Add strategy labels
    for i, s in enumerate(strategies):
        label = s.replace(' (Optimized)', '').replace(' (Baseline)', '')
        if len(label) > 15:
            label = label[:15] + '...'
        axes[1, 1].annotate(label, (success_rates[i], precisions[i]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')
    
    axes[1, 1].set_xlabel('Success Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Success vs Precision Trade-off', fontsize=14, fontweight='bold')
    axes[1, 1].axvline(x=60, color='red', linestyle='--', alpha=0.3)
    axes[1, 1].axhline(y=10, color='red', linestyle='--', alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add target zone
    axes[1, 1].fill_between([60, 100], 10, 30, alpha=0.1, color='green', label='Target Zone')
    axes[1, 1].legend()
    
    plt.suptitle('Enhanced Strategies Performance Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('results/plots/enhanced_strategies_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "ENHANCED PEAK DETECTION BACKTEST" + " " * 26 + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Features:" + " " * 68 + "║")
    print("║" + "    • Multi-scale differential features (velocity, acceleration)" + " " * 14 + "║")
    print("║" + "    • Regime classification (extreme_spike, gradual_rise, etc.)" + " " * 12 + "║")
    print("║" + "    • Regime-adaptive triggers" + " " * 48 + "║")
    print("║" + "    • Dynamic thresholds (seasonal, hourly)" + " " * 38 + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    results = run_comprehensive_comparison()
    
    print("\n🎯 Next Steps:")
    print("   1. Review results in: results/enhanced_strategies_comparison.csv")
    print("   2. View visualization: results/plots/enhanced_strategies_comparison.png")
    print("   3. Check detailed data: results/enhanced_strategies_results.json")
    print("\n")

