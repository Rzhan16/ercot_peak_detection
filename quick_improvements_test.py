"""Quick test of improvements - 30 days only"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from src.data_loader import ERCOTDataLoader
from src.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.improved_strategies import ImprovedRegimeAdaptiveStrategy
from src.backtester import Backtester
import logging
logging.getLogger('src.feature_engineering').setLevel(logging.WARNING)

print("\n🔬 QUICK TEST: Three Improvements\n")

# Load data
loader = ERCOTDataLoader()
rtm_df = loader.load_rtm_data('data/rtm_prices.csv')
dam_df = loader.load_dam_data('data/dam_prices.csv')
peaks_df = loader.get_daily_peaks(rtm_df)
rtm_df['date'] = rtm_df['timestamp'].dt.date

# Merge DAM
merged_df = rtm_df.merge(dam_df[['timestamp', 'dam_price']], on='timestamp', how='left')
merged_df['dam_price'].fillna(method='ffill', inplace=True)

# Test on 30 days
test_dates = peaks_df['date'].iloc[:30]
test_df = merged_df[merged_df['date'].isin(test_dates)].copy()
test_peaks = peaks_df.iloc[:30]

print(f"Testing on {len(test_peaks)} days\n")

# Baseline
print("[1/4] Baseline...")
baseline = RegimeAdaptiveStrategy(spike_dam_ratio=0.88, spike_velocity_min=60.0, spike_accel_min=-10.0)
bt_base = Backtester(baseline, test_peaks)
r_base = bt_base.run_backtest(test_df, verbose=False)
print(f"  Success: {r_base['success_rate']:.1%} | Precision: {r_base.get('precision', 0):.1%} | Signals/day: {r_base['total_signals']/30:.2f}")

# +Confirmation
print("[2/4] +Confirmation...")
v1 = ImprovedRegimeAdaptiveStrategy(spike_dam_ratio=0.88, spike_velocity_min=60.0, spike_accel_min=-10.0,
    require_confirmation=True, use_confidence_filter=False, use_price_stratification=False)
bt_v1 = Backtester(v1, test_peaks)
r_v1 = bt_v1.run_backtest(test_df, verbose=False)
print(f"  Success: {r_v1['success_rate']:.1%} | Precision: {r_v1.get('precision', 0):.1%} | Signals/day: {r_v1['total_signals']/30:.2f}")

# +Confidence
print("[3/4] +Confidence...")
v2 = ImprovedRegimeAdaptiveStrategy(spike_dam_ratio=0.88, spike_velocity_min=60.0, spike_accel_min=-10.0,
    require_confirmation=True, use_confidence_filter=True, min_confidence_threshold=50, use_price_stratification=False)
bt_v2 = Backtester(v2, test_peaks)
r_v2 = bt_v2.run_backtest(test_df, verbose=False)
print(f"  Success: {r_v2['success_rate']:.1%} | Precision: {r_v2.get('precision', 0):.1%} | Signals/day: {r_v2['total_signals']/30:.2f}")

# +Stratification
print("[4/4] +Stratification...")
v3 = ImprovedRegimeAdaptiveStrategy(spike_dam_ratio=0.88, spike_velocity_min=60.0, spike_accel_min=-10.0,
    require_confirmation=True, use_confidence_filter=True, min_confidence_threshold=50, use_price_stratification=True)
bt_v3 = Backtester(v3, test_peaks)
r_v3 = bt_v3.run_backtest(test_df, verbose=False)
print(f"  Success: {r_v3['success_rate']:.1%} | Precision: {r_v3.get('precision', 0):.1%} | Signals/day: {r_v3['total_signals']/30:.2f}")

# Summary
print(f"\n📊 IMPROVEMENT SUMMARY:")
prec_gain = (r_v3.get('precision', 0) - r_base.get('precision', 0)) * 100
print(f"  Precision: {r_base.get('precision', 0):.1%} → {r_v3.get('precision', 0):.1%}  ({prec_gain:+.1f} pp)")
print(f"  Signals/day: {r_base['total_signals']/30:.2f} → {r_v3['total_signals']/30:.2f}")
print(f"  Success: {r_base['success_rate']:.1%} → {r_v3['success_rate']:.1%}")
print("\n✅ Quick test complete!\n")
